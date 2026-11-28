//! A baked lightmap reaches the frame through the frame's contracts: a room
//! whose boxes carry lightmap UVs (UV1) is baked, the graph rebuilds with the
//! bake's nodes, the meshes' G-buffer drawer writes each lightmapped pixel's
//! atlas UV as the surface payload, and the bake's irradiance node gives those
//! pixels the frame's indirect diffuse at full weight. Debug views 14 (surface
//! payload) and 27 (indirect-diffuse weight) show both.

use std::sync::Arc;

use glam::{Mat4, Vec3};
use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera,
    RendererBuilder, RendererConfig,
};
use helio_light_api::host_rows::LightComponent;
use helio_mats::host_rows::MaterialComponent;
use helio_mesh_host::{MeshComponent, StaticObjectComponent};

const SIZE: u32 = 64;
const PAYLOAD_VIEW: u32 = 14;
const WEIGHT_VIEW: u32 = 27;

fn scene_db(device: &Arc<wgpu::Device>, queue: &Arc<wgpu::Queue>) -> pulsar_scenedb::SceneDb {
    let mut scene_db = pulsar_scenedb::SceneDb::new();
    let ctx = pulsar_scenedb::gpu::EngineGpuContext::new(device.clone(), queue.clone());
    let store = pulsar_scenedb::gpu::SceneGpuStore::new(
        &ctx,
        pulsar_scenedb::gpu::SceneGpuConfig {
            classes: Vec::new(),
            tombstone_headroom: 0,
            max_cells_metadata: 0,
        },
    );
    MeshComponent::register_gpu_columns_growable(&store, 64, device);
    MaterialComponent::register_gpu_columns_growable(&store, 64, device);
    StaticObjectComponent::register_gpu_columns_growable(&store, 64, device);
    LightComponent::register_gpu_columns_growable(&store, helio_light_api::MAX_LIGHTS, device);
    scene_db
        .world
        .attach_gpu_mirror(pulsar_scenedb::gpu::GpuMirrorHandle::new(
            Arc::new(store),
            queue.clone(),
        ));
    scene_db
}

/// An axis-aligned box of half extent `half` at `centre`, its six faces laid
/// out apart in UV1 (a 3 x 2 grid with a margin): a valid lightmap layout.
fn spawn_box(world: &mut pulsar_scenedb::World, centre: Vec3, half: Vec3) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for axis in 0..3 {
        for sign in [-1.0f32, 1.0] {
            let face = axis * 2 + usize::from(sign > 0.0);
            let cell = [(face % 3) as f32 / 3.0, (face / 3) as f32 / 2.0];
            let n = Vec3::from_array(std::array::from_fn(|k| if k == axis { sign } else { 0.0 }));
            let (u, v) = (
                Vec3::from_array(std::array::from_fn(|k| f32::from(k == (axis + 1) % 3))),
                Vec3::from_array(std::array::from_fn(|k| f32::from(k == (axis + 2) % 3))),
            );
            let (u, v) = if sign > 0.0 { (u, v) } else { (v, u) };
            let base = vertices.len() as u32;
            for (a, b) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                let p = (n + u * a + v * b) * half;
                let mut vertex = helio_mesh_api::PackedVertex::from_components(
                    p.to_array(),
                    n.to_array(),
                    [0.0; 2],
                    u.to_array(),
                    1.0,
                );
                vertex.set_lightmap_uv([
                    cell[0] + (0.05 + 0.9 * (a * 0.5 + 0.5)) / 3.0,
                    cell[1] + (0.05 + 0.9 * (b * 0.5 + 0.5)) / 2.0,
                ]);
                vertices.push(vertex);
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
    let mesh = world.spawn();
    world.insert(mesh, MeshComponent { vertices, indices });
    let material = world.spawn();
    let row = MaterialComponent::from(helio_mats::MaterialRow::new(
        [0.7, 0.7, 0.7, 1.0],
        0.8,
        0.0,
        [0.0; 3],
        0.0,
    ));
    world.insert(material, row);
    let store = world.gpu_mirror().cloned().unwrap();
    let vertex_range = MeshComponent::vertices_gpu_handle(store.store(), mesh.index()).unwrap();
    let index_range = MeshComponent::indices_gpu_handle(store.store(), mesh.index()).unwrap();
    let object = StaticObjectComponent::from(helio_mesh_api::StaticObjectRow::new(
        mesh.index(),
        mesh.generation().wrapping_add(1),
        material.index(),
        material.generation().wrapping_add(1),
        Mat4::from_translation(centre),
        [centre.x, centre.y, centre.z, half.length()],
        index_range.count,
        index_range.offset,
        vertex_range.offset as i32,
        row.material_class,
        0,
        0,
    ));
    let entity = world.spawn();
    world.insert(entity, object);
}

/// A closed room (floor, ceiling, four walls) with a pillar and a point light.
fn room(world: &mut pulsar_scenedb::World) {
    let light = world.spawn();
    world.insert(
        light,
        LightComponent::from(helio_light_api::LightRow::from(helio_light_api::GpuLight {
            position_range: [0.0, 2.5, 0.0, 12.0],
            color_intensity: [1.0, 0.9, 0.8, 20.0],
            shadow_index: u32::MAX,
            light_type: 1,
            ..Default::default()
        })),
    );
    for (centre, half) in [
        (Vec3::new(0.0, -0.25, 0.0), Vec3::new(5.0, 0.25, 5.0)),
        (Vec3::new(0.0, 4.25, 0.0), Vec3::new(5.0, 0.25, 5.0)),
        (Vec3::new(-5.25, 2.0, 0.0), Vec3::new(0.25, 2.5, 5.0)),
        (Vec3::new(5.25, 2.0, 0.0), Vec3::new(0.25, 2.5, 5.0)),
        (Vec3::new(0.0, 2.0, -5.25), Vec3::new(5.0, 2.5, 0.25)),
        (Vec3::new(0.0, 2.0, 5.25), Vec3::new(5.0, 2.5, 0.25)),
        (Vec3::new(1.5, 1.0, -1.0), Vec3::new(0.4, 1.0, 0.4)),
    ] {
        spawn_box(world, centre, half);
    }
}

fn gpu() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .ok()?;
    let features = required_wgpu_features(helio::usable_adapter_features(&adapter));
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: features,
        required_limits: required_wgpu_limits(adapter.limits()),
        experimental_features: required_experimental_features(features),
        ..Default::default()
    }))
    .ok()?;
    Some((Arc::new(device), Arc::new(queue)))
}

/// Renders a few frames of `view` and reads the last back as RGBA8 pixels.
fn frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut helio::Renderer,
    scene: &mut pulsar_scenedb::SceneDb,
    debug_view: u32,
) -> Vec<[u8; 4]> {
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let camera = Camera::perspective_look_at(
        Vec3::new(-3.5, 2.0, 4.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::Y,
        1.0,
        1.0,
        0.1,
        100.0,
    );
    renderer.set_debug_mode(debug_view);
    for _ in 0..4 {
        scene.world.flush_gpu_mirror(queue);
        renderer.render(&camera, &view).unwrap();
        // Pipelines compile off the frame: wait for them between frames, as
        // a host's loading screen does.
        while helio_core::compile::in_flight() > 0 {
            std::thread::yield_now();
        }
    }
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(SIZE * SIZE * 4),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SIZE * 4),
                rows_per_image: Some(SIZE),
            },
        },
        wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let pixels =
        bytemuck::cast_slice::<u8, [u8; 4]>(&readback.slice(..).get_mapped_range().unwrap())
            .to_vec();
    pixels
}

/// Pixels the payload view shows as payload: the atlas UV in red and green,
/// no blue (its stand-in for no payload is a flat grey).
fn payload_pixels(pixels: &[[u8; 4]]) -> usize {
    pixels.iter().filter(|p| p[2] < 8).count()
}

/// Pixels whose indirect-diffuse weight (view 27's red) is full.
fn full_weight_pixels(pixels: &[[u8; 4]]) -> usize {
    pixels.iter().filter(|p| p[0] > 200).count()
}

#[test]
fn a_baked_room_reaches_the_frame_as_payload_and_indirect_diffuse() {
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut scene = scene_db(&device, &queue);
    room(&mut scene.world);
    let config = RendererConfig::new(SIZE, SIZE, wgpu::TextureFormat::Rgba8Unorm);
    let mut renderer =
        RendererBuilder::new(config.clone(), scene.world.gpu_mirror().cloned().unwrap())
            .with_graph(helio_mesh_host::presets::with(
                helio_presets::Preset::Deferred,
                &helio_bake_host::features(),
            ))
            .with_external_device()
            .build(
                device.clone(),
                queue.clone(),
                config.width,
                config.height,
                config.surface_format,
            );
    let total = (SIZE * SIZE) as usize;

    // Without a bake: no payload, no indirect diffuse.
    let before_payload = frame(&device, &queue, &mut renderer, &mut scene, PAYLOAD_VIEW);
    let before_weight = frame(&device, &queue, &mut renderer, &mut scene, WEIGHT_VIEW);
    assert!(renderer
        .find_pass::<helio_bake::BakedIrradiancePass>()
        .is_none());
    assert_eq!(payload_pixels(&before_payload), 0, "payload without a bake");
    assert_eq!(
        full_weight_pixels(&before_weight),
        0,
        "indirect diffuse without a bake"
    );

    // Bake (nothing kept between runs) and install.
    scene.world.flush_gpu_mirror(&queue);
    let scene_geometry = helio_bake_host::bake_scene_from_world(&scene.world);
    assert_eq!(scene_geometry.meshes.len(), 7);
    assert!(scene_geometry
        .meshes
        .iter()
        .all(|mesh| mesh.lightmap_uvs.is_some()));
    helio_bake_host::bake(
        &mut renderer,
        scene_geometry,
        helio_bake::BakeConfig::fast("lightmap_frame"),
    )
    .expect("bake the room");

    // The room fills the view: nearly every pixel is a lightmapped surface.
    let payload = frame(&device, &queue, &mut renderer, &mut scene, PAYLOAD_VIEW);
    let weight = frame(&device, &queue, &mut renderer, &mut scene, WEIGHT_VIEW);
    assert!(
        renderer
            .find_pass::<helio_bake::BakedIrradiancePass>()
            .is_some(),
        "the bake's irradiance node is not in the rebuilt graph"
    );
    let sample = |pixels: &[[u8; 4]]| pixels.iter().step_by(509).copied().collect::<Vec<_>>();
    let (payload_sample, weight_sample) = (sample(&payload), sample(&weight));
    let (payload, weight) = (payload_pixels(&payload), full_weight_pixels(&weight));
    assert!(
        payload > total * 9 / 10,
        "{payload} of {total} pixels carry the payload; samples {payload_sample:?}"
    );
    assert!(
        weight > total * 9 / 10,
        "{weight} of {total} pixels have full indirect weight; samples {weight_sample:?}"
    );

    let error = pollster::block_on(errors.pop());
    assert!(error.is_none(), "{error:?}");
}
