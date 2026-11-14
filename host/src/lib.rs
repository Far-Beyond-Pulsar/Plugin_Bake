//! Baking for standalone Helio hosts (the plugins' examples and tests): a
//! bake's scene from SceneDB rows ([`bake_scene_from_world`]), the bake
//! itself, installed on a renderer ([`bake`]), and the feature that renders
//! it ([`features`]). Engines bake through the same `helio-bake` calls.

use helio_mesh_api::host_rows::{MeshComponent, StaticObjectComponent};

/// Baked lighting's Helio feature, for a host's preset
/// (`helio_presets::Preset::with_features`).
pub fn features() -> helio_feature_api::FeatureSet {
    helio_feature_api::FeatureSet::new().with(helio_bake::BakeFeature)
}

/// Bake `scene` now (blocking) and render it from the next frame: installs
/// the result as the renderer's [`helio_bake::BakedLighting`], which rebuilds
/// the graph with the bake's nodes (the graph's features must include
/// [`features`]).
pub fn bake(
    renderer: &mut helio::Renderer,
    scene: helio_bake::SceneGeometry,
    config: helio_bake::BakeConfig,
) -> Result<(), helio_bake::BakeError> {
    let started = std::time::Instant::now();
    let (objects, lights) = (scene.meshes.len(), scene.lights.len());
    let baked =
        helio_bake::run_bake_blocking(renderer.device(), renderer.queue(), &scene, &config)?;
    log::info!(
        "[helio-bake] bake of '{}' done in {:.2}s: {objects} objects, {lights} lights",
        config.scene_name,
        started.elapsed().as_secs_f32(),
    );
    renderer.set_feature(helio_bake::BakedLighting::new(std::sync::Arc::new(baked)));
    Ok(())
}

/// A [`helio_bake::BakeStore`] in directory `dir`, through the engine's file
/// system (`engine_fs::virtual_fs`): one `{name}.bin` file a pass. What a
/// standalone host keeps its bakes in between runs
/// (`BakeConfig::with_store`).
pub fn disk_store(dir: impl Into<std::path::PathBuf>) -> std::sync::Arc<dyn helio_bake::BakeStore> {
    std::sync::Arc::new(DiskStore(dir.into()))
}

struct DiskStore(std::path::PathBuf);

impl helio_bake::BakeStore for DiskStore {
    fn read(&self, name: &str) -> Option<Vec<u8>> {
        engine_fs::virtual_fs::read_file(&self.0.join(format!("{name}.bin"))).ok()
    }

    fn write(&self, name: &str, bytes: &[u8]) -> std::io::Result<()> {
        engine_fs::virtual_fs::create_dir_all(&self.0)
            .and_then(|()| {
                engine_fs::virtual_fs::write_file(&self.0.join(format!("{name}.bin")), bytes)
            })
            .map_err(std::io::Error::other)
    }
}

/// A mesh with a world-space transform as the bake's input
/// ([`helio_bake::BakeMesh`]), for the object at `object_row` (its
/// `static_objects` row, the entity's index; `None`: geometry that occludes
/// and bounces light but gets no lightmap).
///
/// Positions are pre-multiplied by `transform` so the baker receives world-space
/// geometry.  Normals are rotated by the inverse-transpose to handle non-uniform
/// scaling. Use `SceneGeometry::add_mesh` to add the returned mesh to your scene.
///
/// # Example
/// ```rust,ignore
/// let mut scene = SceneGeometry::new();
/// scene.add_mesh(mesh_upload_to_bake(&box_mesh([0.0,0.0,0.0], [5.0,0.1,5.0]),
///                                    glam::Mat4::IDENTITY, Some(row)));
/// bake(&mut renderer, scene, BakeConfig::fast("my_scene").with_store(disk_store("bake_cache")))?;
/// ```
pub fn mesh_upload_to_bake(
    upload: &helio_mesh_api::MeshUpload,
    transform: glam::Mat4,
    object_row: Option<u32>,
) -> helio_bake::BakeMesh {
    fn unpack_snorm8(b: u8) -> f32 {
        (b as i8) as f32 / 127.0
    }
    let normal_mat = glam::Mat3::from_mat4(transform).inverse().transpose();

    let id = object_row.map_or(uuid::Uuid::nil(), helio_bake::bake_mesh_id);

    // The lightmap UVs are the mesh's UV1 channel, the unique [0,1] layout
    // the G-buffer drawer reads too; a mesh without one (all zero) gets no
    // lightmap (it still occludes and reflects light in the bake). Tiled
    // UV0 is not a lightmap layout: its texels would overlap.
    let has_uv1 = upload
        .vertices
        .iter()
        .any(|v| v.tex_coords1[0].abs() > 1e-4 || v.tex_coords1[1].abs() > 1e-4);
    let lightmap_uvs = has_uv1.then(|| upload.vertices.iter().map(|v| v.tex_coords1).collect());

    helio_bake::BakeMesh {
        id,
        positions: upload
            .vertices
            .iter()
            .map(|v| {
                transform
                    .transform_point3(glam::Vec3::from_array(v.position))
                    .to_array()
            })
            .collect(),
        normals: upload
            .vertices
            .iter()
            .map(|v| {
                let p = v.normal;
                let n = glam::Vec3::new(
                    unpack_snorm8(p as u8),
                    unpack_snorm8((p >> 8) as u8),
                    unpack_snorm8((p >> 16) as u8),
                );
                (normal_mat * n).normalize_or_zero().to_array()
            })
            .collect(),
        uvs: upload.vertices.iter().map(|v| v.tex_coords0).collect(),
        lightmap_uvs,
        indices: upload.indices.clone(),
        material_ids: vec![0u32; upload.indices.len() / 3],
        world_transform: Default::default(),
    }
}

/// Project a SceneDB world into the [`helio_bake::SceneGeometry`] a bake reads
/// (Helio#256): every drawn object (`StaticObjectRow` and the
/// `MeshComponent` it references, in world space, keyed by its object row)
/// and every live light (`LightRow`). The renderer itself never traverses
/// scene entities.
///
/// Call it after the scene is populated, and hand the result to [`bake`].
/// Objects whose mesh row is gone (a stale generation) are skipped.
pub fn bake_scene_from_world(world: &pulsar_scenedb::World) -> helio_bake::SceneGeometry {
    use helio_bake::{LightSource, LightSourceKind};
    use helio_light_api::host_rows::LightComponent;
    use helio_light_api::{GpuLight, LightType};
    use helio_mesh_api::MeshUpload;

    let meshes: std::collections::HashMap<u32, (u32, &MeshComponent)> = world
        .query::<&MeshComponent>()
        .map(|(entity, mesh)| (entity.index(), (entity.generation(), mesh)))
        .collect();
    let mut scene = helio_bake::SceneGeometry::new();
    for (entity, object) in world.query::<&StaticObjectComponent>() {
        let Some(&(generation, mesh)) = meshes.get(&object.mesh_slot) else {
            continue;
        };
        // Object rows store `generation + 1` so zero means "never written".
        if generation.wrapping_add(1) != object.mesh_generation {
            continue;
        }
        let upload = MeshUpload {
            vertices: mesh.vertices.clone(),
            indices: mesh.indices.clone(),
        };
        let transform = glam::Mat4::from_cols_array_2d(&object.transform);
        scene.add_mesh(mesh_upload_to_bake(
            &upload,
            transform,
            Some(entity.index()),
        ));
    }
    for (_, light) in world.query::<&LightComponent>() {
        let light = GpuLight::from(helio_light_api::LightRow::from(*light));
        if light.color_intensity[3] <= 0.0 {
            continue;
        }
        let [x, y, z, range] = light.position_range;
        let direction = [
            light.direction_outer[0],
            light.direction_outer[1],
            light.direction_outer[2],
        ];
        let kind = match light.light_type {
            t if t == LightType::Directional as u32 => LightSourceKind::Directional { direction },
            // GpuLight stores cosines; Nebula takes radians.
            t if t == LightType::Spot as u32 => LightSourceKind::Spot {
                position: [x, y, z],
                direction,
                range,
                inner_angle: light.inner_angle.clamp(-1.0, 1.0).acos(),
                outer_angle: light.direction_outer[3].clamp(-1.0, 1.0).acos(),
            },
            _ => LightSourceKind::Point {
                position: [x, y, z],
                range,
            },
        };
        scene.add_light(LightSource {
            kind,
            color: [
                light.color_intensity[0],
                light.color_intensity[1],
                light.color_intensity[2],
            ],
            intensity: light.color_intensity[3],
            bake_enabled: true,
            casts_shadows: light.shadow_index != u32::MAX,
        });
    }
    scene
}

#[cfg(test)]
mod bake_scene_tests {
    use super::*;
    use helio_bake::LightSourceKind;
    use helio_components::WorldRows;
    use helio_light_api::LightRow;
    use helio_light_api::{GpuLight, LightType};
    use helio_mesh_api::PackedVertex;
    use helio_mesh_api::StaticObjectRow;

    fn vertex(position: [f32; 3]) -> PackedVertex {
        PackedVertex {
            position,
            ..Default::default()
        }
    }

    /// Helio#256: the bake input comes from SceneDB rows, in world space,
    /// and skips what is not really there.
    #[test]
    fn projects_objects_and_live_lights_from_the_world() {
        let mut world = pulsar_scenedb::World::new();
        let mesh = world.spawn();
        world.insert(
            mesh,
            MeshComponent {
                vertices: vec![
                    vertex([0.0, 0.0, 0.0]),
                    vertex([1.0, 0.0, 0.0]),
                    vertex([0.0, 1.0, 0.0]),
                ],
                indices: vec![0, 1, 2],
            },
        );
        let object = |mesh_generation: u32, offset: f32| {
            StaticObjectRow::new(
                mesh.index(),
                mesh_generation,
                0,
                1,
                glam::Mat4::from_translation(glam::Vec3::new(offset, 0.0, 0.0)),
                [offset, 0.0, 0.0, 1.0],
                3,
                0,
                0,
                0,
                0,
                0,
            )
        };
        let live = world.spawn();
        world.insert_row(live, object(mesh.generation().wrapping_add(1), 10.0));
        // A row pointing at a mesh generation that no longer exists.
        let stale = world.spawn();
        world.insert_row(stale, object(mesh.generation().wrapping_add(2), 20.0));

        let sun = GpuLight {
            light_type: LightType::Directional as u32,
            direction_outer: [0.0, -1.0, 0.0, 0.0],
            color_intensity: [1.0, 0.9, 0.8, 3.0],
            ..GpuLight::default()
        };
        let sun_entity = world.spawn();
        world.insert_row(sun_entity, LightRow::from(sun));
        let vacant = world.spawn();
        world.insert_row(
            vacant,
            LightRow::from(GpuLight {
                color_intensity: [0.0; 4],
                ..GpuLight::default()
            }),
        );

        let scene = bake_scene_from_world(&world);
        assert_eq!(scene.meshes.len(), 1, "stale object rows are skipped");
        assert_eq!(
            scene.meshes[0].id,
            helio_bake::bake_mesh_id(live.index()),
            "a mesh is keyed by its object's row"
        );
        assert_eq!(
            scene.meshes[0].positions[1],
            [11.0, 0.0, 0.0],
            "positions are world space"
        );
        assert_eq!(scene.meshes[0].indices, vec![0, 1, 2]);
        assert_eq!(scene.lights.len(), 1, "dark rows are not lights");
        assert!(
            matches!(scene.lights[0].kind, LightSourceKind::Directional { direction } if direction == [0.0, -1.0, 0.0])
        );
        assert_eq!(scene.lights[0].intensity, 3.0);
        assert!(
            !scene.lights[0].casts_shadows,
            "GpuLight::default() requests no shadow"
        );
    }
}
