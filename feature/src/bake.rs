use std::sync::Arc;

use glam::Vec3;
use nebula::prelude::{BakeContext, BakePass, NullReporter, SceneGeometry};

use crate::cache::{
    input_key, BakeCache, CachedAtlasRegion, CachedLightmap, CachedProbes, CachedPvs,
};
use crate::config::BakeConfig;
use crate::data::BakedData;

// ── Error type ─────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum BakeError {
    #[error("Nebula bake error: {0}")]
    Nebula(#[from] nebula::prelude::NebulaError),

    #[error("Bake store error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Cache serialization error: {0}")]
    Serialize(#[from] Box<bincode::ErrorKind>),

    #[error("Cache deserialization error: {0}")]
    Deserialize(Box<bincode::ErrorKind>),
}

// ── Public entry point ─────────────────────────────────────────────────────────

/// Each pass's cache input key ([`crate::cache::input_key`]).
struct Keys {
    lightmap: u64,
    probes: u64,
    pvs: u64,
}

/// The lightmap settings a bake runs with: `config`'s, indirect light only.
/// Direct light stays the lights' own at runtime (dynamic, shadowed by the
/// shadow maps); the lightmap adds what bounces, through the frame's
/// indirect diffuse. Baking direct light too would count it twice.
fn indirect_only(config: &nebula::light::LightmapConfig) -> nebula::light::LightmapConfig {
    nebula::light::LightmapConfig {
        direct: false,
        ..config.clone()
    }
}

/// Run all configured bake passes, using the config's store where it holds
/// a fresh result.
///
/// **This function blocks** the calling thread until every cache-miss pass
/// completes on the GPU.
///
/// # Cache behaviour
/// A pass whose stored result holds the same format and inputs (the scene
/// and the pass's settings) is loaded without GPU work; any other is baked
/// and stored before returning. Without a store every pass bakes.
///
/// # Device sharing
/// Nebula bakes on the renderer's device; the results are uploaded there and
/// live as long as the returned [`BakedData`].
pub fn run_bake_blocking(
    helio_device: &Arc<wgpu::Device>,
    helio_queue: &Arc<wgpu::Queue>,
    scene: &SceneGeometry,
    config: &BakeConfig,
) -> Result<BakedData, BakeError> {
    let cache = BakeCache::new(config.store.as_deref(), &config.scene_name);
    // A lightmap needs a mesh with lightmap UVs (UV1); without one there is
    // nothing to bake into it.
    let lightmapped = scene.meshes.iter().any(|mesh| mesh.lightmap_uvs.is_some());
    if config.lightmap.is_some() && !lightmapped {
        log::warn!(
            "[helio-bake] '{}': no mesh carries lightmap UVs (UV1); no lightmap baked",
            config.scene_name
        );
    }
    let lightmap_config = config
        .lightmap
        .as_ref()
        .filter(|_| lightmapped)
        .map(indirect_only);
    let keys = Keys {
        lightmap: input_key(scene, &lightmap_config)?,
        probes: input_key(
            scene,
            &config
                .probes
                .as_ref()
                .map(|spec| (&spec.positions, &spec.config)),
        )?,
        pvs: input_key(scene, &config.pvs)?,
    };

    // ── Determine which passes need work ──────────────────────────────────────
    let mut lm = match lightmap_config {
        Some(_) => cache.load_lightmap(keys.lightmap)?,
        None => None,
    };
    let mut probes = match config.probes {
        Some(_) => cache.load_probes(keys.probes)?,
        None => None,
    };
    let mut pvs = match config.pvs {
        Some(_) => cache.load_pvs(keys.pvs)?,
        None => None,
    };
    let need_lm = lightmap_config.is_some() && lm.is_none();
    let need_probes = config.probes.is_some() && probes.is_none();
    let need_pvs = config.pvs.is_some() && pvs.is_none();

    if need_lm || need_probes || need_pvs {
        log::info!(
            "[helio-bake] Starting GPU bake for '{}' (lightmap={}, probes={}, pvs={})",
            config.scene_name,
            need_lm,
            need_probes,
            need_pvs
        );

        pollster::block_on(async {
            let adapter = helio_device.adapter_info();
            let ctx = BakeContext::from_wgpu(
                helio_device.clone(),
                helio_queue.clone(),
                adapter.name,
                adapter.vendor,
                helio_device.limits(),
            );

            // Lightmap
            if let (true, Some(lm_cfg)) = (need_lm, lightmap_config.as_ref()) {
                log::info!(
                    "[helio-bake] Baking indirect lightmap ({}×{}, {}spp, {} bounces)…",
                    lm_cfg.resolution,
                    lm_cfg.resolution,
                    lm_cfg.samples_per_texel,
                    lm_cfg.bounce_count
                );
                let out = nebula::light::LightmapBaker
                    .execute(scene, lm_cfg, &ctx, &NullReporter)
                    .await?;
                let regions: Vec<CachedAtlasRegion> = out
                    .atlas_regions
                    .iter()
                    .map(|r| {
                        let bytes = r.mesh_id.as_bytes();
                        let row = u64::from_le_bytes(bytes[..8].try_into().unwrap());
                        let tag = u64::from_le_bytes(bytes[8..].try_into().unwrap());
                        CachedAtlasRegion {
                            mesh_id: [row, tag],
                            uv_offset: r.uv_offset,
                            uv_scale: r.uv_scale,
                        }
                    })
                    .collect();
                let cached = CachedLightmap {
                    width: out.width,
                    height: out.height,
                    channels: out.channels,
                    is_f32: out.is_f32,
                    texels: out.texels,
                    atlas_regions: regions,
                };
                cache.save_lightmap(keys.lightmap, &cached)?;
                lm = Some(cached);
                log::info!("[helio-bake] Lightmap bake complete.");
            }

            // Probes
            if need_probes {
                let spec = config.probes.as_ref().unwrap();
                log::info!(
                    "[helio-bake] Baking {} probe(s) ({}px faces, {} mips)…",
                    spec.positions.len(),
                    spec.config.face_resolution,
                    spec.config.specular_mip_levels
                );

                let mut all_face_data: Vec<u8> = Vec::new();
                let mut bytes_per_probe = 0usize;
                // Metadata extracted from the first probe result; fall back to config values.
                let mut face_resolution = spec.config.face_resolution;
                let mut mip_levels = spec.config.specular_mip_levels;
                let mut is_rgbe = false;

                for (i, pos) in spec.positions.iter().enumerate() {
                    let pos_v = Vec3::from(*pos);
                    let (refl, _irradiance) = nebula::probe::ProbeBaker::bake_at(
                        pos_v,
                        scene,
                        &spec.config,
                        &ctx,
                        &NullReporter,
                    )
                    .await?;

                    if i == 0 {
                        bytes_per_probe = refl.face_data.len();
                        face_resolution = refl.face_resolution;
                        mip_levels = refl.mip_levels;
                        is_rgbe = refl.is_rgbe;
                    }
                    all_face_data.extend_from_slice(&refl.face_data);
                }

                let cached = CachedProbes {
                    face_resolution,
                    mip_levels,
                    is_rgbe,
                    face_data: all_face_data,
                    bytes_per_probe,
                    positions: spec.positions.clone(),
                };
                cache.save_probes(keys.probes, &cached)?;
                probes = Some(cached);
                log::info!("[helio-bake] Probe bake complete.");
            }

            // PVS
            if need_pvs {
                let pvs_cfg = config.pvs.as_ref().unwrap();
                log::info!(
                    "[helio-bake] Baking PVS (cell_size={}, ray_budget={})…",
                    pvs_cfg.cell_size,
                    pvs_cfg.ray_budget
                );
                let out = nebula::visibility::PvsBaker
                    .execute(scene, pvs_cfg, &ctx, &NullReporter)
                    .await?;
                let cached = CachedPvs {
                    world_min: out.world_min,
                    world_max: out.world_max,
                    grid_dims: out.grid_dims,
                    cell_size: out.cell_size,
                    cell_count: out.cell_count,
                    words_per_cell: out.words_per_cell,
                    bits: out.bits,
                };
                cache.save_pvs(keys.pvs, &cached)?;
                pvs = Some(cached);
                log::info!("[helio-bake] PVS bake complete.");
            }

            Ok::<(), BakeError>(())
        })?;
    } else {
        log::info!(
            "[helio-bake] All passes for '{}' loaded from cache.",
            config.scene_name
        );
    }

    // ── Upload the results to the renderer's device ─────────────────────────
    Ok(BakedData::upload_to_gpu(
        helio_device,
        helio_queue,
        lm,
        probes,
        pvs,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{bake_mesh_id, BakeMesh, BakeStore, LightSource, LightSourceKind};

    /// A store in memory.
    #[derive(Default)]
    struct Memory(std::sync::Mutex<std::collections::HashMap<String, Vec<u8>>>);

    impl BakeStore for Memory {
        fn read(&self, name: &str) -> Option<Vec<u8>> {
            self.0.lock().unwrap().get(name).cloned()
        }

        fn write(&self, name: &str, bytes: &[u8]) -> std::io::Result<()> {
            self.0
                .lock()
                .unwrap()
                .insert(name.to_owned(), bytes.to_vec());
            Ok(())
        }
    }

    /// A unit-normal quad over `[x0, x1] x [z0, z1]` at height `y`, facing
    /// `ny` (+1 up, -1 down), with UV1 over the whole quad when `uv1`.
    fn quad(
        id: uuid::Uuid,
        (x0, x1): (f32, f32),
        (z0, z1): (f32, f32),
        y: f32,
        ny: f32,
        uv1: bool,
    ) -> BakeMesh {
        let positions = vec![[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]];
        let uvs = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        BakeMesh {
            id,
            positions,
            normals: vec![[0.0, ny, 0.0]; 4],
            uvs: uvs.clone(),
            lightmap_uvs: uv1.then_some(uvs),
            indices: vec![0, 1, 2, 0, 2, 3],
            material_ids: vec![0; 2],
            world_transform: Default::default(),
        }
    }

    fn gpu() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .ok()?;
        Some((Arc::new(device), Arc::new(queue)))
    }

    /// The texels of `lightmap` as linear RGB.
    fn texels(lightmap: &CachedLightmap) -> Vec<[f32; 3]> {
        let floats: Vec<f32> = if lightmap.is_f32 {
            bytemuck::cast_slice(&lightmap.texels).to_vec()
        } else {
            bytemuck::cast_slice::<u8, u16>(&lightmap.texels)
                .iter()
                .map(|&bits| half::f16::from_bits(bits).to_f32())
                .collect()
        };
        floats
            .as_chunks::<4>()
            .0
            .iter()
            .map(|t| [t[0], t[1], t[2]])
            .collect()
    }

    /// Each lightmapped mesh is baked into its own atlas region, a mesh
    /// without UV1 gets none yet still bounces light, and nothing else
    /// lands in a region: floor A under a lit ceiling (no UV1) holds the
    /// ceiling's bounce, floor B far away in the dark holds none. Laying
    /// both floors over the whole atlas (the old bake) mixes them.
    #[test]
    fn each_lightmapped_mesh_bakes_into_its_own_region() {
        let Some((device, queue)) = gpu() else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let mut scene = SceneGeometry::new();
        scene.add_mesh(quad(
            bake_mesh_id(3),
            (-1.0, 1.0),
            (-1.0, 1.0),
            0.0,
            1.0,
            true,
        ));
        scene.add_mesh(quad(
            uuid::Uuid::nil(),
            (-2.0, 2.0),
            (-2.0, 2.0),
            2.0,
            -1.0,
            false,
        ));
        scene.add_mesh(quad(
            bake_mesh_id(5),
            (99.0, 101.0),
            (-1.0, 1.0),
            0.0,
            1.0,
            true,
        ));
        scene.add_light(LightSource {
            kind: LightSourceKind::Point {
                position: [0.0, 1.0, 0.0],
                range: 3.0,
            },
            color: [1.0; 3],
            intensity: 10.0,
            bake_enabled: true,
            casts_shadows: true,
        });
        let store = Arc::new(Memory::default());
        let mut config = BakeConfig::fast("regions").with_store(store.clone());
        if let Some(lightmap) = config.lightmap.as_mut() {
            lightmap.resolution = 128;
            lightmap.samples_per_texel = 64;
        }
        let baked = run_bake_blocking(&device, &queue, &scene, &config).expect("bake");
        assert!(baked.has_lightmap());

        let lightmap_config = config.lightmap.as_ref().map(indirect_only);
        let key = input_key(&scene, &lightmap_config).unwrap();
        let lightmap = BakeCache::new(Some(store.as_ref()), "regions")
            .load_lightmap(key)
            .unwrap()
            .expect("the bake stored its lightmap");

        // Two regions, side by side, keyed by the floors' object rows.
        let regions = &lightmap.atlas_regions;
        assert_eq!(regions.len(), 2, "only the UV1 meshes get regions");
        let region = |row: u64| {
            regions
                .iter()
                .find(|r| r.mesh_id == [row, 1])
                .unwrap_or_else(|| panic!("no region for row {row}"))
        };
        let (a, b) = (region(3), region(5));
        let disjoint = a.uv_offset[0] + a.uv_scale[0] <= b.uv_offset[0]
            || b.uv_offset[0] + b.uv_scale[0] <= a.uv_offset[0]
            || a.uv_offset[1] + a.uv_scale[1] <= b.uv_offset[1]
            || b.uv_offset[1] + b.uv_scale[1] <= a.uv_offset[1];
        assert!(
            disjoint,
            "regions overlap: {:?} {:?}",
            (a.uv_offset, a.uv_scale),
            (b.uv_offset, b.uv_scale)
        );

        // The texels well inside a region (one texel in from its edges).
        let texels = texels(&lightmap);
        let (w, h) = (lightmap.width as usize, lightmap.height as usize);
        let inside = |r: &CachedAtlasRegion| {
            let x0 = (r.uv_offset[0] * w as f32).ceil() as usize + 1;
            let x1 = ((r.uv_offset[0] + r.uv_scale[0]) * w as f32).floor() as usize - 1;
            let y0 = (r.uv_offset[1] * h as f32).ceil() as usize + 1;
            let y1 = ((r.uv_offset[1] + r.uv_scale[1]) * h as f32).floor() as usize - 1;
            (y0..y1)
                .flat_map(move |y| (x0..x1).map(move |x| y * w + x))
                .map(|i| texels[i])
                .map(|t| (t[0] + t[1] + t[2]) / 3.0)
                .collect::<Vec<f32>>()
        };
        let (lit, dark) = (inside(a), inside(b));
        let lit_texels = lit.iter().filter(|&&v| v > 1e-4).count();
        assert!(
            lit_texels * 10 > lit.len() * 9,
            "floor A: {lit_texels} of {} texels hold the ceiling's bounce",
            lit.len()
        );
        let brightest = dark.iter().copied().fold(0.0f32, f32::max);
        assert!(
            brightest < 1e-5,
            "floor B is in the dark but its region holds {brightest}"
        );
    }

    /// A scene with no UV1 mesh bakes no lightmap: the frame then requests
    /// no surface payload and runs no irradiance pass.
    #[test]
    fn a_scene_without_lightmap_uvs_bakes_no_lightmap() {
        let Some((device, queue)) = gpu() else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let mut scene = SceneGeometry::new();
        scene.add_mesh(quad(
            bake_mesh_id(0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            0.0,
            1.0,
            false,
        ));
        let config = BakeConfig::fast("no_uv1");
        let baked = run_bake_blocking(&device, &queue, &scene, &config).expect("bake");
        assert!(!baked.has_lightmap());
    }
}
