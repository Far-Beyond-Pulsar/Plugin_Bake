use helio_bake_api::{BakedPvsData, BakedPvsRef, LightmapRegion};

use crate::cache::{CachedAtlasRegion, CachedLightmap, CachedProbes, CachedPvs};

/// What a bake left on the GPU (and the PVS on the CPU), read by the bake's
/// passes ([`crate::BakeFeature`]) for as long as the host keeps it
/// installed ([`crate::BakedLighting`]).
pub struct BakedData {
    pub(crate) lightmap: Option<Lightmap>,
    pub(crate) reflection: Option<Reflection>,
    pub(crate) pvs: Option<BakedPvsData>,
}

/// The lightmap atlas: indirect irradiance (not divided by pi) per texel,
/// and each object's region by `static_objects` row.
pub(crate) struct Lightmap {
    _texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
    pub(crate) sampler: wgpu::Sampler,
    /// `helio_bake_api::LIGHTMAP_REGIONS`: one [`LightmapRegion`] per object
    /// row up to the highest baked one; rows without a lightmap are zero.
    pub(crate) regions: wgpu::Buffer,
}

/// The baked reflection probes: one cube array, six layers a probe.
pub(crate) struct Reflection {
    _texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
    pub(crate) sampler: wgpu::Sampler,
}

/// The baked regions as [`LightmapRegion`]s indexed by object row (a baked
/// mesh's id is its object's row, [`crate::bake_mesh_id`]), with
/// half-texel-inset clamps so bilinear filtering never reads a neighbouring
/// region. Always at least one (zero) region.
pub(crate) fn object_regions(
    regions: &[CachedAtlasRegion],
    (atlas_w, atlas_h): (u32, u32),
) -> Vec<LightmapRegion> {
    let half = [0.5 / atlas_w.max(1) as f32, 0.5 / atlas_h.max(1) as f32];
    let rows = regions
        .iter()
        .filter(|r| r.mesh_id[1] == 1)
        .map(|r| r.mesh_id[0] as usize + 1)
        .max()
        .unwrap_or(1);
    let mut out = vec![LightmapRegion::default(); rows];
    for r in regions.iter().filter(|r| r.mesh_id[1] == 1) {
        out[r.mesh_id[0] as usize] = LightmapRegion {
            offset: r.uv_offset,
            scale: r.uv_scale,
            clamp_min: [r.uv_offset[0] + half[0], r.uv_offset[1] + half[1]],
            clamp_max: [
                r.uv_offset[0] + r.uv_scale[0] - half[0],
                r.uv_offset[1] + r.uv_scale[1] - half[1],
            ],
        };
    }
    out
}

impl BakedData {
    /// The baked PVS, as `helio_bake_api::BAKED_PVS` publishes it.
    pub fn pvs(&self) -> Option<BakedPvsRef<'_>> {
        self.pvs.as_ref().map(|p| BakedPvsRef {
            world_min: p.world_min,
            world_max: p.world_max,
            grid_dims: p.grid_dims,
            cell_size: p.cell_size,
            cell_count: p.cell_count,
            words_per_cell: p.words_per_cell,
            bits: &p.bits,
        })
    }

    /// Whether the bake holds a lightmap (the graph then requests
    /// `helio_bake_api::LIGHTMAP_UV`).
    pub fn has_lightmap(&self) -> bool {
        self.lightmap.is_some()
    }

    // ── Upload cached data to the renderer's device ───────────────────────

    pub(crate) fn upload_to_gpu(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        lm: Option<CachedLightmap>,
        probes: Option<CachedProbes>,
        pvs: Option<CachedPvs>,
    ) -> Self {
        Self {
            lightmap: None,
            reflection: None,
            pvs: None,
        }
        .with_lightmap(device, queue, lm)
        .with_probes(device, queue, probes)
        .with_pvs(pvs)
    }

    fn with_lightmap(
        mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        lm: Option<CachedLightmap>,
    ) -> Self {
        let Some(lm) = lm else { return self };

        // Always upload as Rgba16Float — universally filterable on all backends (Vulkan, DX12,
        // Metal, WebGPU) without requiring FLOAT32_FILTERABLE. Full HDR range up to ~65504
        // is more than sufficient for lightmap irradiance values in any practical scene.
        // This also halves VRAM usage vs Rgba32Float.
        let format = wgpu::TextureFormat::Rgba16Float;
        let bytes_per_row = lm.width * 8; // 4 channels × 2 bytes (f16)

        // If the bake cache stored f32 data, convert each channel to f16 before upload.
        // Otherwise the texels are already f16 bytes and can be used directly.
        let converted: Vec<u8>;
        let texels: &[u8] = if lm.is_f32 {
            let src: &[f32] = bytemuck::cast_slice(&lm.texels);
            let dst: Vec<u16> = src
                .iter()
                .map(|&v| half::f16::from_f32(v).to_bits())
                .collect();
            converted = bytemuck::cast_slice::<u16, u8>(&dst).to_vec();
            &converted
        } else {
            &lm.texels
        };

        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Baked Lightmap"),
            size: wgpu::Extent3d {
                width: lm.width,
                height: lm.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            texels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(lm.height),
            },
            wgpu::Extent3d {
                width: lm.width,
                height: lm.height,
                depth_or_array_layers: 1,
            },
        );
        let view = tex.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Baked Lightmap Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        log::info!(
            "[helio-bake] Uploaded lightmap atlas ({}×{}, {:?}) to GPU.",
            lm.width,
            lm.height,
            format
        );
        let regions = object_regions(&lm.atlas_regions, (lm.width, lm.height));
        let regions_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Lightmap regions"),
            size: std::mem::size_of_val(regions.as_slice()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&regions_buf, 0, bytemuck::cast_slice(&regions));
        self.lightmap = Some(Lightmap {
            _texture: tex,
            view,
            sampler,
            regions: regions_buf,
        });
        self
    }

    fn with_probes(
        mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        probes: Option<CachedProbes>,
    ) -> Self {
        let Some(probes) = probes else { return self };
        if probes.positions.is_empty() {
            return self;
        }

        let probe_count = probes.positions.len() as u32;
        let res = probes.face_resolution;
        let format = if probes.is_rgbe {
            wgpu::TextureFormat::Rgba8Unorm
        } else {
            wgpu::TextureFormat::Rgba32Float
        };
        let bytes_per_pixel = if probes.is_rgbe { 4u32 } else { 16u32 };

        // One cube array covering every baked probe: layer 6*i..6*i+6 is probe
        // i's faces, which is what GpuReflectionCapture.cubemap_index selects.
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Baked Reflection Cubemap Array"),
            size: wgpu::Extent3d {
                width: res,
                height: res,
                depth_or_array_layers: 6 * probe_count,
            },
            mip_level_count: probes.mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Each probe's slice of face_data is laid out [mip][face], so walk it
        // per probe and clamp to that probe's slice — running the cursor past
        // bytes_per_probe would bleed one probe's mips into the next.
        for probe in 0..probe_count as usize {
            let probe_start = probe * probes.bytes_per_probe;
            let probe_end = (probe_start + probes.bytes_per_probe).min(probes.face_data.len());
            let mut offset = probe_start;
            for mip in 0..probes.mip_levels {
                let mip_res = (res >> mip).max(1);
                let face_bytes = (mip_res * mip_res * bytes_per_pixel) as usize;
                for face in 0..6u32 {
                    let end = (offset + face_bytes).min(probe_end);
                    if end <= offset {
                        break;
                    }
                    queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &tex,
                            mip_level: mip,
                            origin: wgpu::Origin3d {
                                x: 0,
                                y: 0,
                                z: probe as u32 * 6 + face,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        &probes.face_data[offset..end],
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(mip_res * bytes_per_pixel),
                            rows_per_image: Some(mip_res),
                        },
                        wgpu::Extent3d {
                            width: mip_res,
                            height: mip_res,
                            depth_or_array_layers: 1,
                        },
                    );
                    offset = end;
                }
            }
        }

        let view = tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Baked Reflection View"),
            dimension: Some(wgpu::TextureViewDimension::CubeArray),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Baked Reflection Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        log::info!(
            "[helio-bake] Uploaded {} reflection probe(s) ({}×{}, {} mips) to GPU.",
            probe_count,
            res,
            res,
            probes.mip_levels
        );
        self.reflection = Some(Reflection {
            _texture: tex,
            view,
            sampler,
        });
        self
    }

    fn with_pvs(mut self, pvs: Option<CachedPvs>) -> Self {
        let Some(pvs) = pvs else { return self };
        log::info!(
            "[helio-bake] Loaded PVS ({} cells, {:.1}m cell size).",
            pvs.cell_count,
            pvs.cell_size
        );
        self.pvs = Some(BakedPvsData {
            world_min: pvs.world_min,
            world_max: pvs.world_max,
            grid_dims: pvs.grid_dims,
            cell_size: pvs.cell_size,
            cell_count: pvs.cell_count,
            words_per_cell: pvs.words_per_cell,
            bits: pvs.bits,
        });
        self
    }
}
