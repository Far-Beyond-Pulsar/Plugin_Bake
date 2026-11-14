//! The lightmap's irradiance into the frame's indirect diffuse.

use std::sync::Arc;

use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, RenderPass, ResourceKey, Result as HelioResult};
use helio_frame_api::{gbuffer, lighting::INDIRECT_DIFFUSE};

use crate::data::BakedData;

/// Writes the lightmap's indirect irradiance into [`INDIRECT_DIFFUSE`] at
/// every pixel whose G-buffer surface carries the bake's payload
/// (`shaders/baked_irradiance.wgsl`). Runs at `Slot::LightingInputs` after
/// the GI probes' resolve: it refines the buffer an earlier publisher wrote
/// this frame in place, and publishes its own only where none did. A
/// lightmapped surface's indirect light is the bake's (static, noise-free);
/// every other surface keeps the probes'.
pub struct BakedIrradiancePass {
    data: Arc<BakedData>,
    layout: wgpu::BindGroupLayout,
    /// `[publish, refine]`: the `REFINE` override's two values.
    pipelines: [wgpu::ComputePipeline; 2],
    /// Its own indirect diffuse and the frame size it holds, for frames no
    /// earlier publisher wrote one.
    own: Option<(wgpu::Buffer, [u32; 2])>,
    /// Whether this frame wrote `own` (so publishes it).
    publish_own: bool,
    cache: helio_core::gpu_cache::GpuCache,
}

impl BakedIrradiancePass {
    pub fn new(device: &wgpu::Device, data: Arc<BakedData>) -> Self {
        let texture = |binding: u32, filterable: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Baked irradiance"),
            entries: &[
                texture(0, false),
                texture(1, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let module = helio_core::shader::module_with(
            device,
            "Baked irradiance",
            helio_core::include_wgsl!("../shaders/baked_irradiance.wgsl"),
            &[],
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Baked irradiance"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |refine: bool| {
            helio_core::shader::create_compute_pipeline(
                device,
                wgpu::ComputePipelineDescriptor {
                    label: Some("baked_irradiance"),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some("baked_irradiance"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &[("REFINE", if refine { 1.0 } else { 0.0 })],
                        ..Default::default()
                    },
                    cache: None,
                },
            )
        };
        Self {
            data,
            layout,
            pipelines: [pipeline(false), pipeline(true)],
            own: None,
            publish_own: false,
            cache: helio_core::gpu_cache::GpuCache::new(),
        }
    }
}

impl RenderPass for BakedIrradiancePass {
    fn name(&self) -> &'static str {
        "BakedIrradiance"
    }

    fn writes(&self) -> &'static [&'static str] {
        &[INDIRECT_DIFFUSE]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read(gbuffer::SURFACE);
        // The GI probes' resolve (or another earlier publisher), when it ran.
        builder.read_optional(INDIRECT_DIFFUSE);
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        self.publish_own = false;
        let Some(lightmap) = &self.data.lightmap else {
            return Ok(());
        };
        let Some(surface) = ctx
            .registry
            .texture_view(ResourceKey::new(gbuffer::SURFACE))
        else {
            return Err(helio_core::Error::ResourceNotFound(gbuffer::SURFACE.into()));
        };
        let earlier = ctx.registry.buffer_binding(INDIRECT_DIFFUSE);
        let size = [ctx.width.max(1), ctx.height.max(1)];
        if earlier.is_none() {
            if self.own.as_ref().is_none_or(|(_, s)| *s != size) {
                self.own = Some((
                    ctx.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Baked indirect diffuse"),
                        size: u64::from(size[0]) * u64::from(size[1]) * 8,
                        usage: wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    }),
                    size,
                ));
            }
            self.publish_own = true;
        }
        let target = match (earlier, &self.own) {
            (Some(buffer), _) => buffer,
            (None, Some((own, _))) => own,
            (None, None) => unreachable!("allocated above"),
        };
        self.cache.end_frame();
        let group = self.cache.bind_group(
            ctx.device,
            &wgpu::BindGroupDescriptor {
                label: Some("Baked irradiance"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(surface),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&lightmap.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&lightmap.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: target.as_entire_binding(),
                    },
                ],
            },
        );
        let pipeline = &self.pipelines[usize::from(earlier.is_some())];
        let mut cmds = ctx.graphics_cmds();
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Baked irradiance"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, Some(&group), &[]);
        pass.set_pipeline(pipeline);
        pass.dispatch_workgroups(size[0].div_ceil(8), size[1].div_ceil(8), 1);
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        if let (true, Some((own, _))) = (self.publish_own, &self.own) {
            frame.write_buffer_binding(INDIRECT_DIFFUSE, own, "BakedIrradiance");
        }
    }
}
