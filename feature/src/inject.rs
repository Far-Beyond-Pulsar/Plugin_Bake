use std::sync::Arc;

use helio_core::ResourceRegistry;
use helio_core::{PassContext, RenderPass, Result as HelioResult};

use crate::data::BakedData;

/// Publishes a bake's results as frame inputs: the lightmap regions
/// (`helio_bake_api::LIGHTMAP_REGIONS`, which the meshes' G-buffer drawer
/// reads), the reflection probes (the frame's environment cubes) and the PVS
/// (`helio_bake_api::BAKED_PVS`, which the meshes' culling reads).
///
/// No GPU work. Frame inputs are published before any pass of the frame
/// runs, so every consumer sees them wherever this pass sits.
pub struct BakeInjectPass {
    data: Arc<BakedData>,
}

impl BakeInjectPass {
    pub fn new(data: Arc<BakedData>) -> Self {
        Self { data }
    }
}

/// Extends a borrow from `self`'s own lifetime to the registry's `'a`.
///
/// Sound here specifically because `BakeInjectPass` only ever borrows out of
/// `self.data: Arc<BakedData>`, which the graph keeps across many frames,
/// never frame-scoped, so any single frame's `'a` is always shorter than the
/// data's real lifetime. The trait's `publish_frame_inputs(&self, ..)` (not
/// `&'a self`) cannot express that relationship; this is the same
/// lifetime-extension idiom `ResourceRegistry::write_texture_binding` uses.
unsafe fn extend_lifetime<'a, T: ?Sized>(value: &T) -> &'a T {
    &*(value as *const T)
}

impl RenderPass for BakeInjectPass {
    fn name(&self) -> &'static str {
        "BakeInject"
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn publish_frame_inputs<'a>(
        &self,
        _camera: &helio_core::GpuCameraUniforms,
        frame: &mut ResourceRegistry<'a>,
    ) {
        // SAFETY: every borrow below is extended per `extend_lifetime`'s doc:
        // out of `self.data: Arc<BakedData>`, which outlives any single frame.
        if let Some(lightmap) = &self.data.lightmap {
            frame.write_buffer_binding(
                helio_bake_api::LIGHTMAP_REGIONS,
                &lightmap.regions,
                "BakeInject",
            );
        }
        if let Some(reflection) = &self.data.reflection {
            frame.write(
                helio_core::ResourceKey::new(helio_frame_api::lighting::ENVIRONMENT_CUBES),
                unsafe { extend_lifetime(&reflection.view) },
                "BakeInject",
            );
            frame.write(
                helio_core::ResourceKey::new(helio_frame_api::lighting::ENVIRONMENT_CUBES_SAMPLER),
                unsafe { extend_lifetime(&reflection.sampler) },
                "BakeInject",
            );
        }
        if let Some(pvs) = self.data.pvs() {
            frame.write(
                helio_core::ResourceKey::new(helio_bake_api::BAKED_PVS),
                helio_bake_api::BakedPvsRef {
                    world_min: pvs.world_min,
                    world_max: pvs.world_max,
                    grid_dims: pvs.grid_dims,
                    cell_size: pvs.cell_size,
                    cell_count: pvs.cell_count,
                    words_per_cell: pvs.words_per_cell,
                    bits: unsafe { extend_lifetime(pvs.bits) },
                },
                "BakeInject",
            );
        }
    }

    fn execute(&mut self, _ctx: &mut PassContext) -> HelioResult<()> {
        Ok(()) // no GPU commands
    }
}
