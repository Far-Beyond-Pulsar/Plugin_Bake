//! Baked lighting as a feature: what it adds to a graph while a bake is
//! installed.
//!
//! A host bakes (`helio-bake-host`) and installs the result as the
//! [`BakedLighting`] setting, which rebuilds the graph. Building, the
//! feature's preparer requests the G-buffer surface payload
//! `helio_bake_api::LIGHTMAP_UV` (so the meshes' drawer writes each
//! lightmapped object's atlas UV), the inject node publishes the bake's frame
//! inputs at [`Slot::Prepare`] and the irradiance node writes the lightmap
//! into the frame's indirect diffuse at [`Slot::LightingInputs`], after the
//! GI probes' resolve. Without a bake the nodes build nothing.

use std::sync::Arc;

use helio_feature_api::{Feature, FeatureRegistrar, NodeBuild, NodeDescriptor, ParamDesc};
use helio_frame_api::lighting::{inputs_order, InputsOrder};
use helio_frame_api::{FrameNode, Slot};

use crate::data::BakedData;
use crate::{BakeInjectPass, BakedIrradiancePass};

/// The bake a renderer renders with: a feature setting
/// (`Renderer::set_feature`), so installing one rebuilds the graph with the
/// bake's nodes and its surface payload. Default: no bake.
#[derive(Clone, Default)]
pub struct BakedLighting(Option<Arc<BakedData>>);

impl BakedLighting {
    pub fn new(data: Arc<BakedData>) -> Self {
        Self(Some(data))
    }

    pub fn data(&self) -> Option<&Arc<BakedData>> {
        self.0.as_ref()
    }
}

/// Baked lighting ([`crate`]).
pub struct BakeFeature;

const CATEGORY: &str = "Passes: Lighting";

/// The installed bake, if the graph's builder supplied one.
fn installed(build: &NodeBuild<'_, '_>) -> Option<Arc<BakedData>> {
    build
        .env
        .setting::<BakedLighting>()
        .and_then(|baked| baked.data().cloned())
}

fn node(
    name: &str,
    title: &str,
    description: &str,
    build: impl Fn(&mut NodeBuild<'_, '_>) -> Result<(), String> + Send + Sync + 'static,
) -> NodeDescriptor {
    NodeDescriptor::new(
        &format!("helio.pass.{name}"),
        title,
        CATEGORY,
        description,
        build,
    )
    .with_param(ParamDesc::boolean("enabled", "Enabled", true))
    .with_generated_pins()
}

impl Feature for BakeFeature {
    fn id(&self) -> &'static str {
        "helio.bake"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.prepare(Arc::new(|env| {
            let lightmapped = env
                .setting::<BakedLighting>()
                .and_then(BakedLighting::data)
                .is_some_and(|data| data.has_lightmap());
            if lightmapped {
                helio_frame_api::gbuffer::request_payload(env, helio_bake_api::LIGHTMAP_UV);
            }
        }));
        registrar.node(
            node(
                "bake_inject",
                "Bake: Inject",
                "Publishes the installed bake's frame inputs: lightmap regions, reflection probes, PVS. Builds nothing without a bake.",
                |build| {
                    if let Some(data) = installed(build) {
                        build.graph.add_pass(Box::new(BakeInjectPass::new(data)));
                    }
                    Ok(())
                },
            )
            .in_slot(Slot::Prepare, -10),
        );
        registrar.node(
            node(
                "baked_irradiance",
                "Bake: Lightmap Irradiance",
                "The installed bake's lightmapped indirect irradiance into the frame's indirect diffuse, over the GI probes'. Builds nothing without a lightmap, or where the lighting ran before it (forward).",
                |build| {
                    let read = inputs_order(build.env) == InputsOrder::BeforeLighting;
                    match installed(build) {
                        Some(data) if read && data.has_lightmap() => {
                            let pass = BakedIrradiancePass::new(build.env.device, data);
                            build.graph.add_pass(Box::new(pass));
                        }
                        _ => {}
                    }
                    Ok(())
                },
            )
            .in_slot(Slot::LightingInputs, 5),
        );
    }
}
