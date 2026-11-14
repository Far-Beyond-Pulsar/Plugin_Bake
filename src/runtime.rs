//! What the bake plugin registers for a game (and the editor): every runtime
//! contribution, grouped by the engine seam it plugs into.

use engine_backend::subsystems::render::RenderFeatureRegistration;

// ── Rendering ───────────────────────────────────────────────────────────────

/// Baked lighting's feature: while a bake is installed
/// (`helio_bake::BakedLighting`), its frame inputs, the lightmap-UV surface
/// payload and the lightmap's irradiance into the frame's indirect diffuse.
fn render_features() -> helio_feature_api::FeatureSet {
    helio_feature_api::FeatureSet::new().with(helio_bake::BakeFeature)
}

/// Nothing to configure: a bake is installed at runtime.
fn configure_renderer(config: helio::RendererConfig) -> helio::RendererConfig {
    config
}

/// Everything the plugin registers for a game (and the editor).
pub(crate) fn build(p: &mut pulsar_plugins::PluginBuilder) {
    p.add(RenderFeatureRegistration {
        origin: p.origin(),
        features: render_features,
        configure: configure_renderer,
    });
}

#[cfg(test)]
mod tests {
    use engine_backend::subsystems::render::helio_renderer::render_features;

    #[test]
    fn registers_its_feature() {
        let features = render_features::plugin_features();
        assert!(
            features.ids().contains(&"helio.bake"),
            "features: {:?}",
            features.ids()
        );
    }

    /// Every registration of the plugin names this root, so it resolves to
    /// the plugin's id.
    #[test]
    fn every_registration_resolves_to_the_plugin() {
        let ours: Vec<_> = pulsar_plugins::contributions()
            .into_iter()
            .filter(|summary| summary.origin == pulsar_plugins::origin!())
            .collect();
        assert!(
            ours.iter()
                .any(|summary| summary.category == "render.features")
        );
        for summary in &ours {
            assert_eq!(
                pulsar_plugins::plugin_of(summary.origin).map(|plugin| plugin.id.as_str()),
                Some("pulsar.bake"),
                "{summary:?}"
            );
        }
    }
}
