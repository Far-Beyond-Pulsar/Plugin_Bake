//! Baked lighting (`pulsar.bake`): the plugin's root crate, the one place the
//! plugin registers everything it contributes.
//!
//! The plugin's other crates hold the code: `helio-bake-api` (the contract
//! other kinds read: the lightmap-UV surface payload, the per-object
//! lightmap regions, the PVS), the feature crate (`helio-bake`: Nebula bakes,
//! their cache, and the passes that render them through the frame's
//! contracts) and `helio-bake-host` (baking for standalone hosts). Baking has
//! no SceneDB component yet (a host bakes and installs the result on its
//! renderer) and no editor UI. Neither crate registers a contribution; this
//! crate imports them and registers them through its [`build`]:
//!
//! - [`runtime`]: what a game links. The Helio feature, which adds its nodes
//!   while a bake is installed.
//!
//! Every registration names this crate as its origin (`p.origin()`), so the
//! plugin gate skips all of them while a project disables the plugin.

// The plugin's descriptor (`pulsar-plugin.toml`): linking this crate
// installs the plugin; a project enables it or not (`pulsar_plugins`).
pulsar_plugins::installed!("pulsar-plugin.toml", build);

/// Everything the plugin registers.
pub fn build(p: &mut pulsar_plugins::PluginBuilder) {
    // The icon the editor shows beside this plugin (Extensions window).
    #[cfg(feature = "editor")]
    p.add(pulsar_plugins::PluginIcon {
        origin: p.origin(),
        format: pulsar_plugins::IconFormat::Png,
        bytes: include_bytes!("../icon.png"),
    });
    runtime::build(p);
}

pub mod runtime;
