# Plugin_Bake

Baked lighting: offline lightmaps (indirect light), reflection probes and
visibility, rendered through the frame's contracts.

`pulsar.bake` ("Baked Lighting") is a plugin of
[Pulsar-Native](https://github.com/Far-Beyond-Pulsar/Pulsar-Native), the engine and editor that renders with
[Helio](https://github.com/Far-Beyond-Pulsar/Helio). Pulsar-Native vendors this repository as a git submodule
at `plugins/vendor/bake`.

## What it owns

Every object kind is a plugin. A plugin owns its SceneDB components (their
data and scriptable methods), the contracts other kinds read (its `api`
crate: graph keys, row ABIs, WGSL snippets), the Helio feature that renders
it, its file formats and its editor UI. Helio knows no object kind, and other
kinds meet this one only through frame contracts, extension points, draw
phases and api crates.

| Crate | Path | What |
|---|---|---|
| `helio-bake-api` | `api/` | The contract other kinds read: the lightmap-UV G-buffer surface payload, the per-object lightmap regions (by `static_objects` row) and their WGSL, the PVS |
| `bake_plugin` | `./` (`src/`) | The plugin's root crate: registers the Helio feature |
| `helio-bake` | `feature/` | Nebula bakes cached by their inputs, and the feature that renders them: the inject node (regions, reflection probes, PVS) and the lightmap's irradiance into the frame's indirect diffuse |
| `helio-bake-host` | `host/` | Baking for standalone hosts: a bake's scene from SceneDB rows, the bake installed on a renderer |
| Nebula | `vendor/nebula/` | The baker (its own Cargo workspace; `helio-bake` uses its lightmap, probe and visibility passes) |

## How a bake renders

- **Lightmaps need a lightmap UV channel (UV1).** A mesh without one gets no
  lightmap region (it still occludes and reflects light in the bake); a
  scene with none bakes no lightmap. Tiled texture UVs are never used.
- **Lightmaps hold indirect light only.** Lights stay dynamic: direct light
  and shadows are the lights' own at runtime (Unreal's stationary lights),
  so a moved light stays right and nothing is counted twice.
- The feature requests the G-buffer's surface payload
  (`helio_frame_api::gbuffer::request_payload`, `LIGHTMAP_UV`); the meshes'
  G-buffer drawer then writes each lightmapped object's atlas UV, from the
  regions the bake publishes.
- After the GI probes' resolve, the irradiance node writes the lightmap's
  irradiance into the frame's indirect diffuse at every lightmapped pixel;
  every other pixel keeps the probes'. No lighting model samples a lightmap.
- A host bakes and installs the result with `Renderer::set_feature(BakedLighting::new(..))`
  (`helio_bake_host::bake` does both); the graph rebuilds with the bake's
  nodes.
- Bakes persist in a store the host supplies (`BakeStore`, bytes by name;
  `helio_bake_host::disk_store` keeps `{dir}/{scene}_{pass}.bin` through
  the engine's file system). Each entry holds a hash of the pass's inputs
  (the scene and the pass's settings): editing either re-bakes. The bake
  itself does no IO; without a store every run bakes.

## Building and testing

This plugin's crates are members of Pulsar-Native's workspace: their
dependencies are `workspace = true`, and Helio comes from Pulsar-Native's
`crates/renderer/helio` submodule. It builds only inside a Pulsar-Native
checkout:

```bash
git clone --recursive https://github.com/Far-Beyond-Pulsar/Pulsar-Native
cd Pulsar-Native
cargo check -p helio-bake-api -p bake_plugin -p helio-bake -p helio-bake-host --tests
cargo test -p helio-bake-api -p bake_plugin -p helio-bake -p helio-bake-host
```

The plugin's real-frame checks are the cathedral examples of the meshes, GI,
decals, post and water plugins, which bake their scene with
`helio-bake-host`.

A change that spans this plugin and Pulsar-Native (or Helio) merges
dependency-first: merge here, re-pin `plugins/vendor/bake` in Pulsar-Native to
this repository's `main`, then merge the Pulsar-Native PR (Pulsar-Native's
`.agents/CRATES.md`, "Submodule pins").
