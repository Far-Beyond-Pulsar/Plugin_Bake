use std::sync::Arc;

use crate::BakeStore;

/// Configuration for offline baking.
///
/// Controls which passes run, their quality settings, and the cache directory.
/// Use [`BakeConfig::fast`] or [`BakeConfig::ultra`] for common presets,
/// or construct manually for custom settings.
#[derive(Clone)]
pub struct BakeConfig {
    /// Unique name for this scene's bake artifacts.
    ///
    /// The store keeps each pass under `{scene_name}_{pass}`, with the hash
    /// of the inputs it was baked from.
    /// Example: `"outdoor_rocks"`, `"indoor_corridor"`.
    pub scene_name: String,

    /// Where bakes persist between runs ([`BakeStore`], supplied by the
    /// host). `None`: nothing is kept and every run bakes.
    pub store: Option<Arc<dyn BakeStore>>,

    /// Bake lightmaps: each object's indirect (bounced) irradiance, in an
    /// atlas with one region per object. Always baked without direct light
    /// (`direct` is ignored): direct light stays the lights' own at runtime,
    /// and the lightmap reaches the lighting as the frame's indirect diffuse.
    /// Only meshes with lightmap UVs (UV1) get a region; a scene with none
    /// bakes no lightmap. Set to `None` to skip (real-time lighting only).
    pub lightmap: Option<nebula::light::LightmapConfig>,

    /// Bake reflection + irradiance probes at one or more world positions.
    ///
    /// Each probe produces:
    /// - A pre-filtered RGBA32F cubemap mip chain (specular IBL)
    /// - L2 spherical-harmonic coefficients (diffuse irradiance) — 9 RGB values
    ///
    /// Set to `None` to skip (runtime IBL from sky LUT only).
    pub probes: Option<ProbeSpec>,

    /// Bake potentially-visible sets (PVS) for CPU-side visibility culling.
    ///
    /// Provides a fast `is_visible(from_cell, to_cell)` query at runtime,
    /// which can gate draw calls before they reach GPU culling.
    /// Set to `None` to skip.
    pub pvs: Option<nebula::visibility::PvsConfig>,
}

/// Probe bake spec: where probes are placed and at what quality.
#[derive(Clone, Debug)]
pub struct ProbeSpec {
    /// World-space positions to bake.
    ///
    /// In a typical scene: one probe per room/zone, placed at head height.
    /// The baked probes are stored sequentially and can be indexed by position
    /// using the closest-probe logic in [`BakedData`](crate::BakedData).
    pub positions: Vec<[f32; 3]>,

    /// Quality / resolution settings for probe capture.
    pub config: nebula::probe::ProbeConfig,
}

impl Default for BakeConfig {
    fn default() -> Self {
        Self {
            scene_name: "scene".into(),
            store: None,
            lightmap: Some(nebula::light::LightmapConfig::default()),
            probes: None,
            pvs: None,
        }
    }
}

impl BakeConfig {
    /// Fast preset: a one-bounce lightmap at 512² with a few samples, no
    /// probes or PVS. For iteration. (Nebula has no denoiser yet: `denoise`
    /// is not read.)
    pub fn fast(scene_name: impl Into<String>) -> Self {
        Self {
            scene_name: scene_name.into(),
            store: None,
            lightmap: Some(nebula::light::LightmapConfig {
                samples_per_texel: 32,
                denoise: true,
                ..nebula::light::LightmapConfig::fast()
            }),
            probes: None,
            pvs: None,
        }
    }

    /// Medium preset: multi-bounce GI lightmap with denoising.
    ///
    /// 64 spp, 2 bounces, spatial denoising — good balance of quality and
    /// bake time (seconds to low minutes).  Suitable for playtesting.
    pub fn medium(scene_name: impl Into<String>) -> Self {
        Self {
            scene_name: scene_name.into(),
            store: None,
            lightmap: Some(nebula::light::LightmapConfig {
                resolution: 1024,
                samples_per_texel: 64,
                bounce_count: 2,
                denoise: true,
                ..nebula::light::LightmapConfig::default()
            }),
            probes: None,
            pvs: None,
        }
    }

    /// Ultra preset: all passes at maximum quality.
    ///
    /// Bake once for shipping. Bake times are minutes.
    pub fn ultra(scene_name: impl Into<String>) -> Self {
        Self {
            scene_name: scene_name.into(),
            store: None,
            lightmap: Some(nebula::light::LightmapConfig::ultra()),
            probes: Some(ProbeSpec {
                positions: vec![[0.0, 2.0, 0.0]],
                config: nebula::probe::ProbeConfig::ultra(),
            }),
            pvs: Some(nebula::visibility::PvsConfig::default()),
        }
    }

    /// Keep bakes in `store` between runs (builder pattern).
    pub fn with_store(mut self, store: Arc<dyn BakeStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// Add a probe spec (builder pattern).
    pub fn with_probes(mut self, probe_spec: ProbeSpec) -> Self {
        self.probes = Some(probe_spec);
        self
    }

    /// Enable PVS baking with default settings (builder pattern).
    pub fn with_pvs(mut self) -> Self {
        self.pvs = Some(nebula::visibility::PvsConfig::default());
        self
    }
}

impl std::fmt::Debug for BakeConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BakeConfig")
            .field("scene_name", &self.scene_name)
            .field("store", &self.store.is_some())
            .field("lightmap", &self.lightmap)
            .field("probes", &self.probes)
            .field("pvs", &self.pvs)
            .finish()
    }
}
