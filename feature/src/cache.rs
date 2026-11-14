use serde::{Deserialize, Serialize};

use crate::bake::BakeError;

/// Magic header written at the start of every cache file.
const CACHE_MAGIC: &[u8; 8] = b"HLBKCACH";
/// Bump when the serialized layout or what a pass bakes changes, so stale
/// files are rejected (4: indirect-only lightmaps, regions by object row).
const CACHE_VERSION: u32 = 4;

// ── On-disk representations ────────────────────────────────────────────────────

/// Cached lightmap atlas bake.
#[derive(Serialize, Deserialize)]
pub(crate) struct CachedLightmap {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    /// `true` = RGBA32F (4 × f32/texel), `false` = RGBA16F (4 × f16/texel).
    pub is_f32: bool,
    pub texels: Vec<u8>,
    pub atlas_regions: Vec<CachedAtlasRegion>,
}

/// Per-mesh UV region inside the lightmap atlas.
#[derive(Serialize, Deserialize, Clone)]
pub struct CachedAtlasRegion {
    /// UUID of the mesh this region covers (as `[u64; 2]` for serde-free storage).
    pub mesh_id: [u64; 2],
    /// Top-left corner in atlas UV space `[0, 1]`.
    pub uv_offset: [f32; 2],
    /// Width/height in atlas UV space.
    pub uv_scale: [f32; 2],
}

/// Cached reflection + irradiance probe (all probes concatenated).
#[derive(Serialize, Deserialize)]
pub(crate) struct CachedProbes {
    pub face_resolution: u32,
    pub mip_levels: u32,
    pub is_rgbe: bool,
    /// Concatenated face data for all probes: probe_count × face_data_per_probe bytes.
    pub face_data: Vec<u8>,
    /// Number of bytes per probe in `face_data`.
    pub bytes_per_probe: usize,
    /// Baked positions (parallel to the probe entries in `face_data`).
    pub positions: Vec<[f32; 3]>,
}

/// Cached potentially-visible set.
#[derive(Serialize, Deserialize)]
pub(crate) struct CachedPvs {
    pub world_min: [f32; 3],
    pub world_max: [f32; 3],
    pub grid_dims: [u32; 3],
    pub cell_size: f32,
    pub cell_count: u32,
    pub words_per_cell: u32,
    pub bits: Vec<u64>,
}

// ── Cache manager ──────────────────────────────────────────────────────────────

/// Where bakes persist between runs: bytes by name. The host supplies it
/// ([`crate::BakeConfig::with_store`]): the engine's file system, a standalone
/// host's directory (`helio-bake-host`), memory in a test. The bake itself
/// does no IO.
pub trait BakeStore: Send + Sync {
    /// The bytes stored under `name`, if any.
    fn read(&self, name: &str) -> Option<Vec<u8>>;
    /// Stores `bytes` under `name`, replacing what was there.
    fn write(&self, name: &str, bytes: &[u8]) -> std::io::Result<()>;
}

/// A scene's bake cache in a [`BakeStore`].
///
/// Entries are named `{scene_name}_{pass}` and begin with a magic header, a
/// version and the pass's input key (a hash of the scene and the pass's
/// settings, [`input_key`]): an entry baked from other inputs or an older
/// format is stale and re-baked, so editing the scene never reuses an old
/// bake. Without a store nothing is kept and every run bakes.
pub(crate) struct BakeCache<'a> {
    store: Option<&'a dyn BakeStore>,
    scene_name: &'a str,
}

impl<'a> BakeCache<'a> {
    pub fn new(store: Option<&'a dyn BakeStore>, scene_name: &'a str) -> Self {
        Self { store, scene_name }
    }

    fn name(&self, suffix: &str) -> String {
        format!("{}_{}", self.scene_name, suffix)
    }

    // ── Generic helpers ───────────────────────────────────────────────────────

    fn write<T: Serialize>(&self, suffix: &str, key: u64, value: &T) -> Result<(), BakeError> {
        let Some(store) = self.store else {
            return Ok(());
        };
        let payload = bincode::serialize(value).map_err(BakeError::Serialize)?;
        let mut bytes = Vec::with_capacity(20 + payload.len());
        bytes.extend_from_slice(CACHE_MAGIC);
        bytes.extend_from_slice(&CACHE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&key.to_le_bytes());
        bytes.extend_from_slice(&payload);
        let name = self.name(suffix);
        store.write(&name, &bytes)?;
        log::info!("[helio-bake] Stored {name}");
        Ok(())
    }

    fn read<T: for<'de> Deserialize<'de>>(
        &self,
        suffix: &str,
        key: u64,
    ) -> Result<Option<T>, BakeError> {
        let name = self.name(suffix);
        let Some(bytes) = self.store.and_then(|store| store.read(&name)) else {
            return Ok(None);
        };
        // Validate magic, version and input key.
        let header = bytes.get(..20);
        let fresh = header.is_some_and(|h| {
            &h[..8] == CACHE_MAGIC
                && h[8..12] == CACHE_VERSION.to_le_bytes()
                && h[12..20] == key.to_le_bytes()
        });
        if !fresh {
            log::info!("Bake cache {name} is stale (other format or inputs). Will re-bake.");
            return Ok(None);
        }
        let value: T = bincode::deserialize(&bytes[20..]).map_err(|e| {
            log::warn!("Bake cache {name} failed to deserialize: {e}. Will re-bake.");
            BakeError::Deserialize(e)
        })?;
        log::info!("[helio-bake] Loaded from cache: {name}");
        Ok(Some(value))
    }

    // ── Lightmap ──────────────────────────────────────────────────────────────

    pub fn load_lightmap(&self, key: u64) -> Result<Option<CachedLightmap>, BakeError> {
        self.read("lightmap", key)
    }

    pub fn save_lightmap(&self, key: u64, data: &CachedLightmap) -> Result<(), BakeError> {
        self.write("lightmap", key, data)
    }

    // ── Probes ────────────────────────────────────────────────────────────────

    pub fn load_probes(&self, key: u64) -> Result<Option<CachedProbes>, BakeError> {
        self.read("probes", key)
    }

    pub fn save_probes(&self, key: u64, data: &CachedProbes) -> Result<(), BakeError> {
        self.write("probes", key, data)
    }

    // ── PVS ───────────────────────────────────────────────────────────────────

    pub fn load_pvs(&self, key: u64) -> Result<Option<CachedPvs>, BakeError> {
        self.read("pvs", key)
    }

    pub fn save_pvs(&self, key: u64, data: &CachedPvs) -> Result<(), BakeError> {
        self.write("pvs", key, data)
    }
}

/// The input key of a pass: a stable hash (xxHash3) of the scene it bakes
/// and the pass's settings, as they serialize.
pub(crate) fn input_key<S: Serialize>(
    scene: &nebula::prelude::SceneGeometry,
    settings: &S,
) -> Result<u64, BakeError> {
    let mut hasher = twox_hash::XxHash3_64::default();
    std::hash::Hasher::write(&mut hasher, &bincode::serialize(scene)?);
    std::hash::Hasher::write(&mut hasher, &bincode::serialize(settings)?);
    Ok(std::hash::Hasher::finish(&hasher))
}
