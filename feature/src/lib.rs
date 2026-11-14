//! # helio-bake
//!
//! Baked lighting (`pulsar.bake`): offline bakes through Nebula, cached on
//! disk by their inputs, rendered through the frame's contracts.
//!
//! | Bake | What it produces | How the frame reads it |
//! |---|---|---|
//! | Lightmap | Each object's indirect (bounced) irradiance in an atlas | The objects' atlas UV as the G-buffer surface payload; [`BakedIrradiancePass`] writes it into the frame's indirect diffuse, over the GI probes' |
//! | Reflection probes | A pre-filtered cube array | The frame's environment cubes |
//! | PVS | A cell-to-cell visibility bitfield | The meshes' culling (`helio_bake_api::BAKED_PVS`) |
//!
//! Lights stay dynamic: the lightmap holds indirect light only (stationary
//! lights in Unreal's terms), so a moved light's direct light and shadows
//! stay right and nothing is counted twice.
//!
//! ## Usage
//!
//! ```rust,ignore
//! // Bake (blocking) and install: the next graph build adds the bake's nodes.
//! let baked = helio_bake::run_bake_blocking(device, queue, &scene, &BakeConfig::fast("hall"))?;
//! renderer.set_feature(helio_bake::BakedLighting::new(std::sync::Arc::new(baked)));
//! ```
//!
//! A standalone host builds the scene from its SceneDB rows with
//! `helio-bake-host`, which does both steps.

mod bake;
mod cache;
mod config;
mod data;
mod feature;
mod inject;
mod irradiance;

pub use bake::{run_bake_blocking, BakeError};
pub use cache::BakeStore;
pub use config::{BakeConfig, ProbeSpec};
pub use data::BakedData;
pub use feature::{BakeFeature, BakedLighting};
pub use inject::BakeInjectPass;
pub use irradiance::BakedIrradiancePass;

// Re-export the Nebula scene types users need to build a bake's scene.
pub use nebula::core::scene::{
    BakeMesh, LightSource, LightSourceKind, MaterialDesc as BakeMaterialDesc,
};
pub use nebula::light::LightmapConfig;
pub use nebula::prelude::{BakeContext, NullReporter, SceneGeometry};
pub use nebula::probe::ProbeConfig;
pub use nebula::visibility::PvsConfig;

/// The id a baked mesh carries: its object's `static_objects` row (the
/// object entity's index) in bytes 0..8 and the tag 1 in bytes 8..16, both
/// little-endian, so the bake's lightmap region for the mesh lands at that
/// row of `helio_bake_api::LIGHTMAP_REGIONS`. The nil id (no object) gets
/// no region.
pub fn bake_mesh_id(object_row: u32) -> uuid::Uuid {
    let mut id_bytes = [0u8; 16];
    id_bytes[0..8].copy_from_slice(&u64::from(object_row).to_le_bytes());
    id_bytes[8..16].copy_from_slice(&1u64.to_le_bytes());
    uuid::Uuid::from_bytes(id_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mesh id recovers to its object row at the region the drawer reads.
    #[test]
    fn mesh_ids_land_at_their_object_rows() {
        let regions: Vec<_> = [7u32, 2]
            .iter()
            .map(|&row| {
                let bytes = bake_mesh_id(row).into_bytes();
                cache::CachedAtlasRegion {
                    mesh_id: [
                        u64::from_le_bytes(bytes[..8].try_into().unwrap()),
                        u64::from_le_bytes(bytes[8..].try_into().unwrap()),
                    ],
                    uv_offset: [0.5, 0.25],
                    uv_scale: [0.25, 0.25],
                }
            })
            .chain([cache::CachedAtlasRegion {
                mesh_id: [0, 0],
                uv_offset: [0.0, 0.0],
                uv_scale: [1.0, 1.0],
            }])
            .collect();
        let dense = data::object_regions(&regions, (1024, 1024));
        assert_eq!(dense.len(), 8);
        assert_eq!(dense[7].offset, [0.5, 0.25]);
        assert_eq!(dense[2].scale, [0.25, 0.25]);
        assert_eq!(
            dense[0],
            helio_bake_api::LightmapRegion::default(),
            "nil ids get no region"
        );
        let half = 0.5 / 1024.0;
        assert_eq!(dense[7].clamp_min, [0.5 + half, 0.25 + half]);
        assert_eq!(dense[7].clamp_max, [0.75 - half, 0.5 - half]);
    }
}
