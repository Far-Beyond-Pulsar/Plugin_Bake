//! Baked lighting's public contract (`pulsar.bake`, Plugin_Bake).
//!
//! Baking reads meshes, lights and materials and writes lightmaps, probes and
//! visibility; it is one concept with one owner, the bake plugin (Unreal's
//! Lightmass and Unity's Progressive Lightmapper are separate systems for the
//! same reason). Other kinds meet it here:
//!
//! - [`LIGHTMAP_UV`]: the G-buffer surface payload the bake requests
//!   (`helio_frame_api::gbuffer::request_payload`). A G-buffer drawer whose
//!   vertices carry lightmap UVs writes the object's atlas UV there
//!   ([`LIGHTMAP_REGIONS_SNIPPET`]'s `lightmap_atlas_uv`) when the graph
//!   requested it.
//! - [`LIGHTMAP_REGIONS`]: the per-object atlas regions ([`LightmapRegion`],
//!   indexed by `static_objects` row), a frame input the bake publishes.
//! - [`BAKED_PVS`]: the potentially-visible set the meshes' culling reads.
//!
//! The baked irradiance itself reaches lighting through the frame's
//! `indirect_diffuse` contract, never by a lighting model sampling a lightmap.

use bytemuck::{Pod, Zeroable};

/// The G-buffer surface payload baked lighting requests: a lightmapped
/// surface's atlas UV.
pub const LIGHTMAP_UV: helio_frame_api::gbuffer::SurfacePayload =
    helio_frame_api::gbuffer::SurfacePayload("lightmap_uv");

/// Graph key of the per-object lightmap regions: a storage buffer of
/// [`LightmapRegion`]s indexed by the `static_objects` row an object occupies
/// (SceneDB keeps a row at its entity's index). Published as a frame input
/// while a bake is installed; absent, no surface has a lightmap.
pub const LIGHTMAP_REGIONS: &str = "lightmap_regions";

/// One object's region of the lightmap atlas, as [`LIGHTMAP_REGIONS`] holds
/// it (`LightmapRegion` in WGSL). `scale` zero: no lightmap.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct LightmapRegion {
    pub offset: [f32; 2],
    pub scale: [f32; 2],
    /// Half-texel-inset bounds of the region in atlas UV.
    pub clamp_min: [f32; 2],
    pub clamp_max: [f32; 2],
}

const _: () = assert!(std::mem::size_of::<LightmapRegion>() == 32);

/// Marker opting a shader into [`LIGHTMAP_REGIONS_SNIPPET`].
pub const LIGHTMAP_REGIONS_MARKER: &str = "//!use lightmap_regions";

/// `LightmapRegion` and `lightmap_atlas_uv` in WGSL.
pub const LIGHTMAP_REGIONS_SNIPPET: helio_core::shader::ShaderSnippet =
    helio_core::wgsl_snippet!(LIGHTMAP_REGIONS_MARKER, "../shaders/lightmap_regions.wgsl");

/// Graph key of the baked potentially-visible set ([`BakedPvsRef`]), a frame
/// input the bake publishes; the meshes' occlusion culling reads it.
pub const BAKED_PVS: &str = "baked_pvs";

/// Owned CPU-side PVS data, as a bake stores it.
pub struct BakedPvsData {
    pub world_min: [f32; 3],
    pub world_max: [f32; 3],
    pub grid_dims: [u32; 3],
    pub cell_size: f32,
    pub cell_count: u32,
    pub words_per_cell: u32,
    pub bits: Vec<u64>,
}

/// Borrowed view of a baked PVS bitfield grid, as [`BAKED_PVS`] publishes it.
/// Valid for the frame.
#[derive(Clone, Copy)]
pub struct BakedPvsRef<'a> {
    pub world_min: [f32; 3],
    pub world_max: [f32; 3],
    pub grid_dims: [u32; 3],
    pub cell_size: f32,
    pub cell_count: u32,
    pub words_per_cell: u32,
    /// Packed bitfield: `bits[from * words_per_cell + to/64] >> (to%64) & 1 == 1` means
    /// cell `to` is potentially visible from cell `from`.
    pub bits: &'a [u64],
}

impl BakedPvsRef<'_> {
    /// Returns `true` if cell `to_cell` is potentially visible from cell `from_cell`.
    #[inline]
    pub fn is_visible(&self, from_cell: usize, to_cell: usize) -> bool {
        let idx = from_cell * self.words_per_cell as usize + to_cell / 64;
        if idx >= self.bits.len() {
            return true;
        } // conservative default
        (self.bits[idx] >> (to_cell % 64)) & 1 == 1
    }

    /// Returns the grid-cell index at world position `p`, or `None` if out of bounds.
    #[inline]
    pub fn cell_at(&self, p: [f32; 3]) -> Option<usize> {
        let [gx, gy, gz] = self.grid_dims;
        let dx = ((p[0] - self.world_min[0]) / self.cell_size) as i32;
        let dy = ((p[1] - self.world_min[1]) / self.cell_size) as i32;
        let dz = ((p[2] - self.world_min[2]) / self.cell_size) as i32;
        if dx < 0 || dy < 0 || dz < 0 || dx >= gx as i32 || dy >= gy as i32 || dz >= gz as i32 {
            return None;
        }
        Some(dx as usize + dy as usize * gx as usize + dz as usize * gx as usize * gy as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_region_has_its_wgsl_layout() {
        let source = helio_core::shader::resolve_with(
            "//!use lightmap_regions\n@group(0) @binding(0) var<storage, read> regions: array<LightmapRegion>;\n",
            &[LIGHTMAP_REGIONS_SNIPPET],
        );
        let module = naga::front::wgsl::parse_str(&source).expect("parses");
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).expect("lays out");
        let (handle, _) = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("LightmapRegion"))
            .expect("LightmapRegion");
        assert_eq!(
            layouter[handle].size as usize,
            std::mem::size_of::<LightmapRegion>()
        );
    }
}
