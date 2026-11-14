// Baked lighting's per-object lightmap regions (`helio_bake_api::LIGHTMAP_REGIONS`),
// indexed by the `static_objects` row an object occupies. A drawer whose
// vertices carry lightmap UVs writes the atlas UV into the G-buffer's surface
// payload when the graph requested `helio_bake_api::LIGHTMAP_UV`.

// One object's region of the lightmap atlas (32 bytes). `scale` zero: the
// object has no lightmap.
struct LightmapRegion {
    offset: vec2<f32>,
    scale: vec2<f32>,
    // Half-texel-inset bounds: bilinear filtering never reads a neighbour.
    clamp_min: vec2<f32>,
    clamp_max: vec2<f32>,
}

// The atlas UV of lightmap coordinate `uv` (0..1 over the object) in
// `region`, or a negative value when the object has no lightmap (the frame's
// plain-surface tag is written instead).
fn lightmap_atlas_uv(region: LightmapRegion, uv: vec2<f32>) -> vec2<f32> {
    if region.scale.x <= 0.0 {
        return vec2<f32>(-1.0);
    }
    return clamp(region.offset + uv * region.scale, region.clamp_min, region.clamp_max);
}
