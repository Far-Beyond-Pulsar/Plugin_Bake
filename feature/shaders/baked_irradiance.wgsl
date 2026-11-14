//!use helio_prelude
// Baked lighting's indirect irradiance at every lightmapped pixel, into the
// frame's indirect diffuse (`helio_frame_api::lighting::INDIRECT_DIFFUSE`):
// a pixel whose G-buffer surface carries the bake's payload (its lightmap
// atlas UV, `helio_bake_api::LIGHTMAP_UV`) takes the lightmap's irradiance
// over pi with full weight. Refining an earlier publisher's buffer (the GI
// probes'), every other pixel keeps what it holds; publishing its own, every
// other pixel gets weight 0 (the lighting's own ambient).

@group(0) @binding(0) var gbuf_surface: texture_2d<f32>;
@group(0) @binding(1) var lightmap: texture_2d<f32>;
@group(0) @binding(2) var lightmap_sampler: sampler;
@group(0) @binding(3) var<storage, read_write> indirect_diffuse: array<vec2<u32>>;

// Whether an earlier publisher wrote `indirect_diffuse` this frame.
override REFINE: bool = false;

// 1 / pi: the contract holds irradiance over pi, the lightmap irradiance.
const BAKED_INV_PI: f32 = 0.31830988618;

@compute @workgroup_size(8, 8)
fn baked_irradiance(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(gbuf_surface);
    if id.x >= size.x || id.y >= size.y { return; }
    let at = id.y * size.x + id.x;
    let surface = textureLoad(gbuf_surface, vec2<i32>(id.xy), 0).xy;
    if !gbuffer_has_payload(surface) {
        if !REFINE { indirect_diffuse[at] = vec2<u32>(0u); }
        return;
    }
    let e = textureSampleLevel(lightmap, lightmap_sampler, surface, 0.0).rgb * BAKED_INV_PI;
    indirect_diffuse[at] = vec2<u32>(pack2x16float(e.rg), pack2x16float(vec2<f32>(e.b, 1.0)));
}
