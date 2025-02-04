#include "bindless_descriptor.slang"
#include "hit_logic.slang"
#include "gbuffer_helpers.slang"
#include "math.slang"
#include "brdf.slang"
#include "random.slang"

[[vk::binding(0, 1)]] RWTexture2D<uint4> gbuffer;
[[vk::binding(1, 1)]] RWTexture2D<float> gbuffer_depth;
[[vk::binding(2, 1)]] RWTexture2D<uint16_t> directions;
[[vk::binding(0, 2)]] RWTexture2D<float4> probe_atlas;
[[vk::binding(0, 3)]] RWTexture2D<float4> prev_probe_atlas;

[shader("raygeneration")]
void main() {
    RayPayload payload;
    var probe_atlas_index = DispatchRaysIndex().xy;
    uint2 probe_coord = probe_atlas_index / 8;
    
    var probe_rng = RandomSamplerState(probe_coord, gconst.frame_index);
    var ray_rng = RandomSamplerState(probe_atlas_index, gconst.frame_index);

    uint2 pixel_coord = probe_coord * 16;

    var gbuffer_depth = gbuffer_depth[pixel_coord];
    if(gbuffer_depth == BACKGROUND_DEPTH) {
        probe_atlas[probe_atlas_index] = float4(0.0, 0.0, 0.0, 1.0); //TEMPORARY
        return;
    }


    var gbuffer_data = GbufferDataPacked::from_uint4(gbuffer[pixel_coord]).unpack();
    DiffuseBrdf primary_brdf = {gbuffer_data.albedo};  

    uint2 ray_probe_index = probe_atlas_index % 8;
    uint16_t direction = directions[probe_atlas_index];
    uint direction_index = direction & 0b0111111111111111;
    uint mip = direction >> 15;
    uint size = (1 << mip) * 8;
    var direction_2d = float2(uint(direction_index / size), direction_index % size);

    var ray_probe_direction = octa_decode((direction_2d + float2(ray_rng.uniform_float(), ray_rng.uniform_float())) / float(size)); 
    // var ray_probe_direction = octa_decode((float2(ray_probe_index) + float2(ray_rng.uniform_float(), ray_rng.uniform_float())) / 8.0);

    var tangend = build_orthonormal_basis(gbuffer_data.normal);

    RayDesc ray;
    ray.Origin = world_pos_from_depth(gbuffer_depth, pixel_coord, gconst.view);
    ray.Direction = ray_probe_direction;
    ray.TMin = 0.0005;
    ray.TMax = BACKGROUND_DEPTH;
    
    trace(ray, payload);
    var secondary_surface = hit_info(payload);
    var probe_depth = payload.depth;

    DiffuseBrdf secondary_brdf = {secondary_surface.albedo};
    var radiance = secondary_surface.emissive;

    // var sample = secondary_brdf.sample(ray.Direction, float2(ray_rng.uniform_float(), ray_rng.uniform_float()));
    // var orthogonal_basis = build_orthonormal_basis(secondary_surface.normal);

    // ray.Origin += payload.depth * ray.Direction;
    // ray.Direction = mul(orthogonal_basis, sample.wi);

    // trace(ray, payload);
    // var emissive = hit_info(payload).emissive;
    
    // var radiance = emissive * sample.value_over_pdf * primary_brdf.evaluate(ray_probe_direction, ray.Origin).value_over_pdf;

    probe_atlas[probe_atlas_index] = float4(lerp(prev_probe_atlas[probe_atlas_index].xyz, radiance, gconst.blendfactor), probe_depth); //float4(float3(exp(-payload.depth / 100.0)), 1.0);

    // probe_atlas[probe_atlas_index] = float4(probe_rng.uniform_float(), probe_rng.uniform_float(), probe_rng.uniform_float(), 1.0);
}