#include "bindless_descriptor.slang"
#include "hit_logic.slang"
#include "gbuffer_helpers.slang"
#include "math.slang"
#include "brdf.slang"
#include "random.slang"

[[vk::binding(0, 1)]] RWTexture2D<uint4> gbuffer;
[[vk::binding(1, 1)]] RWTexture2D<float> gbuffer_depth;
[[vk::binding(2, 1)]] RWTexture2D<float4> probe_atlas;

[shader("raygeneration")]
void main() {
    RayPayload payload;
    var probeAtlasIndex = DispatchRaysIndex().xy;
    uint2 probeCoord = probeAtlasIndex / 8;
    uint2 pixelCoord = probeCoord * 16;

    var gbuffer_depth = gbuffer_depth[pixelCoord];
    if(gbuffer_depth == BACKGROUND_DEPTH) {
        probe_atlas[probeAtlasIndex] = float4(0.0, 0.0, 0.0, 1.0); //TEMPORARY
        return;
    }

    var probeRng = RandomSamplerState(probeCoord, gconst.frame_index);
    var rayRng = RandomSamplerState(probeAtlasIndex, gconst.frame_index);

    var gbuffer_data = GbufferDataPacked::from_uint4(gbuffer[pixelCoord]).unpack();
    DiffuseBrdf primary_brdf = {gbuffer_data.albedo}; 

    uint2 rayProbeIndex = probeAtlasIndex % 8;
    var rayProbeDirection = octa_decode(float2(rayProbeIndex) / 8.0);
    RayDesc ray;
    ray.Origin = gconst.view.cameraDirectionOrPosition.xyz + view_dir(pixelCoord, gconst.view) * gbuffer_depth;
    ray.Direction = rayProbeDirection;
    ray.TMin = 0.001;
    ray.TMax = BACKGROUND_DEPTH;
    
    trace(ray, payload);
    var seconary_surface = hit_info(payload);

    DiffuseBrdf secondary_brdf = {seconary_surface.albedo};
    var sample = secondary_brdf.sample(ray.Direction, float2(rayRng.uniform_float(), rayRng.uniform_float()));
    var orthogonal_basis = build_orthonormal_basis(seconary_surface.normal);

    ray.Origin += payload.depth * ray.Direction;
    ray.Direction = mul(orthogonal_basis, sample.wi);

    trace(ray, payload);
    var emissive = hit_info(payload).emissive;
    
    var radiance = emissive * sample.value_over_pdf * primary_brdf.evaluate(rayProbeDirection, ray.Origin).value_over_pdf;

    probe_atlas[probeAtlasIndex] = float4(radiance, 0.0); //float4(float3(exp(-payload.depth / 100.0)), 1.0);
}