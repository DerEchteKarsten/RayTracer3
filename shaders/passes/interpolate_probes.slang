#include "bindless_descriptor.slang"
#include "gbuffer_helpers.slang"
#include "spherical_harmonics.slang"
#include "brdf.slang"

[[vk::binding(0, 1)]] RWTexture2D<uint4> gbuffer;
[[vk::binding(1, 1)]] RWTexture2D<float> gbuffer_depth;
[[vk::binding(2, 1)]] RWStructuredBuffer<float3x3> sh_coeficents;
[[vk::binding(0, 2)]] RWTexture2D<float4> Light;

[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 threadId : SV_DispatchThreadID) {
    if(any(threadId.xy > gconst.view.viewportSize)) {
        return;
    }

    var pixel_coord = threadId.xy;
    var pixel_depth = gbuffer_depth[pixel_coord];
    if(pixel_depth == BACKGROUND_DEPTH) {
        return;
    }    
    
    var rng = RandomSamplerState(pixel_coord, gconst.frame_index);
    
    var surface = GbufferDataPacked::from_uint4(gbuffer[pixel_coord]).unpack();

    var pixel_pos = world_pos_from_depth(pixel_depth, pixel_coord, gconst.view);
    var cell_size = pixel_depth * gconst.cell_size;

    var jitter = int2((2.0 * float2(rng.uniform_float(), rng.uniform_float()) - 1.0) * 16.0);
    var jitterd_pixel_coord = clamp(int2(pixel_coord) + jitter, 0, int2(gconst.view.viewportSize) - 1);
    var jitterd_depth = gbuffer_depth[jitterd_pixel_coord];
    var jitterd_pos = world_pos_from_depth(jitterd_depth, jitterd_pixel_coord, gconst.view);
    
    if(abs(dot(normalize(jitterd_pos - pixel_pos), surface.normal)) < 0.01) {//0.5f * cell_size) {
        pixel_coord = jitterd_pixel_coord;
    }

    var local_probe_coord = uint2(floor(float2(pixel_coord) / 16.0));
    var local_probe_pixel_coord = local_probe_coord * 16;
    uint2 probe_coords[] = {
        uint2(local_probe_coord.x, local_probe_coord.y), 
        uint2(local_probe_coord.x + 1, local_probe_coord.y),
        uint2(local_probe_coord.x, local_probe_coord.y + 1),
        uint2(local_probe_coord.x + 1, local_probe_coord.y + 1),
    };

    float4 probe_weights = float4(0.0);

    for (uint i = 0; i < 4; ++i) {
        var probe_coord = probe_coords[i];
        var probe_pixel_coord = probe_coord * 16;
        var probe_depth = gbuffer_depth[probe_pixel_coord];

        if(probe_depth == BACKGROUND_DEPTH || any(probe_coord >= uint2(gconst.view.viewportSize) / 16)) {
            continue;
        }
        
        var probe_position = world_pos_from_depth(probe_depth, probe_pixel_coord, gconst.view);
        
        if (abs(dot(probe_position - pixel_pos, surface.normal)) > cell_size) {
            probe_weights[i] = 0.0;
        }else {
            probe_weights[i] = saturate(1.0 - abs(probe_depth - pixel_depth) / pixel_depth);
            probe_weights[i] *= max(dot(surface.normal, GbufferDataPacked::from_uint4(gbuffer[probe_pixel_coord]).unpack_normal()), 0.0);
            probe_weights[i] = pow(probe_weights[i], 8.0);
        }
    }
    
    float3 radiance = 0.0;
    if(dot(probe_weights, probe_weights) == 0.0) {
        radiance = float3(1.0, 0.0, 0.0); //Interpolation failed
        Light[pixel_coord] = float4(radiance, 1.0);
        return;
    }

    probe_weights /= probe_weights.x + probe_weights.y + probe_weights.z + probe_weights.w;


    for(uint i = 0; i < 4; ++i) {
        float3 probe_radiance;
        var probe_coord = probe_coords[i];
        if(gconst.proberng == 1) {
            var probe_pixel_coord = probe_coord * 16;
            // RandomSamplerState probe_rng = RandomSamplerState(probe_coord, 1);
            // radiance = float3(probe_rng.uniform_float(), probe_rng.uniform_float(), probe_rng.uniform_float());
            probe_radiance = (GbufferDataPacked::from_uint4(gbuffer[probe_pixel_coord]).unpack_normal() + 1.0) / 2.0;
        }else {
            sh3 shR = sh_coeficents[ZCurveToLinearIndex(uint2(probe_coord.x * 3 + 0, probe_coord.y))];
            sh3 shG = sh_coeficents[ZCurveToLinearIndex(uint2(probe_coord.x * 3 + 1, probe_coord.y))];
            sh3 shB = sh_coeficents[ZCurveToLinearIndex(uint2(probe_coord.x * 3 + 2, probe_coord.y))];
            
            sh3 sh[] = {shR, shG, shB};
            probe_radiance = sh3UnprojectCosLobe(sh, surface.normal);   
        }
        radiance += probe_weights[i] * max(0.0, probe_radiance);
    }

    radiance *= surface.albedo * FRAC_1_PI;
    radiance += surface.emissive;

    Light[threadId.xy] = float4(radiance, 1.0);
}