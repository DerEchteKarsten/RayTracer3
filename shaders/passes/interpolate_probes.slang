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
    uint2 pixel_coord = threadId.xy;
    
    RandomSamplerState rng = RandomSamplerState(pixel_coord, gconst.frame_index);

    var pixel_depth = gbuffer_depth[pixel_coord];
    if(pixel_depth == BACKGROUND_DEPTH) {
        return;
    }
    
    uint2 local_probe_coord = uint2(floor(float2(pixel_coord) / 16.0));
    uint2 probe_coords[] = {
        uint2(local_probe_coord.x, local_probe_coord.y), 
        uint2(local_probe_coord.x + 1, local_probe_coord.y + 1), 
    };
    
    var pixel_pos = world_pos_from_depth(pixel_depth, pixel_coord, gconst.view);
    var surface = GbufferDataPacked::from_uint4(gbuffer[pixel_coord]).unpack();
    DiffuseBrdf brdf = {surface.albedo};  
    var orth_bias = build_orthonormal_basis(surface.normal);

    float4 probe_values[4] = {float4(0.0)};

    [unroll]
    for (int y = 0; y < 2; y++) {
        [unroll]
        for (int x = 0; x < 2; x++) {
            var probe_coord = uint2(probe_coords[x].x, probe_coords[y].y);

            var probe_position = world_pos_from_depth(gbuffer_depth[probe_coord * 16], probe_coord * 16, gconst.view);
            var to_probe = pixel_pos - probe_position;
            var probe_weight = 1.0 - abs(dot(surface.normal, to_probe));
            probe_weight *= fract(probe_coord / 16.0);
            if(probe_weight < 0.9) {
                probe_weight = 0;
            }
            if(gbuffer_depth[probe_coord * 16] == BACKGROUND_DEPTH || any(probe_coord * 16 > uint2(gconst.view.viewportSize))) {
                probe_weight = 0;
            }

            sh3 shR = sh_coeficents[ZCurveToLinearIndex(uint2(probe_coord.x * 3 + 0, probe_coord.y))];
            sh3 shG = sh_coeficents[ZCurveToLinearIndex(uint2(probe_coord.x * 3 + 1, probe_coord.y))];
            sh3 shB = sh_coeficents[ZCurveToLinearIndex(uint2(probe_coord.x * 3 + 2, probe_coord.y))];
            
            float3 radiance;
            if(gconst.proberng == 1) {
                RandomSamplerState probe_rng = RandomSamplerState(probe_coord, 1);
                radiance = float3(probe_rng.uniform_float(), probe_rng.uniform_float(), probe_rng.uniform_float());
            }else {
                radiance = sh3UnprojectCosLobe(shR, shG, shB, surface.normal);   
            }

            probe_values[y*2+x].xyz = max(0.0, radiance);
            probe_values[y*2+x].w = probe_weight; 
        }
    }
  
    var lerp1 = lerp(probe_values[0].xyz, probe_values[1].xyz, probe_values[1].w / probe_values[0].w);
    
    var lerp2 = lerp(probe_values[2].xyz, probe_values[3].xyz, probe_values[3].w / probe_values[2].w);

    var radiance = lerp(lerp1.xyz, lerp2.xyz, (probe_values[3].w + probe_values[2].w) / (probe_values[1].w + probe_values[0].w)); //TODOOOOO

    if((abs(probe_values[0].w) + abs(probe_values[1].w) + abs(probe_values[2].w) + abs(probe_values[3].w)) < 0.001) {
        radiance = float3(1.0, 0.0, 0.0); //Interpolation failed
    }

    radiance *= surface.albedo;
    radiance += surface.emissive;

    Light[pixel_coord] = select(all(pixel_coord == local_probe_coord*16), float4(1.0, 0.0, 0.0, 0.0), float4(radiance, 1.0));
}