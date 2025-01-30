#include "math.slang"
#include "random.slang"
#include "bindless_descriptor.slang"
#include "brdf.slang"
#include "gbuffer_helpers.slang"

[[vk::binding(0, 2)]] RWTexture2D<float4> probe_atlas;
[[vk::binding(0, 1)]] RWTexture2D<uint4> gbuffer;
[[vk::binding(1, 1)]] RWTexture2D<float> gbuffer_depth;
[[vk::binding(2, 1)]] RWTexture2D<uint16_t> out;
[[vk::binding(3, 1)]] RWTexture2D<float> debug;


groupshared float3 normals[16*16];
groupshared float combined_pdfs[8*8];
groupshared float brdf_pdfs[8*8];

groupshared uint2 culled_rays[];

[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 threadId : SV_DispatchThreadID, uint2 local_thread_id : SV_GroupThreadID, uint2 work_group_index : SV_GroupID) {
    [unroll]
    for(int x = 0; x < 2; ++x) {
        [unroll]
        for(int y = 0; y < 2; ++y) {
            normals[local_thread_id.y * 16 + local_thread_id.x * 2 + y*2+x] = GbufferDataPacked::from_uint4(gbuffer[threadId.xy * 2 + uint2(x, y)]).unpack_normal();
        }
    }
    GroupMemoryBarrierWithGroupSync();
    var thread_index = local_thread_id.y*8+local_thread_id.x;
    var direction_2d = float2(uint(thread_index / 8), thread_index % 8);
    float3 ray_direction = octa_decode((direction_2d + 0.5) / 8.0);

    var brdf_pdf = 0.0;
    for(int i = 0; i < 64; i++) {
        brdf_pdf += max(dot(normals[i], ray_direction), 0.0) / 64.0;
    }

    var light_pdf = luminance(probe_atlas[threadId.xy].xyz); 
    combined_pdfs[thread_index] = brdf_pdf * light_pdf;
    brdf_pdfs[thread_index] = brdf_pdf;
    
    sort(thread_index, combined_pdfs);
    sort(thread_index, brdf_pdfs);

    if (brdf_pdfs[thread_index] < 0.4) {

    }

    var ray_rng = RandomSamplerState(threadId.xy, gconst.frame_index);
    out[threadId.xy] = uint16_t(0u << 14u) | uint16_t(local_thread_id.y * 8 + local_thread_id.x);
    debug[threadId.xy] = combined_pdfs[local_thread_id.y * 8 + local_thread_id.x];
}