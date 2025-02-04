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
groupshared float pdfs[8*8];
groupshared uint pdf_thread_ids[8*8];
groupshared uint supersample_ray_index;

[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 threadId : SV_DispatchThreadID, uint2 local_thread_id : SV_GroupThreadID, uint2 work_group_index : SV_GroupID) {
    var thread_index = local_thread_id.y*8+local_thread_id.x;

    [unroll]
    for(int y = 0; y < 2; ++y) {
        [unroll]
        for(int x = 0; x < 2; ++x) {
            normals[thread_index * 4 + y*2+x] = GbufferDataPacked::from_uint4(gbuffer[threadId.xy * 2 + uint2(x, y)]).unpack_normal();
        }
    }
    supersample_ray_index = 0;
    GroupMemoryBarrierWithGroupSync();
    var direction_2d = float2(uint(thread_index / 8), thread_index % 8);
    float3 ray_direction = octa_decode((direction_2d + 0.5) / 8.0);

    var brdf_pdf = 0.0;
    for(int i = 0; i < 16*16; i++) {
        brdf_pdf += max(dot(normals[i], ray_direction), 0.0) / 256.0;
    }

    RandomSamplerState rng = RandomSamplerState(threadId.xy, 0);
    var light_pdf = luminance(probe_atlas[threadId.xy].xyz); 
    pdfs[thread_index] = brdf_pdf;// * light_pdf;
    
    sort(thread_index, 64, pdfs, pdf_thread_ids);
    
    if (brdf_pdf < 0.3) {
        uint index;
        InterlockedAdd(supersample_ray_index, 1, index);
        thread_index = pdf_thread_ids[index >> 4];
        // var direction_2d = float2(uint(thread_index / 8), thread_index % 8);
        // float3 ray_direction = octa_decode((direction_2d + 0.5) / 8.0);
        out[uint2(thread_index / 8, thread_index % 8)] = uint16_t(0u << 15u) | uint16_t(thread_index);
        debug[threadId.xy] = 1.0;
    }else {
        out[threadId.xy] = uint16_t(0u << 15u) | uint16_t(thread_index);
        debug[threadId.xy] = 0.0;
    }

}