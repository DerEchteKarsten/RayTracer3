#include "spherical_harmonics.slang"
#include "packing.slang"
#include "math.slang"
#include "random.slang"
#include "bindless_descriptor.slang"

[[vk::binding(0, 2)]] RWTexture2D<float4> probe_atlas;
[[vk::binding(0, 1)]] RWTexture2D<uint4> gbuffer;
[[vk::binding(1, 1)]] RWTexture2D<float> gbuffer_depth;
[[vk::binding(2, 1)]] RWTexture2D<uint16_t> out;

[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 threadId : SV_DispatchThreadID, uint2 local_thread_id : SV_GroupThreadID, uint2 work_group_index : SV_GroupID) {
    if(any(threadId.xy > gconst.view.viewportSize)) {
        return;
    }
    var ray_rng = RandomSamplerState(threadId.xy, gconst.frame_index);
    out[threadId.xy] = (uint16_t(1) << 14) | uint16_t(local_thread_id.y * 8 + local_thread_id.x);
}