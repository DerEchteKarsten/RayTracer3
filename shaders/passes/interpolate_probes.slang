#include "bindless_descriptor.slang"

[[vk::binding(0, 1)]] RWTexture2D<uint4> gbuffer;
[[vk::binding(1, 1)]] RWTexture2D<float> gbuffer_depth;
[[vk::binding(2, 1)]] RWTexture2D<float> probe_atlas;
[[vk::binding(0, 2)]] RWTexture2D<float4> Light;


[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 threadId : SV_DispatchThreadID) {
    uint2 pixelCoord = threadId.xy;
    var probeCoord = float2(pixelCoord) / 16.0;
    uint2 probeCoords[] = {
        uint2(uint(floor(probeCoord.x)), floor(probeCoord.y)), 
        uint2(uint(ceil(probeCoord.x)), floor(probeCoord.y)), 
        uint2(uint(floor(probeCoord.x)), ceil(probeCoord.y)), 
        uint2(uint(ceil(probeCoord.x)), ceil(probeCoord.y))
    };

    var gbuffer_depth = gbuffer_depth[pixelCoord];
    if(gbuffer_depth == BACKGROUND_DEPTH) {
        return;
    }
}