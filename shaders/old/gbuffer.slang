#include "bindless_descriptor.slang"
#include "hit_logic.slang"
#include "gbuffer_helpers.slang"

[[vk::binding(0, 1)]] RWTexture2D<uint4> gbuffer;
[[vk::binding(1, 1)]] RWTexture2D<float> gbuffer_depth;

[shader("raygeneration")]
void main() {
    var pixelCoord = DispatchRaysIndex().xy;
    var ray = setupPrimaryRay(pixelCoord, gconst.view);
    RayPayload payload;
    trace(ray, payload);

    if(payload.geometryIndex == ~0u) {
        gbuffer_depth[pixelCoord] = BACKGROUND_DEPTH;
    }else {
        gbuffer[pixelCoord] = hit_info(payload).pack().data0;
        gbuffer_depth[pixelCoord] = payload.depth;
    }
}