#include "bindless_descriptor.slang"

[shader("miss")]
void main(inout RayPayload p) {
    p.uv = float2(0.0, 0.0);
    p.geometryIndex = ~0u;
    p.primitiveId = ~0u;
}