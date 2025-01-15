#include "bindless_descriptor.slang"

[shader("closesthit")]
void main(inout RayPayload p, in BuiltInTriangleIntersectionAttributes attribs) {
    p.depth = RayTCurrent();
    p.uv = attribs.barycentrics;
    p.geometryIndex = GeometryIndex();
    p.primitiveId = PrimitiveIndex();
}