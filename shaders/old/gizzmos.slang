#include "bindless_descriptor.slang"

[shader("fragment")]
float4 fragment() : SV_Target
{
    return float4(0.0, 0.0, 0.0, 1.0);
}

struct MeshOutput {
    float4 position : SV_Position;
}

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void mesh(out indices uint3 triangles[1], out vertices MeshOutput vertices[3]) {
    // Must be called before writing the geometry output
    SetMeshOutputCounts(3, 1); // 3 vertices, 1 primitive

    triangles[0] = uint3(0, 1, 2);
    vertices[0].position = float4(-0.5, 0.5, 0.0, 1.0);
    vertices[1].position = float4(0.5, 0.5, 0.0, 1.0);
    vertices[2].position = float4(0.0, -0.5, 0.0, 1.0);
}