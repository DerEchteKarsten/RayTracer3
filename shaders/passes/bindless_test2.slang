#include "bindless.slang"
#include "random.slang"
#include "datatypes.slang"
// struct Bindings{
//     Image1 out;
// }

struct Fragment
{
    float4 color;
};

[shader("fragment")]
Fragment fragment() : SV_Target
{
    Fragment out;
    out.color = float4(1.0, 0.0, 1.0, 1.0); 
    return out;
}

// struct MeshOutput {
//     float4 position : SV_Position;
// }

// [shader("mesh")]
// [outputtopology("triangle")]
// [numthreads(1, 1, 1)]
// void mesh(out indices uint3 triangles[1], out vertices MeshOutput vertices[3]) {
//     // Must be called before writing the geometry output
//     SetMeshOutputCounts(3, 1); // 3 vertices, 1 primitive

//     triangles[0] = uint3(0, 1, 2);
//     vertices[0].position = float4(-0.5, 0.5, 0.0, 1.0);
//     vertices[1].position = float4(0.5, 0.5, 0.0, 1.0);
//     vertices[2].position = float4(0.0, -0.5, 0.0, 1.0);
// }

struct Out {
    float4 position : SV_Position;
}

const static float3 vertices[] = {
    float3(0.0, -0.5, 0.0),
    float3(0.5, 0.5, 0.0),
    float3(-0.5, 0.5, 0.0),
};

[shader("vertex")]
Out vertex(uint vertexid: SV_VertexID) {
    let g_Const = loadConstants<GConst>();
    Out out;

    float4 worldPosition = float4(vertices[vertexid], 1.0);
    float4 viewPosition = mul(g_Const.view, worldPosition);
    out.position = mul(g_Const.proj, viewPosition);

    // printf("%f %f %f %f \n", out.position.x, out.position.y, out.position.z, out.position.w);

    return out;
}