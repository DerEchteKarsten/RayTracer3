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

const static float3 Vertices[] = {
    float3(0.0, -0.5, 0.0),
    float3(0.5, 0.5, 0.0),
    float3(-0.5, 0.5, 0.0),
};

struct MeshOutput {
    float4 position : SV_Position;
}

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void mesh(out OutputIndices<uint3, 1> indices, out OutputVertices<MeshOutput, 3> vertices) {
    let g_Const = loadConstants<GConst>();
    // Must be called before writing the geometry output
    SetMeshOutputCounts(3, 1); // 3 vertices, 1 primitive
    indices[0] = uint3(0, 1, 2);

    for(int i = 0; i<3; i++) {
        float4 worldPosition = float4(Vertices[i], 1.0);
        float4 viewPosition = mul(g_Const.view, worldPosition);
        vertices[i].position = mul(g_Const.proj, viewPosition);
    }
}

// struct Out {
//     float4 position : SV_Position;
// }


// [shader("mesh")]
// Out mesh(uint vertexid: SV_VertexID) {
//     
//     Out out;


//     // printf("%f %f %f %f \n", out.position.x, out.position.y, out.position.z, out.position.w);
//     return out;
// }