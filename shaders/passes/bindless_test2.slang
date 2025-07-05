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

struct MeshOutput {
    float4 position : SV_Position;
}

struct Bindings{
    Buffer<Vertex> vertecies;
    Buffer<uint8_t> indecies;
    Buffer<Meshlet> meshlets;
}

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(64, 1, 1)]
void mesh(out OutputIndices<uint3, 124> indices, out OutputVertices<MeshOutput, 64> vertices, uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID) {
    let g_Const = loadConstants<GConst>();
    let bindings = loadBindings<Bindings>();

    let meshlet = bindings.meshlets[groupID.x];
    SetMeshOutputCounts(meshlet.vertex_count, meshlet.triangle_count); 
    for (uint i = localID.x; i < meshlet.vertex_count; i += 64) {
        if(i > meshlet.vertex_count) {
            break;
        }
        let vertex = bindings.vertecies[meshlet.vertex_offset + i];

        float4 worldPosition = float4(vertex.pos, 1.0);
        float4 viewPosition = mul(g_Const.view, worldPosition);
        vertices[i].position = mul(g_Const.proj, viewPosition);
    }

    for (uint i = localID.x; i < meshlet.triangle_count; i += 64) {
        if(i > meshlet.triangle_count) {
            break;
        }
        uint offset = meshlet.triangle_offset + i * 3;
        indices[i] = uint3(bindings.indecies[offset], bindings.indecies[offset + 1], bindings.indecies[offset + 2]);
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