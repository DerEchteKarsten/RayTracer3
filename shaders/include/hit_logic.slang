#pragma once
#include "bindless_descriptor.slang"
#include "gbuffer_helpers.slang"

GbufferData hit_info(
    RayPayload payload,
) {
    GeometryInfo geometryInfo = GeometryInfos.Load(NonUniformResourceIndex(payload.geometryIndex));

    uint vertexOffset = geometryInfo.vertexOffset;
    uint indexOffset = geometryInfo.indexOffset + (3 * payload.primitiveId);

    uint i0 = vertexOffset + Indices.Load(NonUniformResourceIndex(indexOffset));
    uint i1 = vertexOffset + Indices.Load(NonUniformResourceIndex(indexOffset + 1));
    uint i2 = vertexOffset + Indices.Load(NonUniformResourceIndex(indexOffset + 2));

    Vertex v0 = Vertices.Load(NonUniformResourceIndex(i0));
    Vertex v1 = Vertices.Load(NonUniformResourceIndex(i1));
    Vertex v2 = Vertices.Load(NonUniformResourceIndex(i2));

    const float3 barycentricCoords = float3(1.0f - payload.uv.x - payload.uv.y, payload.uv.x, payload.uv.y);
    float3 normal = normalize(v0.normal * barycentricCoords.x + v1.normal * barycentricCoords.y + v2.normal * barycentricCoords.z);
    normal = normalize(mul(geometryInfo.transform, float4(normal, 0.0)).xyz);

    float2 uvs = v0.uvs * barycentricCoords.x + v1.uvs * barycentricCoords.y + v2.uvs * barycentricCoords.z;

    float3 vertexColor = v0.color * barycentricCoords.x + v1.color * barycentricCoords.y + v2.color * barycentricCoords.z;
    var color = geometryInfo.baseColor.rgb * vertexColor;

    if (geometryInfo.baseColorTextureIndex > -1) {
        color = color * Textures[NonUniformResourceIndex(geometryInfo.baseColorTextureIndex)].SampleLevel(uvs, 0.0).xyz;
    };
    GbufferData outData;
    outData.metalness = geometryInfo.metallicFactor;
    outData.roughness = geometryInfo.roughness;
    outData.emissive = geometryInfo.emission.xyz * 12.0;
    outData.normal = normal;
    outData.albedo = color;
    return outData;
}