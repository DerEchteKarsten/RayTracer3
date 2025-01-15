#pragma once

#include "datatypes.slang"
#include "random.slang"

[[vk::binding(0, 0)]] RaytracingAccelerationStructure SceneBVH;
[[vk::binding(1, 0)]] StructuredBuffer<GeometryInfo> GeometryInfos;
[[vk::binding(2, 0)]] StructuredBuffer<Vertex> Vertices;
[[vk::binding(3, 0)]] StructuredBuffer<uint> Indices;
[[vk::binding(4, 0)]] Sampler2D Textures[];
[[vk::binding(5, 0)]] Sampler2D Skybox;
[[vk::binding(6, 0)]] ConstantBuffer<GConst> gconst;
[[vk::binding(7, 0)]] Texture2D<float4> BlueNoise;

void trace(RayDesc ray, inout RayPayload payload) {
    TraceRay(SceneBVH, RAY_FLAG_FORCE_OPAQUE, 0xff, 0, 0, 0, ray, payload);
}

float4 blue_noise_for_pixel(uint2 px, uint n) {
    const uint2 tex_dims = uint2(256, 256);
    const uint2 offset = uint2(r2_sequence(n) * float2(tex_dims));

    return BlueNoise[
        (px + offset) % tex_dims
    ] * 255.0 / 256.0 + 0.5 / 256.0;
}
