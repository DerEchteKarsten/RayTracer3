#pragma once

static const float BACKGROUND_DEPTH = 100000.0;

struct Vertex {
	float3 pos;
	float3 normal;
	float3 color;
	float2 uvs;
};

struct GeometryInfo {
	float4x4 transform;
	float4 baseColor;
	int baseColorTextureIndex;
	float metallicFactor;
	uint indexOffset;
	uint vertexOffset;
	float4 emission;
	float roughness;
};

struct GConst {
    column_major float4x4 proj;
    column_major float4x4 view;
    column_major float4x4 proj_inverse;
    column_major float4x4 view_inverse;
    float2 window_size;
    uint frame_index;
    float blendfactor;
    uint bounces;
    uint samples;

    uint proberng;
    float cell_size;
    uint2 mouse;
};

struct RayPayload {
    float depth;
    float2 uv;
    uint geometryIndex;
    uint primitiveId;
};