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

struct PlanarViewConstants
{
    float4x4 matWorldToView;
    float4x4 matViewToClip;
    float4x4 matWorldToClip;
    float4x4 matClipToView;
    float4x4 matViewToWorld;
    float4x4 matClipToWorld;

    float2 viewportOrigin;
    float2 viewportSize;

    float2 viewportSizeInv;
    float2 pixelOffset;

    float2 clipToWindowScale;
    float2 clipToWindowBias;

    float2 windowToClipScale;
    float2 windowToClipBias;

	float4 cameraDirectionOrPosition;
};

struct GConst {
    PlanarViewConstants view;
    uint frame_index;
    float blendfactor;
    uint bounces;
    uint samples;
};

struct RayPayload {
    float depth;
    float2 uv;
    uint geometryIndex;
    uint primitiveId;
};