#pragma once
#include "datatypes.slang"
#include "packing.slang"

struct GbufferData {
    float3 albedo;
    float3 emissive;
    float3 normal;
    float roughness;
    float metalness;

    static GbufferData create_zero() {
        GbufferData res;
        res.albedo = 0;
        res.emissive = 0;
        res.normal = 0;
        res.roughness = 0;
        res.metalness = 0;
        return res;
    }

    GbufferDataPacked pack() {
        float4 res = 0.0;
        res.x = asfloat(pack_color_888(albedo));
        res.y = pack_normal_11_10_11(normal);

        float2 roughness_metalness = float2(roughness_to_perceptual_roughness(roughness), metalness);
        res.z = asfloat(pack_2x16f_uint(roughness_metalness));
        res.w = asfloat(float3_to_rgb9e5(emissive));

        GbufferDataPacked packed;
        packed.data0 = asuint(res);
        return packed;
    }
};

struct GbufferDataPacked {
    uint4 data0;

    static GbufferDataPacked from_uint4(uint4 data0) {
        GbufferDataPacked res;
        res.data0 = data0;
        return res;
    }

    float3 unpack_normal() {
        return unpack_normal_11_10_11(asfloat(data0.y));
    }

    float3 unpack_albedo() {
        return unpack_color_888(data0.x);
    }

    float3 unpack_emissive() {
        return rgb9e5_to_float3(data0.w);
    }


    GbufferData unpack() {
        GbufferData res;
        res.albedo = unpack_albedo();
        res.normal = unpack_normal();

        float2 roughness_metalness = unpack_2x16f_uint(data0.z);
        res.roughness = perceptual_roughness_to_roughness(roughness_metalness.x);
        res.metalness = roughness_metalness.y;
        res.emissive = unpack_emissive();

        return res;
    }
};

float roughness_to_perceptual_roughness(float r) {
    return sqrt(r);
}

float perceptual_roughness_to_roughness(float r) {
    return r * r;
}

float3 view_dir(uint2 pixelCoord, PlanarViewConstants view) {
    const var pixelCenter = float2(pixelCoord) + float2(0.5);
    const var inUV = pixelCenter/float2(view.viewportSize);
    var d = inUV * 2.0 - 1.0;
    var dir = inUV * 2.0 - 1.0;
    var target = mul(view.matClipToView, float4(dir.x, dir.y, 1, 1));
    var direction = mul(view.matViewToWorld, float4(normalize(target.xyz), 0));
    return direction.xyz;
}

RayDesc setupPrimaryRay(uint2 pixelCoord, PlanarViewConstants view)
{
    RayDesc ray;
    ray.Origin = view.cameraDirectionOrPosition.xyz;
    ray.Direction = view_dir(pixelCoord, view);
    ray.TMin = 0;
    ray.TMax = BACKGROUND_DEPTH;
    return ray;
}
