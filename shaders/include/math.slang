#pragma once

#define TAU 6.28318530717958647692528676655900576
#define FRAC_1_PI 0.318309886183790671537767526745028724

float2 direction_to_equirect_uv(float3 normalizedDirection)
{
    float u = (0.5 + atan2(normalizedDirection.z, normalizedDirection.x)/(2*float.getPi()));
    float v = (0.5 - asin(normalizedDirection.y)/float.getPi());

    return float2(u, v);
}

float max3(float x, float y, float z) {
    return max(x, max(y, z));
}

float square(float x) { return x * x; }
float2 square(float2 x) { return x * x; }
float3 square(float3 x) { return x * x; }
float4 square(float4 x) { return x * x; }

float length_squared(float2 v) { return dot(v, v); }
float length_squared(float3 v) { return dot(v, v); }
float length_squared(float4 v) { return dot(v, v); }

// Building an Orthonormal Basis, Revisited
// http://jcgt.org/published/0006/01/01/
float3x3 build_orthonormal_basis(float3 n) {
    float3 b1;
    float3 b2;

    if (n.z < 0.0) {
        const float a = 1.0 / (1.0 - n.z);
        const float b = n.x * n.y * a;
        b1 = float3(1.0 - n.x * n.x * a, -b, n.x);
        b2 = float3(b, n.y * n.y * a - 1.0, -n.y);
    } else {
        const float a = 1.0 / (1.0 + n.z);
        const float b = -n.x * n.y * a;
        b1 = float3(1.0 - n.x * n.x * a, b, -n.x);
        b2 = float3(b, 1.0 - n.y * n.y * a, -n.y);
    }

    return float3x3(
        b1.x, b2.x, n.x,
        b1.y, b2.y, n.y,
        b1.z, b2.z, n.z
    );
}

float3 uniform_sample_cone(float2 urand, float cos_theta_max) {
    float cos_theta = (1.0 - urand.x) + urand.x * cos_theta_max;
    float sin_theta = sqrt(saturate(1.0 - cos_theta * cos_theta));
    float phi = urand.y * TAU;
    return float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

// Calculates vector d such that
// lerp(a, d.rgb, d.a) equals lerp(lerp(a, b.rgb, b.a), c.rgb, c.a)
//
// Lerp[a_, b_, c_] := a  (1-c) + b  c
// FullSimplify[Lerp[a,(b(c (1 -  e)) + d e) /(c + e - c e), 1-(1-c)(1-e)]] == FullSimplify[Lerp[Lerp[a, b, c], d, e]]
float4 prelerp(float4 b, float4 c) {
    float denom = b.a + c.a * (1.0 - b.a);
    return select(denom > 1e-5, float4(
        (b.rgb * (b.a * (1.0 - c.a)) + c.rgb * c.a) / denom,
        1.0 - (1.0 - b.a) * (1.0 - c.a)
    ), 0.0);
}

float inverse_depth_relative_diff(float primary_depth, float secondary_depth) {
    return abs(max(1e-20, primary_depth) / max(1e-20, secondary_depth) - 1.0);
}

// Encode a scalar a space which heavily favors small values.
float exponential_squish(float len, float squish_scale) {
    return exp2(-clamp(squish_scale * len, 0, 100));
}

// Ditto, decode.
float exponential_unsquish(float len, float squish_scale) {
    return max(0.0, -1.0 / squish_scale * log2(1e-30 + len));
}

float3 uniform_sample_hemisphere(float2 urand) {
     float phi = urand.y * TAU;
     float cos_theta = 1.0 - urand.x;
     float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
     return float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

float3 uniform_sample_sphere(float2 urand) {
    float z = 1.0 - 2.0 * urand.x;
    float xy = sqrt(max(0.0, 1.0 - z * z));
    float sn = sin(TAU * urand.y);
	float cs = cos(TAU * urand.y);
	return float3(cs * xy, sn * xy, z);
}

float inverse_lerp(float minv, float maxv, float v) {
    return (v - minv) / (maxv - minv);
}

uint integer_explode(uint x)
{
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    return x;
}

uint ZCurveToLinearIndex(uint2 xy)
{
    return integer_explode(xy[0]) | (integer_explode(xy[1]) << 1);
}


float luminance(float3 color)
{
    return dot(color.xyz, float3(0.299f, 0.587f, 0.114f));
}

void sort<T: IComparable>(uint i, inout T array[]) {
    for(uint k = 2; k <= 32; k *= 2) {
        for (uint j = k/2; j > 0; j /= 2){
            int l = i ^ j;
            if(l>i) {
                if(((i & k) == 0) && (array[i] > array[l])
                    || ((i & k) != 0) && (array[i] < array[l])) {
                        var temp = array[i];
                        array[i] = array[l];
                        array[l] = temp;
                    }
            } 
            GroupMemoryBarrierWithGroupSync();
        }
    }
}