#pragma once
#define M_PLASTIC 1.32471795724474602596
#include "math.slang"

uint hash(uint a)
{
    // http://burtleburtle.net/bob/hash/integer.html
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

float radical_inverse_vdc(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

float2 hammersley(uint i, uint n) {
    return float2(float(i + 1) / n, radical_inverse_vdc(i + 1));
}

float2 r2_sequence(uint i) {
    const float a1 = 1.0 / M_PLASTIC;
    const float a2 = 1.0 / (M_PLASTIC * M_PLASTIC);
    
    return frac(float2(a1, a2) * i + 0.5);
}

struct RandomSamplerState
{
    uint seed;
    uint index;

    __init(uint2 pixelPos, uint frameIndex) {
        var linear_index = ZCurveToLinearIndex(pixelPos);
        index = 0;
        seed = hash(linear_index) + frameIndex;
    }

    [mutating]
    uint murmur3()
    {
    #define ROT32(x, y) ((x << y) | (x >> (32 - y)))
    
        // https://en.wikipedia.org/wiki/MurmurHash
        uint c1 = 0xcc9e2d51;
        uint c2 = 0x1b873593;
        uint r1 = 15;
        uint r2 = 13;
        uint m = 5;
        uint n = 0xe6546b64;
    
        uint hash = seed;
        uint k = index++;
        k *= c1;
        k = ROT32(k, r1);
        k *= c2;
    
        hash ^= k;
        hash = ROT32(hash, r2) * m + n;
    
        hash ^= 4;
        hash ^= (hash >> 16);
        hash *= 0x85ebca6b;
        hash ^= (hash >> 13);
        hash *= 0xc2b2ae35;
        hash ^= (hash >> 16);
    
    #undef ROT32
    
        return hash;
    }

    [mutating]
    float uniform_float()
    {
        uint v = murmur3();
        const uint one = asuint(1.f);
        const uint mask = (1 << 23) - 1;
        return asfloat((mask & v) | one) - 1.f;
    }
};
