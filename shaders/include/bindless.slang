#pragma once

#define NUM_STATIC_SAMPLERS 4
struct Offsets {
    uint32_t constants;
    uint32_t bindings;
    RenderResourceHandle bindings_buffer;
    RenderResourceHandle constants_buffer;

}
[[vk::push_constant]] ConstantBuffer<Offsets> g_offsets;

[vk::binding(0, 0)] RWByteAddressBuffer bufferHandles[];
[vk::binding(0, 1)] RWTexture2D<float4> imageHandlesf4[];
[vk::binding(0, 1)] RWTexture2D<float2> imageHandlesf2[];
[vk::binding(0, 1)] RWTexture2D<float> imageHandlesf1[];
[vk::binding(NUM_STATIC_SAMPLERS, 2)] Texture2D<float4> textureHandles4[];
[vk::binding(NUM_STATIC_SAMPLERS, 2)] Texture2D<float3> textureHandles3[];
[vk::binding(NUM_STATIC_SAMPLERS, 2)] Texture2D<float> textureHandles1[];
[vk::binding(0, 2)] SamplerState samplerHandles[];
[vk::binding(0, 3)] RaytracingAccelerationStructure tlasHandles[];

T loadBindings<T>() {
    return bufferHandles[g_offsets.bindings_buffer.index].Load<T>(g_offsets.bindings);
}


T loadConstants<T>() {
    return bufferHandles[g_offsets.constants_buffer.index].Load<T>(g_offsets.constants);
}


struct RenderResourceHandle {
    uint index;
}

struct Buffer<T> {
    RenderResourceHandle handle;

    __subscript(int i) -> T
    {
        get { return bufferHandles[handle.index].Load<T>(sizeof(T) * i); }
        set { bufferHandles[handle.index].Store<T>(sizeof(T) * i, newValue); }
    }
}

struct Image4 {
    RenderResourceHandle handle;

    __subscript(vector<uint, 2> i) -> vector<float, 4>
    {
        get { return imageHandlesf4[handle.index][i]; }
        set { imageHandlesf4[handle.index][i] = newValue; }
    }
}

struct Image2 {
    RenderResourceHandle handle;

    __subscript(int2 i) -> vector<float, 2>
    {
        get { return imageHandlesf2[handle.index][i]; }
        set { imageHandlesf2[handle.index][i] = newValue; }
    }
}

struct Image1 {
    RenderResourceHandle handle;

    __subscript(int2 i) -> float
    {
        get { return imageHandlesf1[handle.index][i]; }
        set { imageHandlesf1[handle.index][i] = newValue; }
    }
}


struct Texture4 {
    RenderResourceHandle handle;

    __subscript(int2 i) -> vector<float, 4>
    {
        get { return textureHandles4[handle.index][i]; }
    }
    
    float4 sampleLevel2D(SamplerState s, float2 uv, float mip) {
        return textureHandles4[handle.index].SampleLevel(s, uv, mip);
    }
}

struct Texture3 {
    RenderResourceHandle handle;

    __subscript(int2 i) -> vector<float, 3>
    {
        get { return textureHandles3[handle.index][i]; }
    }

    float3 sampleLevel2D(SamplerState s, float2 uv, float mip) {
        return textureHandles3[handle.index].SampleLevel(s, uv, mip);
    }
}

struct Texture1 {
    RenderResourceHandle handle;

    __subscript(int2 i) -> float
    {
        get { return textureHandles1[handle.index][i]; }
    }

    float3 sampleLevel2D(SamplerState s, float2 uv, float mip) {
        return textureHandles1[handle.index].SampleLevel(s, uv, mip);
    }
}
