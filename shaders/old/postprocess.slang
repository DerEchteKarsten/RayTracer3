#include "gbuffer_helpers.slang"
#include "bindless_descriptor.slang"
#include "math.slang"

[[vk::binding(0, 1)]] RWTexture2D<float> Depth;
[[vk::binding(1, 1)]] RWTexture2D<float4> Out;
[[vk::binding(0, 2)]] RWTexture2D<float4> In;


#define AGX_LOOK 2

// Mean error^2: 3.6705141e-06
float3 agxDefaultContrastApprox(float3 x) {
  float3 x2 = x * x;
  float3 x4 = x2 * x2;
  
  return + 15.5     * x4 * x2
         - 40.14    * x4 * x
         + 31.96    * x4
         - 6.868    * x2 * x
         + 0.4298   * x2
         + 0.1191   * x
         - 0.00232;
}

float3 agx(float3 val) {
  const float3x3 agx_mat = float3x3(
    0.842479062253094, 0.0423282422610123, 0.0423756549057051,
    0.0784335999999992,  0.878468636469772,  0.0784336,
    0.0792237451477643, 0.0791661274605434, 0.879142973793104);
    
  const float min_ev = -12.47393f;
  const float max_ev = 4.026069f;

  // Input transform
  val = mul(val, agx_mat);
  
  // Log2 space encoding
  val = clamp(log2(val), min_ev, max_ev);
  val = (val - min_ev) / (max_ev - min_ev);
  
  // Apply sigmoid function approximation
  val = agxDefaultContrastApprox(val);

  return val;
}

float3 agxEotf(float3 val) {
  const float3x3 agx_mat_inv = float3x3(
    1.19687900512017, -0.0528968517574562, -0.0529716355144438,
    -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
    -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
    
  // Undo input transform
  val = mul(val, agx_mat_inv);
  
  // sRGB IEC 61966-2-1 2.2 Exponent Reference EOTF Display
  //val = pow(val, float3(2.2));

  return val;
}

float3 agxLook(float3 val) {
  const float3 lw = float3(0.2126, 0.7152, 0.0722);
  float luma = dot(val, lw);
  
  // Default
  float3 offset = float3(0.0);
  float3 slope = float3(1.0);
  float3 power = float3(1.0);
  float sat = 1.0;
 
#if AGX_LOOK == 1
  // Golden
  slope = float3(1.0, 0.9, 0.5);
  power = float3(0.8);
  sat = 0.8;
#elif AGX_LOOK == 2
  // Punchy
  slope = float3(1.0);
  power = float3(1.1, 1.1, 1.1);
  sat = 1.1;
#endif
  
  // ASC CDL
  val = pow(val * slope + offset, power);
  return luma + sat * (val - luma);
}

[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 threadId : SV_DispatchThreadID) {
    if(any(threadId.xy > gconst.view.viewportSize)) {
        return;
    }
    
    var depth = Depth[threadId.xy];
    float3 color;
    if(depth != BACKGROUND_DEPTH) {
        color = In[threadId.xy].xyz;
    }else {
        var ray = setupPrimaryRay(threadId.xy, gconst.view);
        var uv = direction_to_equirect_uv(ray.Direction);
        color = Skybox.SampleLevel(uv, 0.0).xyz;
    }

    color = agx(color);
    color = agxLook(color);
    color = agxEotf(color);

    Out[threadId.xy] = float4(color, 1.0);
}