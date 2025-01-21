#include "spherical_harmonics.slang"
#include "packing.slang"
#include "math.slang"

[[vk::binding(0, 1)]] RWTexture2D<float4> probe_atlas;
[[vk::binding(0, 0)]] RWStructuredBuffer<float3x3> out;

[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 threadId : SV_DispatchThreadID, uint2 local_thread_id : SV_GroupThreadID, uint2 work_group_index : SV_GroupID) {
    float3 local_ray_direction = octa_decode(float2(local_thread_id) / 8.0);
    float3 ray_color = probe_atlas[threadId.xy].xyz;
    sh3 sh = sh3Evaluate(local_ray_direction);

    sh3 shR = WaveActiveSum(sh3Scale(sh, ray_color.x));
    sh3 shG = WaveActiveSum(sh3Scale(sh, ray_color.y));
    sh3 shB = WaveActiveSum(sh3Scale(sh, ray_color.z));

    if(WaveIsFirstLane()) {
        float shFactor = 4.0 * shPI / (8 * 8);
        shR = sh3Scale(shR, shFactor);
		shG = sh3Scale(shG, shFactor);
		shB = sh3Scale(shB, shFactor);

        out[ZCurveToLinearIndex(uint2(work_group_index.x * 3 + 0, work_group_index.y))] = shR;
        out[ZCurveToLinearIndex(uint2(work_group_index.x * 3 + 1, work_group_index.y))] = shG;
        out[ZCurveToLinearIndex(uint2(work_group_index.x * 3 + 2, work_group_index.y))] = shB;
    }
}