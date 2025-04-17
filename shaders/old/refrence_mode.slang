#include "bindless_descriptor.slang"
#include "hit_logic.slang"
#include "gbuffer_helpers.slang"
#include "math.slang"
#include "brdf.slang"
#include "random.slang"

[[vk::binding(0, 1)]] RWTexture2D<uint4> gbuffer;
[[vk::binding(1, 1)]] RWTexture2D<float> gbuffer_depth;
[[vk::binding(0, 2)]] RWTexture2D<float4> Light;
[[vk::binding(0, 3)]] RWTexture2D<float4> PrevLight;


[shader("raygeneration")]
void main() {
    RayPayload payload;
    var pixelCoord = DispatchRaysIndex().xy;
    var gbuffer_depth = gbuffer_depth[pixelCoord];
    if(gbuffer_depth == BACKGROUND_DEPTH) {
        return;
    }
    
    var gbuffer_data = GbufferDataPacked::from_uint4(gbuffer[pixelCoord]).unpack();

    var rng = RandomSamplerState(pixelCoord, gconst.frame_index);

    var radiance = float3(0.0);
    for(int i = 0; i < gconst.samples; i++) {
        var ray_color = float3(1.0);
        RayDesc ray = setupPrimaryRay(pixelCoord, gconst.view);
        ray.TMin = 0.001;
        var surface = gbuffer_data;
        payload.depth = gbuffer_depth;
        payload.geometryIndex = 0;

        for(int b = 0; b < gconst.bounces; b++){
            if(payload.geometryIndex == ~0u) {
                // radiance += ray_color * Skybox.SampleLevel(direction_to_equirect_uv(ray.Direction), 0.0).xyz;
                break;
            }

            DiffuseBrdf brdf = {surface.albedo};
            var urand = float2(rng.uniform_float(), rng.uniform_float());
            var tangent_to_world = build_orthonormal_basis(surface.normal);
            BrdfSample sample = brdf.sample(ray.Direction, urand);

            ray.Origin += payload.depth * ray.Direction;
            ray.Direction = mul(tangent_to_world, sample.wi);
            
            radiance += ray_color * surface.emissive;
			ray_color *= sample.value_over_pdf;

            if(b != gconst.bounces - 1) {   
                trace(ray, payload);
                surface = hit_info(payload);
            }
        }
    }
    radiance /= float(gconst.samples);

    if(gconst.blendfactor >= 1.0) {
        Light[pixelCoord] = float4(radiance, 0.0);
    }else {
        Light[pixelCoord] = float4(lerp(PrevLight[pixelCoord].xyz, radiance, gconst.blendfactor), 0.0);
    }
}