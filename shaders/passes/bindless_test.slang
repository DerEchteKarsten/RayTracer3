#include "bindless.slang"
#include "datatypes.slang"

struct Bindings{
    UniformBuffer<GConst> constants;
    Image4 out;
}


[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 threadId : SV_DispatchThreadID) {
    let bindings = loadBindings<Bindings>();

    printf("%i", bindings.constants.value.frame_index);

    bindings.out.Store(threadId.xy, float4(1.0, 1.0, 1.0, 1.0));
}