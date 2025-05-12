#include "bindless.slang"

struct Bindings{
    Image4 in1;
    Image4 in2;
    Image4 out;
}


[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 threadId : SV_DispatchThreadID) {
    var bindings = loadBindings<Bindings>();

    bindings.out[threadId.xy] = bindings.in1[threadId.xy] + bindings.in2[threadId.xy];
}