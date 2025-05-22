#include "bindless.slang"
#include "random.slang"

struct Bindings{
    Texture4 depth;
    Texture4 color;
    Image4 out;
}


[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 threadId : SV_DispatchThreadID) {
    var bindings = loadBindings<Bindings>();
    // var rng = RandomSamplerState(threadId.xy, 0);

    // if(all(threadId.xy == 0)) {
    //     printf("depth: %i, color: %i, out: %i \n", bindings.depth.handle.index(), bindings.color.handle.index(), bindings.out.handle.index());
    // }

    bindings.out[threadId.xy] = bindings.color[threadId.xy];
}