#!/bin/sh

base=$(dirname "$0")

function compile {
    for file in $1/*; do
        ./slang/bin/slangc $file -I$base/shaders/include -profile glsl_460 -target spirv -o $base/shaders/bin/$(basename $file).spv
    done
}

compile $base/shaders/passes
