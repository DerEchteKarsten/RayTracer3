#!/bin/sh

base=$(dirname "$0")

function compile {
    for file in $1/*; do
        ./slang/bin/slangc $file -I$base/shaders/include -profile spirv_1_6 -target spirv -o $base/shaders/bin/$(basename $file).spv -fvk-use-entrypoint-name
    done
}

compile $base/shaders/passes
