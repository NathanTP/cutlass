#!/bin/bash

if [ -z "$CUTLASS_PATH" ]; then
    CUTLASS_PATH=../..
fi

CUTLASS_PATH=$(realpath $CUTLASS_PATH)

# Build for adaptor shared library (.so)
/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler \
-I$CUTLASS_PATH/include \
-I$CUTLASS_PATH/examples/00_basic_gemm \
-I$CUTLASS_PATH/g/examples/common \
-I$CUTLASS_PATH/g/build/include \
-I/usr/local/cuda/include \
-I$CUTLASS_PATH/g/tools/util/include \
-O0 -DNDEBUG -Xcompiler=-fPIC \
-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_DEBUG_TRACE_LEVEL=0 \
-Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing \
-gencode=arch=compute_35,code=sm_35 -std=c++11 \
-x cu -o out.so --shared cutlassAdaptors.cu

# Build kernel (.cubin)
/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler \
-I$CUTLASS_PATH/include \
-I$CUTLASS_PATH/examples/00_basic_gemm \
-I$CUTLASS_PATH/g/examples/common \
-I$CUTLASS_PATH/g/build/include \
-I/usr/local/cuda/include \
-I$CUTLASS_PATH/g/tools/util/include \
-O0 -DNDEBUG -Xcompiler=-fPIC \
--cubin \
-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_DEBUG_TRACE_LEVEL=0 \
-Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing \
-gencode=arch=compute_35,code=sm_35 -std=c++11 \
-x cu -c kern.cu -o kern.cubin