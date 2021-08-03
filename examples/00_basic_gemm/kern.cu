// #include "cutlassAdaptors.h"

// extern "C"
// template __global__ void cutlass::Kernel<CutlassGemm::GemmKernel>(CutlassGemm::GemmKernel::Params);
// __global__ void MyKernel(CutlassGemm::GemmKernel::Params *params_ptr, dim3 grid, dim3 block, int smem_size) {
//     cudaStream_t stream = nullptr;
//     cutlass::Kernel<CutlassGemm::GemmKernel><<<grid, block, smem_size, stream>>>(*params_ptr);
// }
#include "cutlass/gemm/device/gemm.h"

using ColumnMajor = cutlass::layout::ColumnMajor;

using CutlassGemm = cutlass::gemm::device::Gemm<float,
                                                ColumnMajor,
                                                float,
                                                ColumnMajor,
                                                float,
                                                ColumnMajor>;

// This is a template kernel
extern "C" {
    template __global__ void cutlass::Kernel<CutlassGemm::GemmKernel>(CutlassGemm::GemmKernel::Params);
}
// __host__ void MyKernel(CutlassGemm::GemmKernel::Params *params_ptr, dim3 grid, dim3 block, int smem_size) {
//     cudaStream_t stream = nullptr;
//     cutlass::Kernel<CutlassGemm::GemmKernel><<<grid, block, smem_size, stream>>>(*params_ptr);
// }