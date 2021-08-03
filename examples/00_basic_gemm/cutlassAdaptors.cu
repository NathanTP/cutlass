//#include "helper.h"
#include "cutlassAdaptors.h"

extern "C"
CutlassGemm::GemmKernel::Params *adaptSGEMMArgs(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {
  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  CutlassGemm::Arguments args({M , N, K},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
                              {C, ldc},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta}); // Scalars used in the Epilogue

  // Launch the CUTLASS GEMM kernel.
  cudaStream_t stream = nullptr;
  gemm_operator.initialize(args, stream=stream);
  CutlassGemm::GemmKernel::Params params_ = gemm_operator.get_params();
  CutlassGemm::GemmKernel::Params *params_ptr = (CutlassGemm::GemmKernel::Params*) malloc(sizeof(params_));
  memcpy(params_ptr, &params_, sizeof(params_));
  return params_ptr;
}

extern "C"
CudaConfig *getCudaConfig(CutlassGemm::GemmKernel::Params *params_ptr) {
  CutlassGemm::ThreadblockSwizzle threadblock_swizzle;
  dim3 grid = threadblock_swizzle.get_grid_shape(params_ptr->grid_tiled_shape);
  dim3 block(CutlassGemm::GemmKernel::kThreadCount, 1, 1);
  int smem_size = int(sizeof(typename CutlassGemm::GemmKernel::SharedStorage));
  CudaConfig* ptr = new CudaConfig;
  ptr->grid_ = grid;
  ptr->block_ = block;
  ptr->smem_size_ = smem_size;
  return ptr;
}