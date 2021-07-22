// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>

// Helper methods to check for errors
#include "helper.h"

//
// CUTLASS includes needed for single-precision GEMM kernel
//

// Defines cutlass::gemm::device::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/device/gemm.h"

using ColumnMajor = cutlass::layout::ColumnMajor;

using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                ColumnMajor,  // Layout of A matrix
                                                float,        // Data-type of B matrix
                                                ColumnMajor,  // Layout of B matrix
                                                float,        // Data-type of C matrix
                                                ColumnMajor>; // Layout of C matrix

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
  
  //cutlass::Status status = gemm_operator(args);
  cudaStream_t stream = nullptr;
  gemm_operator.initialize(args, stream=stream);
  CutlassGemm::GemmKernel::Params params_ = gemm_operator.get_params();
  CutlassGemm::GemmKernel::Params *params_ptr = (CutlassGemm::GemmKernel::Params*) malloc(sizeof(params_));
  memcpy(params_ptr, &params_, sizeof(params_));
  return params_ptr;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(
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

  CutlassGemm::GemmKernel::Params *params_ptr = adaptSGEMMArgs(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  // malloc space for gemm_operator.get_params() and memcopy to heap.
  // Use pointer from malloc to reference
  // Gemm run function
  cudaStream_t stream = nullptr;
  CutlassGemm::ThreadblockSwizzle threadblock_swizzle;
  //typename CutlassGemm::GemmKernel::Params params_;
  dim3 grid = threadblock_swizzle.get_grid_shape(params_ptr->grid_tiled_shape);
  dim3 block(CutlassGemm::GemmKernel::kThreadCount, 1, 1);
  cudaError_t result;
  int smem_size = int(sizeof(typename CutlassGemm::GemmKernel::SharedStorage));
  cutlass::Kernel<CutlassGemm::GemmKernel><<<grid, block, smem_size, stream>>>(*params_ptr);
  free(params_ptr);
  result = cudaGetLastError();
  if (result != cudaSuccess) {
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// The source code after this point in the file is generic CUDA using the CUDA Runtime API
// and simple CUDA kernels to initialize matrices and compute the general matrix product.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to initialize a matrix with small integers.
__global__ void InitializeMatrix_kernel(
  float *matrix,
  int rows,
  int columns,
  int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i + j * rows;

    // Generate arbitrary elements.
    int const k = 16807;
    int const m = 16;
    float value = float(((offset + seed) * k % m) - m / 2);

    matrix[offset] = value;
  }
}

/// Simple function to initialize a matrix to arbitrary small integers.
cudaError_t InitializeMatrix(float *matrix, int rows, int columns, int seed = 0) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  InitializeMatrix_kernel<<< grid, block >>>(matrix, rows, columns, seed);

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocates device memory for a matrix then fills with arbitrary small integers.
cudaError_t AllocateMatrix(float **matrix, int rows, int columns, int seed = 0) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(float) * rows * columns;

  // Allocate device memory.
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Clear the allocation.
  result = cudaMemset(*matrix, 0, sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Initialize matrix elements to arbitrary small integers.
  result = InitializeMatrix(*matrix, rows, columns, seed);

  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive reference GEMM computation.
__global__ void ReferenceGemm_kernel(
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

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float accumulator = 0;

    for (int k = 0; k < K; ++k) {
      accumulator += A[i + k * lda] * B[k + j * ldb];
    }

    C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
  }
}

/// Reference GEMM computation.
cudaError_t ReferenceGemm(
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

  dim3 block(16, 16);
  dim3 grid(
    (M + block.x - 1) / block.x,
    (N + block.y - 1) / block.y
  );

  ReferenceGemm_kernel<<< grid, block >>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel.
cudaError_t TestCutlassGemm(int M, int N, int K, float alpha, float beta) {
  cudaError_t result;

  //
  // Define several matrices to be used as operands to GEMM kernels.
  //

  // Compute leading dimensions for each matrix.
  int lda = M;
  int ldb = K;
  int ldc = M;

  // Compute size in bytes of the C matrix.
  size_t sizeof_C = sizeof(float) * ldc * N;

  // Define pointers to matrices in GPU device memory.
  float *A;
  float *B;
  float *C_cutlass;
  float *C_reference;

  //
  // Allocate matrices in GPU device memory with arbitrary seeds.
  //

  result = AllocateMatrix(&A, M, K, 0);

  if (result !=  cudaSuccess) {
    return result;
  }

  result = AllocateMatrix(&B, K, N, 17);

  if (result !=  cudaSuccess) {
    cudaFree(A);
    return result;
  }

  result = AllocateMatrix(&C_cutlass, M, N, 101);

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    return result;
  }

  result = AllocateMatrix(&C_reference, M, N, 101);

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    cudaFree(C_cutlass);
    return result;
  }

  result = cudaMemcpy(C_reference, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy C_cutlass matrix to C_reference: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  //
  // Launch CUTLASS GEMM.
  //

  result = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc);

  if (result != cudaSuccess) {
    std::cerr << "CUTLASS GEMM kernel failed: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  //
  // Verify.
  //

  // Launch reference GEMM
  result = ReferenceGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);

  if (result != cudaSuccess) {
    std::cerr << "Reference GEMM kernel failed: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  // Copy to host and verify equivalence.
  std::vector<float> host_cutlass(ldc * N, 0);
  std::vector<float> host_reference(ldc * N, 0);

  result = cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy CUTLASS GEMM results: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  result = cudaMemcpy(host_reference.data(), C_reference, sizeof_C, cudaMemcpyDeviceToHost);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy Reference GEMM results: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  //
  // Free device memory allocations.
  //

  cudaFree(C_reference);
  cudaFree(C_cutlass);
  cudaFree(B);
  cudaFree(A);

  //
  // Test for bit equivalence of results.
  //

  if (host_cutlass != host_reference) {
    std::cerr << "CUTLASS results incorrect." << std::endl;

    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to basic_gemm example.
//
// usage:
//
//   00_basic_gemm <M> <N> <K> <alpha> <beta>
//
int main(int argc, const char *arg[]) {

  //
  // Parse the command line to obtain GEMM dimensions and scalar values.
  //

  // GEMM problem dimensions.
  int problem[3] = { 128, 128, 128 };

  for (int i = 1; i < argc && i < 4; ++i) {
    std::stringstream ss(arg[i]);
    ss >> problem[i - 1];
  }

  // Scalars used for linear scaling the result of the matrix product.
  float scalars[2] = { 1, 0 };

  for (int i = 4; i < argc && i < 6; ++i) {
    std::stringstream ss(arg[i]);
    ss >> scalars[i - 4];
  }

  //
  // Run the CUTLASS GEMM test.
  //

  cudaError_t result = TestCutlassGemm(
    problem[0],     // GEMM M dimension
    problem[1],     // GEMM N dimension
    problem[2],     // GEMM K dimension
    scalars[0],     // alpha
    scalars[1]      // beta
  );

  if (result == cudaSuccess) {
    std::cout << "Passed." << std::endl;
  }

  // Exit.
  return result == cudaSuccess ? 0 : -1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////


// #include <cutlass/numeric_types.h>
// #include <cutlass/gemm/device/gemm.h>

// #include <cutlass/util/host_tensor.h>

// int main() {

//     // Define the GEMM operation
//     using Gemm = cutlass::gemm::device::Gemm<
//     cutlass::half_t,                           // ElementA
//     cutlass::layout::ColumnMajor,              // LayoutA
//     cutlass::half_t,                           // ElementB
//     cutlass::layout::ColumnMajor,              // LayoutB
//     cutlass::half_t,                           // ElementOutput
//     cutlass::layout::ColumnMajor,              // LayoutOutput
//     float,                                     // ElementAccumulator
//     cutlass::arch::OpClassTensorOp,            // tag indicating Tensor Cores
//     cutlass::arch::Sm70                        // tag indicating target GPU compute architecture
//     >;

//     Gemm gemm_op;
//     // cutlass::Status status;

//     //
//     // Define the problem size
//     //
//     int M = 512;
//     int N = 256;
//     int K = 128;

//     float alpha = 1.25f;
//     float beta = -1.25f;

//     //
//     // Allocate device memory
//     //

//     cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A({M, K});
//     cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B({K, N});
//     cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C({M, N});

//     cutlass::half_t const *ptrA = A.device_data();
//     cutlass::half_t const *ptrB = B.device_data();
//     cutlass::half_t const *ptrC = C.device_data();
//     cutlass::half_t       *ptrD = C.device_data();

//     int lda = A.device_ref().stride(0);
//     int ldb = B.device_ref().stride(0);
//     int ldc = C.device_ref().stride(0);
//     int ldd = C.device_ref().stride(0);
//     //
//     // Launch GEMM on the device
//     //

//     Gemm::Arguments args({M , N, K},  // Gemm Problem dimensions
//                               {A, lda},    // Tensor-ref for source matrix A
//                               {B, ldb},    // Tensor-ref for source matrix B
//                               {C, ldc},    // Tensor-ref for source matrix C
//                               {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
//                               {alpha, beta}); // Scalars used in the Epilogue

//     cudaStream_t stream = nullptr;
//     gemm_op.initialize(args, stream=stream);
    
//     // Gemm run function
//     Gemm::ThreadblockSwizzle threadblock_swizzle;
//     dim3 grid = threadblock_swizzle.get_grid_shape(gemm_op.params_.grid_tiled_shape);
//     dim3 block(Gemm::GemmKernel::kThreadCount, 1, 1);
//     cudaError_t result;
//     int smem_size = int(sizeof(typename Gemm::GemmKernel::SharedStorage));
//     // Temporarily not used
//     // if (smem_size >= (48 << 10)) {
//     //     result = cudaFuncSetAttribute(Kernel<GemmKernel>,
//     //                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
//     //                                 smem_size);

//     //     if (result != cudaSuccess) {
//     //     return Status::kErrorInternal;
//     //     }

//     //     result = cudaFuncSetAttribute(
//     //         Kernel<GemmKernel>,
//     //         cudaFuncAttributePreferredSharedMemoryCarveout, 100);

//     //     if (result != cudaSuccess) {
//     //     return Status::kErrorInternal;
//     //     }
//     // }

//     cutlass::Kernel<Gemm::GemmKernel><<<grid, block, smem_size, stream>>>(gemm_op.params_);

//     result = cudaGetLastError();
//     if (result != cudaSuccess) {
//         return -1;
//     }

//     return 0;
// }
