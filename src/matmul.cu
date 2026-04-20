#include "kernels.cuh"
#include "common.cuh"

namespace toyllm {

// Naive row-major GEMM: C[m, n] = sum_k A[m, k] * B[k, n].
// A : [M, K], B : [K, N], C : [M, N]
// No shared-memory tiling yet — this is the bootstrap, correctness > speed.
__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int K, int N) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[m * K + k] * B[k * N + n];
    }
    C[m * N + n] = acc;
}

void launch_matmul(const float* A, const float* B, float* C,
                   int M, int K, int N) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_kernel<<<grid, block>>>(A, B, C, M, K, N);
    CUDA_CHECK_LAST();
}

}  // namespace toyllm
