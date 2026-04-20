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

// dA[m, k] = sum_n dC[m, n] * B[k, n]  (dC @ Bᵀ)
__global__ void matmul_bwd_dA_kernel(const float* __restrict__ dC,
                                     const float* __restrict__ B,
                                     float* __restrict__ dA,
                                     int M, int K, int N) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || k >= K) return;
    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        acc += dC[m * N + n] * B[k * N + n];
    }
    dA[m * K + k] = acc;
}

// dB[k, n] = sum_m A[m, k] * dC[m, n]  (Aᵀ @ dC)
__global__ void matmul_bwd_dB_kernel(const float* __restrict__ dC,
                                     const float* __restrict__ A,
                                     float* __restrict__ dB,
                                     int M, int K, int N) {
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K || n >= N) return;
    float acc = 0.0f;
    for (int m = 0; m < M; ++m) {
        acc += A[m * K + k] * dC[m * N + n];
    }
    dB[k * N + n] = acc;
}

void launch_matmul_backward(const float* dC, const float* A, const float* B,
                            float* dA, float* dB,
                            int M, int K, int N) {
    if (dA) {
        dim3 block(16, 16);
        dim3 grid((K + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        matmul_bwd_dA_kernel<<<grid, block>>>(dC, B, dA, M, K, N);
        CUDA_CHECK_LAST();
    }
    if (dB) {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (K + block.y - 1) / block.y);
        matmul_bwd_dB_kernel<<<grid, block>>>(dC, A, dB, M, K, N);
        CUDA_CHECK_LAST();
    }
}

}  // namespace toyllm
