#include "kernels.cuh"
#include "common.cuh"

namespace toyllm {

// SwiGLU: F(z) = W_down (SiLU(z W_up) ⊙ (z W_gate))
//
// Shapes:
//   Z       : [T, D]
//   W_up    : [D, F]   → hidden_up = Z @ W_up     : [T, F]
//   W_gate  : [D, F]   → hidden_gate = Z @ W_gate : [T, F]
//   SiLU(hidden_up) ⊙ hidden_gate                 : [T, F]
//   W_down  : [F, D]   → out = that @ W_down      : [T, D]
//
// We only own the two matmuls + the elementwise SiLU/mul here.
// Matmuls are performed by launch_matmul from outside this kernel wrapper.

__device__ inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void silu_mul_kernel(const float* __restrict__ up,
                                const float* __restrict__ gate,
                                float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = silu(up[i]) * gate[i];
}

// Declared in kernels.cuh; implemented in matmul.cu.
void launch_matmul(const float* A, const float* B, float* C,
                   int M, int K, int N);

void launch_swiglu(const float* Z,
                   const float* W_up, const float* W_gate, const float* W_down,
                   float* hidden_up, float* hidden_gate, float* hidden_mul,
                   float* out,
                   int T, int D, int F) {
    // hidden_up   = Z @ W_up      [T, F]
    launch_matmul(Z, W_up,   hidden_up,   T, D, F);
    // hidden_gate = Z @ W_gate    [T, F]
    launch_matmul(Z, W_gate, hidden_gate, T, D, F);

    // hidden_mul = SiLU(hidden_up) * hidden_gate
    int n = T * F;
    int block = 256;
    int grid  = (n + block - 1) / block;
    silu_mul_kernel<<<grid, block>>>(hidden_up, hidden_gate, hidden_mul, n);
    CUDA_CHECK_LAST();

    // out = hidden_mul @ W_down   [T, D]
    launch_matmul(hidden_mul, W_down, out, T, F, D);
}

}  // namespace toyllm
