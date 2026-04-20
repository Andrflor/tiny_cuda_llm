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

// dmul from forward multiply: du = dmul * gate * silu'(up); dgate = dmul * silu(up)
// silu'(x) = sig + x*sig*(1-sig) with sig = 1/(1+exp(-x))
__global__ void silu_mul_backward_kernel(const float* __restrict__ up,
                                         const float* __restrict__ gate,
                                         const float* __restrict__ dmul,
                                         float* __restrict__ du,
                                         float* __restrict__ dgate,
                                         int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = up[i];
    float sig = 1.0f / (1.0f + expf(-x));
    float silu = x * sig;
    float silu_grad = sig + x * sig * (1.0f - sig);
    float g = gate[i];
    float dm = dmul[i];
    du[i]    = dm * g * silu_grad;
    dgate[i] = dm * silu;
}

// Forward decl for local use.
void launch_matmul_backward(const float* dC, const float* A, const float* B,
                            float* dA, float* dB,
                            int M, int K, int N);

// Elementwise addition helper (in-place y += x).
__global__ void add_inplace_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

void launch_swiglu_backward(const float* Z,
                            const float* W_up, const float* W_gate, const float* W_down,
                            const float* hidden_up, const float* hidden_gate,
                            const float* hidden_mul,
                            const float* dout,
                            float* dmul, float* dup, float* dgate,
                            float* dZ, float* dW_up, float* dW_gate, float* dW_down,
                            int T, int D, int F) {
    // 1) dmul [T,F] = dout [T,D] @ W_downᵀ [D,F]
    //    dW_down [F,D] += hidden_mulᵀ @ dout
    launch_matmul_backward(dout, hidden_mul, W_down, dmul, dW_down, T, F, D);

    // 2) dup, dgate elementwise
    int n = T * F;
    int block = 256;
    int grid  = (n + block - 1) / block;
    silu_mul_backward_kernel<<<grid, block>>>(hidden_up, hidden_gate, dmul, dup, dgate, n);
    CUDA_CHECK_LAST();

    // 3) Through Z @ W_up = hidden_up:   dZ_up = dup @ W_upᵀ, dW_up += Zᵀ @ dup
    //    Through Z @ W_gate:             dZ_gt = dgate @ W_gateᵀ, dW_gate += Zᵀ @ dgate
    // We accumulate dZ into the provided dZ buffer (zero it outside first).
    // Use dmul as scratch for the second dZ contribution (shape [T, D] fits since T*F >= T*D when F >= D, which is our case F=2D in v1 but even F>=D works; for safety we allocate a proper buffer. Here we just do matmul_backward then add.
    // Strategy: first pass writes dZ = dup @ W_upᵀ, second pass writes into scratch (reuse dmul[:T*D]) then adds.
    launch_matmul_backward(dup, Z, W_up, dZ, dW_up, T, D, F);
    // scratch for second term. We need [T, D]; dmul has T*F entries, and F >= D in our configs (ffn_dim >= d_model), so reuse its head.
    float* scratch = dmul;  // overwriting dmul which we no longer need
    launch_matmul_backward(dgate, Z, W_gate, scratch, dW_gate, T, D, F);
    int nd = T * D;
    int grid2 = (nd + block - 1) / block;
    add_inplace_kernel<<<grid2, block>>>(scratch, dZ, nd);
    CUDA_CHECK_LAST();
}

}  // namespace toyllm
