#include "kernels.cuh"
#include "common.cuh"

#include <cmath>

namespace toyllm {

// RoPE (Rotary Position Embedding), paired variant.
//
// Input layout: X ∈ R^[T, H, dh] contiguous, same as Q/K in attention.
// For each position t and each head h, we interpret the dh-vector as dh/2 pairs
// (x_{2k}, x_{2k+1}) and rotate each pair by angle φ_{t,k} = t · θ_k, with
//   θ_k = base^(-2k / dh),   k = 0, ..., dh/2 - 1.
//
//   x'_{2k}   = x_{2k}   · cos(φ) − x_{2k+1} · sin(φ)
//   x'_{2k+1} = x_{2k}   · sin(φ) + x_{2k+1} · cos(φ)
//
// In-place. Requires dh even.
__global__ void rope_kernel(float* __restrict__ X,
                            int T, int H, int dh, float inv_dh, float log_base) {
    int h = blockIdx.z;
    int t = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;   // pair index, 0..dh/2-1
    int half = dh >> 1;
    if (h >= H || t >= T || k >= half) return;

    // θ_k = exp(-2k/dh · log(base))
    float theta = expf(-2.0f * static_cast<float>(k) * inv_dh * log_base);
    float phi = static_cast<float>(t) * theta;
    float c = cosf(phi);
    float s = sinf(phi);

    float* row = X + (t * H + h) * dh;
    float x0 = row[2 * k];
    float x1 = row[2 * k + 1];
    row[2 * k]     = x0 * c - x1 * s;
    row[2 * k + 1] = x0 * s + x1 * c;
}

void launch_rope(float* X, int T, int H, int dh, float base) {
    if ((dh & 1) != 0) {
        std::fprintf(stderr, "launch_rope: dh must be even, got %d\n", dh);
        std::exit(1);
    }
    const int half = dh >> 1;
    const float inv_dh = 1.0f / static_cast<float>(dh);
    const float log_base = logf(base);

    dim3 block(32);
    dim3 grid((half + block.x - 1) / block.x, T, H);
    rope_kernel<<<grid, block>>>(X, T, H, dh, inv_dh, log_base);
    CUDA_CHECK_LAST();
}

}  // namespace toyllm
