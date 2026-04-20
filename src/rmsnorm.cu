#include "kernels.cuh"
#include "common.cuh"

namespace toyllm {

// One block per row. Each block computes mean(x²) via shared-memory reduction,
// then normalizes and rescales with the per-feature gain g.
template <int BLOCK>
__global__ void rmsnorm_kernel(const float* __restrict__ x,
                               const float* __restrict__ g,
                               float* __restrict__ out,
                               int T, int D, float eps) {
    int t = blockIdx.x;
    if (t >= T) return;

    __shared__ float sbuf[BLOCK];
    const float* xr = x + t * D;
    float* yr = out + t * D;

    // Compute sum of squares.
    float local = 0.0f;
    for (int d = threadIdx.x; d < D; d += BLOCK) {
        float v = xr[d];
        local += v * v;
    }
    sbuf[threadIdx.x] = local;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sbuf[threadIdx.x] += sbuf[threadIdx.x + stride];
        __syncthreads();
    }
    float mean_sq = sbuf[0] / static_cast<float>(D);
    float inv = rsqrtf(mean_sq + eps);

    for (int d = threadIdx.x; d < D; d += BLOCK) {
        yr[d] = xr[d] * inv * g[d];
    }
}

void launch_rmsnorm(const float* x, const float* g, float* out,
                    int T, int D, float eps) {
    constexpr int BLOCK = 128;
    rmsnorm_kernel<BLOCK><<<T, BLOCK>>>(x, g, out, T, D, eps);
    CUDA_CHECK_LAST();
}

}  // namespace toyllm
