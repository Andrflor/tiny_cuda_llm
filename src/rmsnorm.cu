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

// One block per row. Computes inv and the scalar dot = sum_j(x_j * g_j * dy_j),
// then writes dx_i = inv*g_i*dy_i - (x_i * inv³ / D) * dot.
// dg is accumulated across rows with atomicAdd.
template <int BLOCK>
__global__ void rmsnorm_bwd_kernel(const float* __restrict__ dy,
                                   const float* __restrict__ x,
                                   const float* __restrict__ g,
                                   float* __restrict__ dx,
                                   float* __restrict__ dg,
                                   int T, int D, float eps) {
    int t = blockIdx.x;
    if (t >= T) return;

    __shared__ float sbuf[BLOCK];
    const float* xr = x + t * D;
    const float* dyr = dy + t * D;
    float* dxr = dx + t * D;

    // Recompute inv.
    float local = 0.0f;
    for (int d = threadIdx.x; d < D; d += BLOCK) {
        float v = xr[d];
        local += v * v;
    }
    sbuf[threadIdx.x] = local;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sbuf[threadIdx.x] += sbuf[threadIdx.x + s];
        __syncthreads();
    }
    float mean_sq = sbuf[0] / static_cast<float>(D);
    float inv = rsqrtf(mean_sq + eps);

    // dot = sum_j(x_j * g_j * dy_j)
    float local_dot = 0.0f;
    for (int d = threadIdx.x; d < D; d += BLOCK) {
        local_dot += xr[d] * g[d] * dyr[d];
    }
    sbuf[threadIdx.x] = local_dot;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sbuf[threadIdx.x] += sbuf[threadIdx.x + s];
        __syncthreads();
    }
    float dot = sbuf[0];
    float inv3_over_D = inv * inv * inv / static_cast<float>(D);

    for (int d = threadIdx.x; d < D; d += BLOCK) {
        dxr[d] = inv * g[d] * dyr[d] - xr[d] * inv3_over_D * dot;
        // dg[d] += x[t,d] * inv * dy[t,d]
        atomicAdd(dg + d, xr[d] * inv * dyr[d]);
    }
}

void launch_rmsnorm_backward(const float* dy, const float* x, const float* g,
                             float* dx, float* dg,
                             int T, int D, float eps) {
    constexpr int BLOCK = 128;
    rmsnorm_bwd_kernel<BLOCK><<<T, BLOCK>>>(dy, x, g, dx, dg, T, D, eps);
    CUDA_CHECK_LAST();
}

}  // namespace toyllm
