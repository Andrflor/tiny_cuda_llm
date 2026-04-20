#include "kernels.cuh"
#include "common.cuh"

namespace toyllm {

// Layout: Q, K, V : [T, H, d_h] stored as [T, H*d_h].
// Compute S[h, i, j] = Q[i, h, :] . K[j, h, :] / sqrt(d_h), with causal mask
// (j > i => -inf).
__global__ void attn_scores_kernel(const float* __restrict__ Q,
                                   const float* __restrict__ K,
                                   float* __restrict__ S,
                                   int T, int H, int dh, float scale) {
    int h = blockIdx.z;
    int i = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H || i >= T || j >= T) return;

    if (j > i) {
        S[((h * T) + i) * T + j] = -INFINITY;
        return;
    }
    const float* qrow = Q + (i * H + h) * dh;
    const float* krow = K + (j * H + h) * dh;
    float acc = 0.0f;
    for (int d = 0; d < dh; ++d) acc += qrow[d] * krow[d];
    S[((h * T) + i) * T + j] = acc * scale;
}

// Apply softmax row-wise to S[h, i, :]. One block per (h, i).
template <int BLOCK>
__global__ void attn_softmax_kernel(float* __restrict__ S, int T, int H) {
    int h = blockIdx.y;
    int i = blockIdx.x;
    float* row = S + ((h * T) + i) * T;

    __shared__ float sbuf[BLOCK];
    // max
    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < T; j += BLOCK) {
        float v = row[j];
        if (v > local_max) local_max = v;
    }
    sbuf[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float a = sbuf[threadIdx.x], b = sbuf[threadIdx.x + s];
            sbuf[threadIdx.x] = a > b ? a : b;
        }
        __syncthreads();
    }
    float row_max = sbuf[0];
    // For a fully-masked row (shouldn't happen with causal + T≥1 since j=i is
    // always allowed), row_max would be -inf; guard anyway.
    if (!isfinite(row_max)) row_max = 0.0f;

    // sum
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < T; j += BLOCK) {
        float v = expf(row[j] - row_max);
        row[j] = v;  // stash e^(x-max)
        local_sum += v;
    }
    sbuf[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sbuf[threadIdx.x] += sbuf[threadIdx.x + s];
        __syncthreads();
    }
    float inv = 1.0f / sbuf[0];
    for (int j = threadIdx.x; j < T; j += BLOCK) row[j] *= inv;
}

// out[i, h, d] = sum_j A[h, i, j] * V[j, h, d]
__global__ void attn_apply_kernel(const float* __restrict__ A,
                                  const float* __restrict__ V,
                                  float* __restrict__ out,
                                  int T, int H, int dh) {
    int h = blockIdx.z;
    int i = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H || i >= T || d >= dh) return;
    const float* arow = A + ((h * T) + i) * T;
    float acc = 0.0f;
    for (int j = 0; j < T; ++j) {
        acc += arow[j] * V[(j * H + h) * dh + d];
    }
    out[(i * H + h) * dh + d] = acc;
}

void launch_causal_mha(const float* Q, const float* K, const float* V,
                       float* scores, float* out,
                       int T, int H, int dh) {
    float scale = 1.0f / sqrtf(static_cast<float>(dh));
    {
        dim3 block(32);
        dim3 grid((T + block.x - 1) / block.x, T, H);
        attn_scores_kernel<<<grid, block>>>(Q, K, scores, T, H, dh, scale);
        CUDA_CHECK_LAST();
    }
    {
        constexpr int BLOCK = 128;
        dim3 grid(T, H);
        attn_softmax_kernel<BLOCK><<<grid, BLOCK>>>(scores, T, H);
        CUDA_CHECK_LAST();
    }
    {
        dim3 block(32);
        dim3 grid((dh + block.x - 1) / block.x, T, H);
        attn_apply_kernel<<<grid, block>>>(scores, V, out, T, H, dh);
        CUDA_CHECK_LAST();
    }
}

}  // namespace toyllm
