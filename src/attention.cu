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

// ---------------- Backward ----------------

// dV[j, h, d] = sum_i A[h, i, j] * dout[i, h, d]
__global__ void attn_bwd_dV_kernel(const float* __restrict__ A,
                                   const float* __restrict__ dout,
                                   float* __restrict__ dV,
                                   int T, int H, int dh) {
    int h = blockIdx.z;
    int j = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H || j >= T || d >= dh) return;
    float acc = 0.0f;
    // i goes from j..T-1 because A[h,i,j] is zero for i < j (causal mask),
    // but summing full range is still correct. Keep simple.
    for (int i = 0; i < T; ++i) {
        acc += A[((h * T) + i) * T + j] * dout[(i * H + h) * dh + d];
    }
    dV[(j * H + h) * dh + d] += acc;
}

// dA[h, i, j] = sum_d V[j, h, d] * dout[i, h, d]. Stored flat [H, T, T].
__global__ void attn_bwd_dA_kernel(const float* __restrict__ V,
                                   const float* __restrict__ dout,
                                   float* __restrict__ dA,
                                   int T, int H, int dh) {
    int h = blockIdx.z;
    int i = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H || i >= T || j >= T) return;
    float acc = 0.0f;
    for (int d = 0; d < dh; ++d) {
        acc += V[(j * H + h) * dh + d] * dout[(i * H + h) * dh + d];
    }
    dA[((h * T) + i) * T + j] = acc;
}

// dS[h, i, j] = A[h, i, j] * (dA[h, i, j] - sum_k A[h, i, k] * dA[h, i, k])
// One block per (h, i). Writes into dS in place of dA (so pass dA as inout).
template <int BLOCK>
__global__ void attn_bwd_dS_kernel(const float* __restrict__ A,
                                   float* __restrict__ dAdS,
                                   int T, int H) {
    int h = blockIdx.y;
    int i = blockIdx.x;
    const float* arow = A + ((h * T) + i) * T;
    float* row = dAdS + ((h * T) + i) * T;

    __shared__ float sbuf[BLOCK];
    float local = 0.0f;
    for (int j = threadIdx.x; j < T; j += BLOCK) local += arow[j] * row[j];
    sbuf[threadIdx.x] = local;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sbuf[threadIdx.x] += sbuf[threadIdx.x + s];
        __syncthreads();
    }
    float dot = sbuf[0];
    for (int j = threadIdx.x; j < T; j += BLOCK) {
        row[j] = arow[j] * (row[j] - dot);
    }
}

// dQ[i, h, d] += scale * sum_j dS[h, i, j] * K[j, h, d]
__global__ void attn_bwd_dQ_kernel(const float* __restrict__ dS,
                                   const float* __restrict__ K,
                                   float* __restrict__ dQ,
                                   int T, int H, int dh, float scale) {
    int h = blockIdx.z;
    int i = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H || i >= T || d >= dh) return;
    float acc = 0.0f;
    for (int j = 0; j <= i; ++j) {  // j > i => dS is ~0 anyway via softmax
        acc += dS[((h * T) + i) * T + j] * K[(j * H + h) * dh + d];
    }
    dQ[(i * H + h) * dh + d] += scale * acc;
}

// dK[j, h, d] += scale * sum_i dS[h, i, j] * Q[i, h, d]
__global__ void attn_bwd_dK_kernel(const float* __restrict__ dS,
                                   const float* __restrict__ Q,
                                   float* __restrict__ dK,
                                   int T, int H, int dh, float scale) {
    int h = blockIdx.z;
    int j = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H || j >= T || d >= dh) return;
    float acc = 0.0f;
    for (int i = j; i < T; ++i) {
        acc += dS[((h * T) + i) * T + j] * Q[(i * H + h) * dh + d];
    }
    dK[(j * H + h) * dh + d] += scale * acc;
}

void launch_causal_mha_backward(const float* Q, const float* K, const float* V,
                                const float* A, const float* dout,
                                float* dS,
                                float* dQ, float* dK, float* dV,
                                int T, int H, int dh) {
    const float scale = 1.0f / sqrtf(static_cast<float>(dh));

    // 1) dV += Aᵀ @ dout
    {
        dim3 block(32);
        dim3 grid((dh + block.x - 1) / block.x, T, H);
        attn_bwd_dV_kernel<<<grid, block>>>(A, dout, dV, T, H, dh);
        CUDA_CHECK_LAST();
    }
    // 2) dA = dout @ Vᵀ (reshape per head)
    {
        dim3 block(32);
        dim3 grid((T + block.x - 1) / block.x, T, H);
        attn_bwd_dA_kernel<<<grid, block>>>(V, dout, dS, T, H, dh);
        CUDA_CHECK_LAST();
    }
    // 3) dS = softmax_bwd(A, dA)  (in-place on dS buffer)
    {
        constexpr int BLOCK = 128;
        dim3 grid(T, H);
        attn_bwd_dS_kernel<BLOCK><<<grid, BLOCK>>>(A, dS, T, H);
        CUDA_CHECK_LAST();
    }
    // 4) dQ += scale * dS @ K (per head)
    {
        dim3 block(32);
        dim3 grid((dh + block.x - 1) / block.x, T, H);
        attn_bwd_dQ_kernel<<<grid, block>>>(dS, K, dQ, T, H, dh, scale);
        CUDA_CHECK_LAST();
    }
    // 5) dK += scale * dSᵀ @ Q
    {
        dim3 block(32);
        dim3 grid((dh + block.x - 1) / block.x, T, H);
        attn_bwd_dK_kernel<<<grid, block>>>(dS, Q, dK, T, H, dh, scale);
        CUDA_CHECK_LAST();
    }
}

}  // namespace toyllm
