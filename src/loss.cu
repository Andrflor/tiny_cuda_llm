#include "kernels.cuh"
#include "common.cuh"

#include <cmath>

namespace toyllm {

// Count valid (>=0) entries of targets into *out (written with atomicAdd).
__global__ void ce_count_valid_kernel(const int* __restrict__ t,
                                      float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (t[i] >= 0) atomicAdd(out, 1.0f);
}

// One block per row. Computes softmax(logits[t]), writes
//   dlogits[t, v] = (p_v - onehot(target)) / n_valid
// and accumulates -log(p[target]) / n_valid into *loss_out.
template <int BLOCK>
__global__ void cross_entropy_kernel(const float* __restrict__ logits,
                                     const int* __restrict__ targets,
                                     float* __restrict__ loss_out,
                                     float* __restrict__ dlogits,
                                     int T, int V, float inv_n_valid) {
    int t = blockIdx.x;
    if (t >= T) return;

    int target = targets[t];
    const float* xr = logits + t * V;
    float* dr = dlogits + t * V;

    if (target < 0) {
        for (int v = threadIdx.x; v < V; v += BLOCK) dr[v] = 0.0f;
        return;
    }

    __shared__ float sbuf[BLOCK];

    float local_max = -INFINITY;
    for (int v = threadIdx.x; v < V; v += BLOCK) {
        float x = xr[v];
        if (x > local_max) local_max = x;
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

    float local_sum = 0.0f;
    for (int v = threadIdx.x; v < V; v += BLOCK) {
        local_sum += expf(xr[v] - row_max);
    }
    sbuf[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sbuf[threadIdx.x] += sbuf[threadIdx.x + s];
        __syncthreads();
    }
    float row_sum = sbuf[0];
    float log_sum = logf(row_sum) + row_max;

    for (int v = threadIdx.x; v < V; v += BLOCK) {
        float p = expf(xr[v] - row_max) / row_sum;
        float g = p - (v == target ? 1.0f : 0.0f);
        dr[v] = g * inv_n_valid;
    }

    if (threadIdx.x == 0) {
        float nll = log_sum - xr[target];
        atomicAdd(loss_out, nll * inv_n_valid);
    }
}

void launch_cross_entropy(const float* logits, const int* targets,
                          float* loss_out, float* dlogits,
                          int T, int V) {
    // Device scratch for n_valid — one float.
    static float* d_nvalid = nullptr;
    if (!d_nvalid) CUDA_CHECK(cudaMalloc(&d_nvalid, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_nvalid, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(loss_out, 0, sizeof(float)));

    ce_count_valid_kernel<<<(T + 255) / 256, 256>>>(targets, d_nvalid, T);
    CUDA_CHECK_LAST();

    float n_valid_host = 0.0f;
    CUDA_CHECK(cudaMemcpy(&n_valid_host, d_nvalid, sizeof(float), cudaMemcpyDeviceToHost));
    if (n_valid_host <= 0.0f) {
        CUDA_CHECK(cudaMemset(dlogits, 0, sizeof(float) * static_cast<size_t>(T) * V));
        return;
    }
    float inv_n = 1.0f / n_valid_host;

    constexpr int BLOCK = 128;
    cross_entropy_kernel<BLOCK><<<T, BLOCK>>>(logits, targets, loss_out, dlogits,
                                              T, V, inv_n);
    CUDA_CHECK_LAST();
}

}  // namespace toyllm
