#include "kernels.cuh"
#include "common.cuh"

namespace toyllm {

// out[t, d] = E[ids[t], d]
__global__ void embedding_lookup_kernel(const int* __restrict__ ids,
                                        const float* __restrict__ E,
                                        float* __restrict__ out,
                                        int T, int V, int D) {
    int t = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T || d >= D) return;
    int id = ids[t];
    // Clamp for safety; ids must be in [0, V).
    if (id < 0) id = 0;
    if (id >= V) id = V - 1;
    out[t * D + d] = E[id * D + d];
}

void launch_embedding_lookup(const int* ids, const float* E, float* out,
                             int T, int V, int D) {
    const int block = 128;
    dim3 grid((D + block - 1) / block, T);
    embedding_lookup_kernel<<<grid, block>>>(ids, E, out, T, V, D);
    CUDA_CHECK_LAST();
}

// logits[t, v] = sum_d X[t, d] * E[v, d]
// Equivalent to X @ Eᵀ with E stored row-major as [V, D].
__global__ void logits_tied_kernel(const float* __restrict__ X,
                                   const float* __restrict__ E,
                                   float* __restrict__ logits,
                                   int T, int D, int V) {
    int t = blockIdx.y;
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T || v >= V) return;
    const float* x_row = X + t * D;
    const float* e_row = E + v * D;
    float acc = 0.0f;
    for (int d = 0; d < D; ++d) acc += x_row[d] * e_row[d];
    logits[t * V + v] = acc;
}

void launch_logits_tied(const float* X, const float* E, float* logits,
                        int T, int D, int V) {
    const int block = 128;
    dim3 grid((V + block - 1) / block, T);
    logits_tied_kernel<<<grid, block>>>(X, E, logits, T, D, V);
    CUDA_CHECK_LAST();
}

// dE[ids[t], d] += dout[t, d]  via atomicAdd (ids can repeat).
__global__ void embedding_bwd_kernel(const int* __restrict__ ids,
                                     const float* __restrict__ dout,
                                     float* __restrict__ dE,
                                     int T, int V, int D) {
    int t = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T || d >= D) return;
    int id = ids[t];
    if (id < 0 || id >= V) return;
    atomicAdd(dE + id * D + d, dout[t * D + d]);
}

void launch_embedding_backward(const int* ids, const float* dout, float* dE,
                               int T, int V, int D) {
    const int block = 128;
    dim3 grid((D + block - 1) / block, T);
    embedding_bwd_kernel<<<grid, block>>>(ids, dout, dE, T, V, D);
    CUDA_CHECK_LAST();
}

// dX[t, d] = sum_v dlogits[t, v] * E[v, d]
__global__ void logits_tied_bwd_dX_kernel(const float* __restrict__ dlogits,
                                          const float* __restrict__ E,
                                          float* __restrict__ dX,
                                          int T, int D, int V) {
    int t = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T || d >= D) return;
    const float* dlr = dlogits + t * V;
    float acc = 0.0f;
    for (int v = 0; v < V; ++v) acc += dlr[v] * E[v * D + d];
    dX[t * D + d] = acc;
}

// dE[v, d] += sum_t dlogits[t, v] * X[t, d]
__global__ void logits_tied_bwd_dE_kernel(const float* __restrict__ dlogits,
                                          const float* __restrict__ X,
                                          float* __restrict__ dE,
                                          int T, int D, int V) {
    int v = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V || d >= D) return;
    float acc = 0.0f;
    for (int t = 0; t < T; ++t) acc += dlogits[t * V + v] * X[t * D + d];
    dE[v * D + d] += acc;
}

void launch_logits_tied_backward(const float* dlogits, const float* X, const float* E,
                                 float* dX, float* dE,
                                 int T, int D, int V) {
    const int block = 128;
    {
        dim3 grid((D + block - 1) / block, T);
        logits_tied_bwd_dX_kernel<<<grid, block>>>(dlogits, E, dX, T, D, V);
        CUDA_CHECK_LAST();
    }
    {
        dim3 grid((D + block - 1) / block, V);
        logits_tied_bwd_dE_kernel<<<grid, block>>>(dlogits, X, dE, T, D, V);
        CUDA_CHECK_LAST();
    }
}

}  // namespace toyllm
