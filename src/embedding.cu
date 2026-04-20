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

}  // namespace toyllm
