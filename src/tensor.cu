#include "tensor.cuh"
#include "kernels.cuh"

namespace toyllm {

__global__ void add_kernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

void launch_add(const float* a, const float* b, float* out, int n) {
    const int block = 256;
    const int grid  = (n + block - 1) / block;
    add_kernel<<<grid, block>>>(a, b, out, n);
    CUDA_CHECK_LAST();
}

}  // namespace toyllm
