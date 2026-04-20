#include "kernels.cuh"
#include "common.cuh"

namespace toyllm {

// Numerically-stable row-wise softmax: subtract max, exp, divide by sum.
// One block per row. Two reductions (max then sum).
template <int BLOCK>
__global__ void softmax_rows_kernel(const float* __restrict__ x,
                                    float* __restrict__ out,
                                    int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;

    __shared__ float sbuf[BLOCK];
    const float* xr = x + r * cols;
    float* yr = out + r * cols;

    // 1. Row max
    float local_max = -INFINITY;
    for (int c = threadIdx.x; c < cols; c += BLOCK) {
        float v = xr[c];
        if (v > local_max) local_max = v;
    }
    sbuf[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float a = sbuf[threadIdx.x], b = sbuf[threadIdx.x + stride];
            sbuf[threadIdx.x] = a > b ? a : b;
        }
        __syncthreads();
    }
    float row_max = sbuf[0];

    // 2. Row sum of exp(x - max)
    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += BLOCK) {
        local_sum += expf(xr[c] - row_max);
    }
    sbuf[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sbuf[threadIdx.x] += sbuf[threadIdx.x + stride];
        __syncthreads();
    }
    float row_sum = sbuf[0];
    float inv_sum = 1.0f / row_sum;

    // 3. Write normalized values.
    for (int c = threadIdx.x; c < cols; c += BLOCK) {
        yr[c] = expf(xr[c] - row_max) * inv_sum;
    }
}

void launch_softmax_rows(const float* x, float* out, int rows, int cols) {
    constexpr int BLOCK = 128;
    softmax_rows_kernel<BLOCK><<<rows, BLOCK>>>(x, out, rows, cols);
    CUDA_CHECK_LAST();
}

}  // namespace toyllm
