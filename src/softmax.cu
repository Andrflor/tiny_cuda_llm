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

// dx[i] = y[i] * (dy[i] - sum_j dy[j] * y[j])
template <int BLOCK>
__global__ void softmax_rows_bwd_kernel(const float* __restrict__ y,
                                        const float* __restrict__ dy,
                                        float* __restrict__ dx,
                                        int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;

    __shared__ float sbuf[BLOCK];
    const float* yr = y + r * cols;
    const float* dyr = dy + r * cols;
    float* dxr = dx + r * cols;

    float local = 0.0f;
    for (int c = threadIdx.x; c < cols; c += BLOCK) local += yr[c] * dyr[c];
    sbuf[threadIdx.x] = local;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sbuf[threadIdx.x] += sbuf[threadIdx.x + s];
        __syncthreads();
    }
    float dot = sbuf[0];

    for (int c = threadIdx.x; c < cols; c += BLOCK) {
        dxr[c] = yr[c] * (dyr[c] - dot);
    }
}

void launch_softmax_rows_backward(const float* y, const float* dy, float* dx,
                                  int rows, int cols) {
    constexpr int BLOCK = 128;
    softmax_rows_bwd_kernel<BLOCK><<<rows, BLOCK>>>(y, dy, dx, rows, cols);
    CUDA_CHECK_LAST();
}

}  // namespace toyllm
