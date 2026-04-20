#include "kernels.cuh"
#include "common.cuh"

#include <cmath>

namespace toyllm {

// Decoupled-weight-decay Adam (AdamW):
//   param -= lr * wd * param
//   m = b1*m + (1-b1)*g
//   v = b2*v + (1-b2)*g*g
//   m_hat = m / (1 - b1^step)
//   v_hat = v / (1 - b2^step)
//   param -= lr * m_hat / (sqrt(v_hat) + eps)
__global__ void adamw_kernel(float* __restrict__ param,
                             const float* __restrict__ grad,
                             float* __restrict__ m,
                             float* __restrict__ v,
                             int n, float lr, float b1, float b2,
                             float eps, float wd,
                             float bc1, float bc2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float p = param[i];
    float g = grad[i];
    // Decoupled WD.
    p -= lr * wd * p;
    float mi = b1 * m[i] + (1.0f - b1) * g;
    float vi = b2 * v[i] + (1.0f - b2) * g * g;
    m[i] = mi;
    v[i] = vi;
    float mh = mi / bc1;
    float vh = vi / bc2;
    p -= lr * mh / (sqrtf(vh) + eps);
    param[i] = p;
}

void launch_adamw_step(float* param, const float* grad,
                       float* m, float* v,
                       int n, int step,
                       float lr, float beta1, float beta2,
                       float eps, float weight_decay) {
    float bc1 = 1.0f - powf(beta1, static_cast<float>(step));
    float bc2 = 1.0f - powf(beta2, static_cast<float>(step));
    int block = 256;
    int grid = (n + block - 1) / block;
    adamw_kernel<<<grid, block>>>(param, grad, m, v, n,
                                  lr, beta1, beta2, eps, weight_decay,
                                  bc1, bc2);
    CUDA_CHECK_LAST();
}

// --- Small helpers ---
__global__ void fill_kernel(float* x, float value, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = value;
}
void launch_fill(float* x, float value, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    fill_kernel<<<grid, block>>>(x, value, n);
    CUDA_CHECK_LAST();
}

__global__ void axpy_kernel(float alpha, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += alpha * x[i];
}
void launch_axpy(float alpha, const float* x, float* y, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    axpy_kernel<<<grid, block>>>(alpha, x, y, n);
    CUDA_CHECK_LAST();
}

}  // namespace toyllm
