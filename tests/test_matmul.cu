#include "kernels.cuh"
#include "tensor.cuh"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace toyllm;

static void cpu_matmul(const std::vector<float>& A, const std::vector<float>& B,
                       std::vector<float>& C, int M, int K, int N) {
    C.assign(M * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) {
            float a = A[m * K + k];
            for (int n = 0; n < N; ++n)
                C[m * N + n] += a * B[k * N + n];
        }
}

int main() {
    const int M = 17, K = 9, N = 13;
    std::vector<float> A(M * K), B(K * N);
    for (int i = 0; i < M * K; ++i) A[i] = 0.01f * (i % 23) - 0.1f;
    for (int i = 0; i < K * N; ++i) B[i] = 0.01f * ((i * 7) % 31) - 0.15f;

    std::vector<float> C_ref;
    cpu_matmul(A, B, C_ref, M, K, N);

    DeviceBuffer<float> dA(M * K), dB(K * N), dC(M * N);
    dA.copy_from_host(A);
    dB.copy_from_host(B);
    launch_matmul(dA.ptr, dB.ptr, dC.ptr, M, K, N);
    auto C_gpu = dC.to_host();

    float max_err = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float e = std::fabs(C_gpu[i] - C_ref[i]);
        if (e > max_err) max_err = e;
    }
    std::printf("test_matmul: max_err=%g\n", max_err);
    assert(max_err < 1e-4f);
    std::printf("test_matmul: OK\n");
    return 0;
}
