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

static void test_forward() {
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
    std::printf("test_matmul forward: max_err=%g\n", max_err);
    assert(max_err < 1e-4f);
}

static void test_backward() {
    const int M = 11, K = 7, N = 5;
    std::vector<float> A(M * K), B(K * N), dC(M * N);
    for (int i = 0; i < M * K; ++i) A[i] = 0.01f * (i % 19) - 0.1f;
    for (int i = 0; i < K * N; ++i) B[i] = 0.02f * ((i * 3) % 17) - 0.12f;
    for (int i = 0; i < M * N; ++i) dC[i] = 0.05f * ((i * 11) % 13) - 0.2f;

    // Reference: dA = dC @ Bᵀ, dB = Aᵀ @ dC
    std::vector<float> dA_ref(M * K, 0.0f), dB_ref(K * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) {
            float acc = 0.0f;
            for (int n = 0; n < N; ++n) acc += dC[m * N + n] * B[k * N + n];
            dA_ref[m * K + k] = acc;
        }
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int m = 0; m < M; ++m) acc += A[m * K + k] * dC[m * N + n];
            dB_ref[k * N + n] = acc;
        }

    DeviceBuffer<float> dA_dev(M * K), dB_dev(K * N), dC_dev(M * N);
    DeviceBuffer<float> gA(M * K), gB(K * N);
    dA_dev.copy_from_host(A);
    dB_dev.copy_from_host(B);
    dC_dev.copy_from_host(dC);
    launch_matmul_backward(dC_dev.ptr, dA_dev.ptr, dB_dev.ptr,
                           gA.ptr, gB.ptr, M, K, N);
    auto gA_h = gA.to_host();
    auto gB_h = gB.to_host();

    float err_A = 0.0f, err_B = 0.0f;
    for (int i = 0; i < M * K; ++i) err_A = std::fmax(err_A, std::fabs(gA_h[i] - dA_ref[i]));
    for (int i = 0; i < K * N; ++i) err_B = std::fmax(err_B, std::fabs(gB_h[i] - dB_ref[i]));
    std::printf("test_matmul backward: err_dA=%g err_dB=%g\n", err_A, err_B);
    assert(err_A < 1e-4f);
    assert(err_B < 1e-4f);
}

int main() {
    test_forward();
    test_backward();
    std::printf("test_matmul: OK\n");
    return 0;
}
