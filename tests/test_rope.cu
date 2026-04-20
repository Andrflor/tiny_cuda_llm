#include "kernels.cuh"
#include "tensor.cuh"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace toyllm;

static void cpu_rope(std::vector<float>& X, int T, int H, int dh, float base) {
    const int half = dh / 2;
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < H; ++h) {
            float* row = X.data() + (t * H + h) * dh;
            for (int k = 0; k < half; ++k) {
                float theta = std::pow(base, -2.0f * k / static_cast<float>(dh));
                float phi = static_cast<float>(t) * theta;
                float c = std::cos(phi);
                float s = std::sin(phi);
                float x0 = row[2 * k];
                float x1 = row[2 * k + 1];
                row[2 * k]     = x0 * c - x1 * s;
                row[2 * k + 1] = x0 * s + x1 * c;
            }
        }
    }
}

int main() {
    const int T = 7, H = 3, dh = 8;
    const float base = 10000.0f;
    const int N = T * H * dh;

    std::vector<float> X(N), X_ref(N);
    for (int i = 0; i < N; ++i) {
        X[i] = 0.01f * ((i * 13) % 53) - 0.25f;
    }
    X_ref = X;
    cpu_rope(X_ref, T, H, dh, base);

    DeviceBuffer<float> dX(N);
    dX.copy_from_host(X);
    launch_rope(dX.ptr, T, H, dh, base);
    auto X_gpu = dX.to_host();

    float max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        float e = std::fabs(X_gpu[i] - X_ref[i]);
        if (e > max_err) max_err = e;
    }
    std::printf("test_rope: max_err=%g\n", max_err);
    assert(max_err < 1e-5f);

    // Sanity: rotation preserves the norm of each (pair).
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < H; ++h) {
            for (int k = 0; k < dh / 2; ++k) {
                int i0 = (t * H + h) * dh + 2 * k;
                int i1 = i0 + 1;
                float in_norm  = X[i0]     * X[i0]     + X[i1]     * X[i1];
                float out_norm = X_gpu[i0] * X_gpu[i0] + X_gpu[i1] * X_gpu[i1];
                assert(std::fabs(in_norm - out_norm) < 1e-5f);
            }
        }
    }

    // Sanity: t=0 is identity.
    for (int h = 0; h < H; ++h) {
        for (int d = 0; d < dh; ++d) {
            int i = (0 * H + h) * dh + d;
            assert(std::fabs(X_gpu[i] - X[i]) < 1e-6f);
        }
    }

    std::printf("test_rope: OK\n");
    return 0;
}
