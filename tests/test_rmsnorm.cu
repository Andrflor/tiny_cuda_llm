#include "kernels.cuh"
#include "tensor.cuh"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace toyllm;

int main() {
    const int T = 3, D = 8;
    std::vector<float> x(T * D), g(D);
    for (int i = 0; i < T * D; ++i) x[i] = 0.1f * (i + 1) - 1.2f;
    for (int d = 0; d < D; ++d) g[d] = 1.0f + 0.05f * d;

    DeviceBuffer<float> dx(T * D), dg(D), dout(T * D);
    dx.copy_from_host(x);
    dg.copy_from_host(g);

    const float eps = 1e-5f;
    launch_rmsnorm(dx.ptr, dg.ptr, dout.ptr, T, D, eps);
    auto y = dout.to_host();

    // Reference on CPU.
    float max_err = 0.0f;
    for (int t = 0; t < T; ++t) {
        float sq = 0.0f;
        for (int d = 0; d < D; ++d) sq += x[t * D + d] * x[t * D + d];
        float inv = 1.0f / std::sqrt(sq / D + eps);
        for (int d = 0; d < D; ++d) {
            float ref = x[t * D + d] * inv * g[d];
            float e = std::fabs(ref - y[t * D + d]);
            if (e > max_err) max_err = e;
        }
    }
    std::printf("test_rmsnorm: max_err=%g\n", max_err);
    assert(max_err < 1e-5f);
    std::printf("test_rmsnorm: OK\n");
    return 0;
}
