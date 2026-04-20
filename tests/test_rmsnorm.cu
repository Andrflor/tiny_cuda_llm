#include "kernels.cuh"
#include "tensor.cuh"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace toyllm;

static void cpu_rmsnorm(const std::vector<float>& x, const std::vector<float>& g,
                        std::vector<float>& y, int T, int D, float eps) {
    y.assign(T * D, 0.0f);
    for (int t = 0; t < T; ++t) {
        float sq = 0.0f;
        for (int d = 0; d < D; ++d) sq += x[t * D + d] * x[t * D + d];
        float inv = 1.0f / std::sqrt(sq / D + eps);
        for (int d = 0; d < D; ++d) y[t * D + d] = x[t * D + d] * inv * g[d];
    }
}

static void test_forward() {
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

    std::vector<float> y_ref;
    cpu_rmsnorm(x, g, y_ref, T, D, eps);
    float max_err = 0.0f;
    for (int i = 0; i < T * D; ++i)
        max_err = std::fmax(max_err, std::fabs(y[i] - y_ref[i]));
    std::printf("test_rmsnorm forward: max_err=%g\n", max_err);
    assert(max_err < 1e-5f);
}

static void test_backward() {
    const int T = 3, D = 16;
    const float eps = 1e-5f;
    std::vector<float> x(T * D), g(D), dy(T * D);
    for (int i = 0; i < T * D; ++i) x[i] = 0.1f * (i + 1) - 1.0f;
    for (int d = 0; d < D; ++d) g[d] = 0.8f + 0.03f * d;
    for (int i = 0; i < T * D; ++i) dy[i] = 0.02f * ((i * 7) % 13) - 0.1f;

    // Finite-diff reference for dx and dg against L = sum(y * dy).
    auto loss_of = [&](const std::vector<float>& x_, const std::vector<float>& g_) {
        std::vector<float> y_;
        cpu_rmsnorm(x_, g_, y_, T, D, eps);
        float s = 0.0f;
        for (int i = 0; i < T * D; ++i) s += y_[i] * dy[i];
        return s;
    };

    const float h = 1e-3f;
    std::vector<float> dx_ref(T * D, 0.0f), dg_ref(D, 0.0f);
    for (int i = 0; i < T * D; ++i) {
        auto xp = x, xm = x;
        xp[i] += h; xm[i] -= h;
        dx_ref[i] = (loss_of(xp, g) - loss_of(xm, g)) / (2 * h);
    }
    for (int d = 0; d < D; ++d) {
        auto gp = g, gm = g;
        gp[d] += h; gm[d] -= h;
        dg_ref[d] = (loss_of(x, gp) - loss_of(x, gm)) / (2 * h);
    }

    DeviceBuffer<float> dxx(T * D), dgg(D), ddy(T * D), ddx(T * D), ddg(D);
    dxx.copy_from_host(x);
    dgg.copy_from_host(g);
    ddy.copy_from_host(dy);
    ddg.zero();
    launch_rmsnorm_backward(ddy.ptr, dxx.ptr, dgg.ptr, ddx.ptr, ddg.ptr, T, D, eps);
    auto dx_h = ddx.to_host();
    auto dg_h = ddg.to_host();

    float err_x = 0.0f, err_g = 0.0f;
    for (int i = 0; i < T * D; ++i) err_x = std::fmax(err_x, std::fabs(dx_h[i] - dx_ref[i]));
    for (int d = 0; d < D; ++d) err_g = std::fmax(err_g, std::fabs(dg_h[d] - dg_ref[d]));
    std::printf("test_rmsnorm backward: err_dx=%g err_dg=%g\n", err_x, err_g);
    assert(err_x < 1e-2f);
    assert(err_g < 1e-2f);
}

int main() {
    test_forward();
    test_backward();
    std::printf("test_rmsnorm: OK\n");
    return 0;
}
