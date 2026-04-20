#include "kernels.cuh"
#include "tensor.cuh"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace toyllm;

// CPU reference: causal MHA with same layout as launch_causal_mha.
static void cpu_mha(const std::vector<float>& Q,
                    const std::vector<float>& K,
                    const std::vector<float>& V,
                    std::vector<float>& out,
                    int T, int H, int dh) {
    out.assign(T * H * dh, 0.0f);
    float scale = 1.0f / std::sqrt(static_cast<float>(dh));
    std::vector<float> scores(T);
    for (int h = 0; h < H; ++h) {
        for (int i = 0; i < T; ++i) {
            float m = -INFINITY;
            for (int j = 0; j <= i; ++j) {
                float s = 0.0f;
                for (int d = 0; d < dh; ++d)
                    s += Q[(i * H + h) * dh + d] * K[(j * H + h) * dh + d];
                s *= scale;
                scores[j] = s;
                if (s > m) m = s;
            }
            float sum = 0.0f;
            for (int j = 0; j <= i; ++j) {
                scores[j] = std::exp(scores[j] - m);
                sum += scores[j];
            }
            for (int j = 0; j <= i; ++j) scores[j] /= sum;
            for (int d = 0; d < dh; ++d) {
                float acc = 0.0f;
                for (int j = 0; j <= i; ++j)
                    acc += scores[j] * V[(j * H + h) * dh + d];
                out[(i * H + h) * dh + d] = acc;
            }
        }
    }
}

int main() {
    const int T = 6, H = 2, dh = 4;
    const int D = H * dh;

    std::vector<float> Q(T * D), K(T * D), V(T * D);
    for (int i = 0; i < T * D; ++i) {
        Q[i] = 0.01f * ((i * 11) % 37) - 0.1f;
        K[i] = 0.01f * ((i *  7) % 41) - 0.1f;
        V[i] = 0.01f * ((i *  5) % 29) - 0.1f;
    }
    std::vector<float> out_ref;
    cpu_mha(Q, K, V, out_ref, T, H, dh);

    DeviceBuffer<float> dQ(T * D), dK(T * D), dV(T * D);
    DeviceBuffer<float> dS(H * T * T), dO(T * D);
    dQ.copy_from_host(Q); dK.copy_from_host(K); dV.copy_from_host(V);
    launch_causal_mha(dQ.ptr, dK.ptr, dV.ptr, dS.ptr, dO.ptr, T, H, dh);
    auto out_gpu = dO.to_host();

    float max_err = 0.0f;
    for (int i = 0; i < T * D; ++i) {
        float e = std::fabs(out_gpu[i] - out_ref[i]);
        if (e > max_err) max_err = e;
    }
    std::printf("test_attention: max_err=%g\n", max_err);
    assert(max_err < 1e-5f);
    std::printf("test_attention: OK\n");
    return 0;
}
