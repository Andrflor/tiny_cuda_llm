#include "model.cuh"
#include "kernels.cuh"
#include "common.h"
#include "common.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace toyllm {

static void fill_uniform(std::vector<float>& v, Xorshift64& rng, float scale) {
    for (auto& x : v) x = rng.next_uniform() * scale;
}

void init_random_weights(ModelWeights& w, const ModelConfig& cfg) {
    Xorshift64 rng(cfg.seed);

    const int V = cfg.vocab_size;
    const int D = cfg.d_model;
    const int F = cfg.ffn_dim;

    // Scale à la "scaled init" : 1/sqrt(fan_in).
    auto scale_for = [](int fan_in) {
        return 1.0f / std::sqrt(static_cast<float>(fan_in));
    };

    // Embeddings [V, D]
    {
        std::vector<float> E(V * D);
        fill_uniform(E, rng, scale_for(D));
        w.E.allocate(V * D);
        w.E.copy_from_host(E);
    }
    // g_final [D]
    {
        std::vector<float> g(D, 1.0f);
        w.g_final.allocate(D);
        w.g_final.copy_from_host(g);
    }

    w.layers.resize(cfg.n_layers);
    for (int l = 0; l < cfg.n_layers; ++l) {
        auto& L = w.layers[l];
        // RMSNorm gains = 1
        {
            std::vector<float> g(D, 1.0f);
            L.g_attn.allocate(D); L.g_attn.copy_from_host(g);
            L.g_ffn.allocate(D);  L.g_ffn.copy_from_host(g);
        }
        auto init_mat = [&](DeviceBuffer<float>& buf, int rows, int cols) {
            std::vector<float> m(rows * cols);
            fill_uniform(m, rng, scale_for(rows));
            buf.allocate(rows * cols);
            buf.copy_from_host(m);
        };
        init_mat(L.W_Q, D, D);
        init_mat(L.W_K, D, D);
        init_mat(L.W_V, D, D);
        init_mat(L.W_O, D, D);
        init_mat(L.W_up,   D, F);
        init_mat(L.W_gate, D, F);
        init_mat(L.W_down, F, D);
    }
}

void alloc_workspace(Workspace& ws, const ModelConfig& cfg) {
    const int T = cfg.seq_len;
    const int D = cfg.d_model;
    const int H = cfg.n_heads;
    const int F = cfg.ffn_dim;
    const int V = cfg.vocab_size;

    ws.X.allocate(T * D);
    ws.U.allocate(T * D);
    ws.Q.allocate(T * D);
    ws.K.allocate(T * D);
    ws.V.allocate(T * D);
    ws.scores.allocate(H * T * T);
    ws.attn_out.allocate(T * D);
    ws.O.allocate(T * D);
    ws.R.allocate(T * D);
    ws.Z.allocate(T * D);
    ws.ffn_up.allocate(T * F);
    ws.ffn_gate.allocate(T * F);
    ws.ffn_mul.allocate(T * F);
    ws.ffn_out.allocate(T * D);
    ws.Y.allocate(T * D);
    ws.logits.allocate(T * V);
    ws.ids.allocate(T);
}

void model_forward(const ModelWeights& w, const ModelConfig& cfg,
                   Workspace& ws, const std::vector<int>& ids,
                   std::vector<float>& last_logits_host) {
    const int T_full = cfg.seq_len;
    const int D = cfg.d_model;
    const int H = cfg.n_heads;
    const int dh = cfg.head_dim;
    const int F = cfg.ffn_dim;
    const int V = cfg.vocab_size;
    if (static_cast<int>(ids.size()) > T_full) die("prompt longer than seq_len");
    const int T = static_cast<int>(ids.size());

    // Pad ids to T_full with id 0. We still run the forward on T_full positions;
    // causal mask ensures position t only sees 0..t, but for greedy generation
    // we read the logit at the *last real* position (t = T - 1).
    std::vector<int> padded(T_full, 0);
    for (int t = 0; t < T; ++t) padded[t] = ids[t];
    ws.ids.copy_from_host(padded);

    // X = embed(ids)
    launch_embedding_lookup(ws.ids.ptr, w.E.ptr, ws.X.ptr, T_full, V, D);

    for (int l = 0; l < cfg.n_layers; ++l) {
        const auto& L = w.layers[l];

        // U = RMSNorm(X)
        launch_rmsnorm(ws.X.ptr, L.g_attn.ptr, ws.U.ptr, T_full, D, cfg.rms_eps);
        // Q, K, V
        launch_matmul(ws.U.ptr, L.W_Q.ptr, ws.Q.ptr, T_full, D, D);
        launch_matmul(ws.U.ptr, L.W_K.ptr, ws.K.ptr, T_full, D, D);
        launch_matmul(ws.U.ptr, L.W_V.ptr, ws.V.ptr, T_full, D, D);
        // Causal MHA  → attn_out [T, D]
        launch_causal_mha(ws.Q.ptr, ws.K.ptr, ws.V.ptr,
                          ws.scores.ptr, ws.attn_out.ptr, T_full, H, dh);
        // O = attn_out @ W_O
        launch_matmul(ws.attn_out.ptr, L.W_O.ptr, ws.O.ptr, T_full, D, D);
        // R = X + O
        launch_add(ws.X.ptr, ws.O.ptr, ws.R.ptr, T_full * D);
        // Z = RMSNorm(R)
        launch_rmsnorm(ws.R.ptr, L.g_ffn.ptr, ws.Z.ptr, T_full, D, cfg.rms_eps);
        // F(Z) = W_down(SiLU(Z W_up) ⊙ (Z W_gate))
        launch_swiglu(ws.Z.ptr, L.W_up.ptr, L.W_gate.ptr, L.W_down.ptr,
                      ws.ffn_up.ptr, ws.ffn_gate.ptr, ws.ffn_mul.ptr,
                      ws.ffn_out.ptr, T_full, D, F);
        // X_next = R + F(Z)  (reuse ws.X as destination — all inputs already consumed)
        launch_add(ws.R.ptr, ws.ffn_out.ptr, ws.X.ptr, T_full * D);
    }

    // Y = RMSNorm(X)
    launch_rmsnorm(ws.X.ptr, w.g_final.ptr, ws.Y.ptr, T_full, D, cfg.rms_eps);
    // logits = Y · Eᵀ
    launch_logits_tied(ws.Y.ptr, w.E.ptr, ws.logits.ptr, T_full, D, V);

    // Copy only the logits row corresponding to the last real token back.
    last_logits_host.resize(V);
    const float* src = ws.logits.ptr + static_cast<size_t>(T - 1) * V;
    CUDA_CHECK(cudaMemcpy(last_logits_host.data(), src,
                          V * sizeof(float), cudaMemcpyDeviceToHost));
}

std::vector<int> generate_greedy(const ModelWeights& w, const ModelConfig& cfg,
                                 Workspace& ws,
                                 const std::vector<int>& prompt_ids,
                                 int n_new) {
    std::vector<int> seq = prompt_ids;
    if (seq.empty()) die("empty prompt");
    if (static_cast<int>(seq.size()) > cfg.seq_len) {
        // Truncate from the left to fit seq_len (with room for generation
        // we keep at most seq_len tokens total; oldest drop first).
        seq.erase(seq.begin(), seq.begin() + (seq.size() - cfg.seq_len));
    }

    std::vector<float> logits;
    for (int step = 0; step < n_new; ++step) {
        // If we'd overflow by adding one more token, drop the oldest.
        if (static_cast<int>(seq.size()) == cfg.seq_len) {
            seq.erase(seq.begin());
        }
        model_forward(w, cfg, ws, seq, logits);
        // Greedy argmax.
        int best_id = 0;
        float best = logits[0];
        for (int v = 1; v < cfg.vocab_size; ++v) {
            if (logits[v] > best) { best = logits[v]; best_id = v; }
        }
        seq.push_back(best_id);
    }
    return seq;
}

}  // namespace toyllm
