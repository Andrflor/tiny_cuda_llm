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

    auto scale_for = [](int fan_in) {
        return 1.0f / std::sqrt(static_cast<float>(fan_in));
    };

    {
        std::vector<float> E(V * D);
        fill_uniform(E, rng, scale_for(D));
        w.E.allocate(V * D);
        w.E.copy_from_host(E);
    }
    {
        std::vector<float> g(D, 1.0f);
        w.g_final.allocate(D);
        w.g_final.copy_from_host(g);
    }

    w.layers.resize(cfg.n_layers);
    for (int l = 0; l < cfg.n_layers; ++l) {
        auto& L = w.layers[l];
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

void alloc_grads(ModelGrads& g, const ModelConfig& cfg) {
    const int V = cfg.vocab_size;
    const int D = cfg.d_model;
    const int F = cfg.ffn_dim;
    g.E.allocate(V * D);
    g.g_final.allocate(D);
    g.layers.resize(cfg.n_layers);
    for (int l = 0; l < cfg.n_layers; ++l) {
        auto& L = g.layers[l];
        L.g_attn.allocate(D);
        L.W_Q.allocate(D * D);
        L.W_K.allocate(D * D);
        L.W_V.allocate(D * D);
        L.W_O.allocate(D * D);
        L.g_ffn.allocate(D);
        L.W_up.allocate(D * F);
        L.W_gate.allocate(D * F);
        L.W_down.allocate(F * D);
    }
}

void zero_grads(ModelGrads& g) {
    g.E.zero();
    g.g_final.zero();
    for (auto& L : g.layers) {
        L.g_attn.zero(); L.W_Q.zero(); L.W_K.zero(); L.W_V.zero(); L.W_O.zero();
        L.g_ffn.zero(); L.W_up.zero(); L.W_gate.zero(); L.W_down.zero();
    }
}

void alloc_train_workspace(TrainWorkspace& tw, const ModelConfig& cfg) {
    const int T = cfg.seq_len;
    const int D = cfg.d_model;
    const int H = cfg.n_heads;
    const int F = cfg.ffn_dim;
    const int V = cfg.vocab_size;

    tw.caches.resize(cfg.n_layers);
    for (auto& c : tw.caches) {
        c.X_in.allocate(T * D);
        c.U.allocate(T * D);
        c.Q_post.allocate(T * D);
        c.K_post.allocate(T * D);
        c.V.allocate(T * D);
        c.A.allocate(H * T * T);
        c.attn_out.allocate(T * D);
        c.R.allocate(T * D);
        c.Z.allocate(T * D);
        c.ffn_up.allocate(T * F);
        c.ffn_gate.allocate(T * F);
        c.ffn_mul.allocate(T * F);
        c.X_out.allocate(T * D);
    }
    tw.X0.allocate(T * D);
    tw.Y.allocate(T * D);
    tw.logits.allocate(T * V);
    tw.ids.allocate(T);
    tw.targets.allocate(T);
    tw.loss_scalar.allocate(1);

    tw.dX.allocate(T * D);
    tw.dU.allocate(T * D);
    tw.dQ.allocate(T * D);
    tw.dK.allocate(T * D);
    tw.dV_act.allocate(T * D);
    tw.dS.allocate(H * T * T);
    tw.dattn_out.allocate(T * D);
    tw.dO.allocate(T * D);
    tw.dR.allocate(T * D);
    tw.dZ.allocate(T * D);
    tw.dffn_up.allocate(T * F);
    tw.dffn_gate.allocate(T * F);
    tw.dffn_mul.allocate(T * F);
    tw.dffn_out.allocate(T * D);
    tw.dY.allocate(T * D);
    tw.dlogits.allocate(T * V);
    tw.dX_next.allocate(T * D);
    tw.scratch_TD.allocate(T * D);
}

void alloc_optimizer(OptState& opt, const ModelConfig& cfg) {
    const int V = cfg.vocab_size;
    const int D = cfg.d_model;
    const int F = cfg.ffn_dim;

    auto init_state = [](AdamWState& s, int n) {
        s.m.allocate(n); s.m.zero();
        s.v.allocate(n); s.v.zero();
    };
    init_state(opt.E, V * D);
    init_state(opt.g_final, D);
    int L = cfg.n_layers;
    opt.layers_g_attn.resize(L);
    opt.layers_W_Q.resize(L);
    opt.layers_W_K.resize(L);
    opt.layers_W_V.resize(L);
    opt.layers_W_O.resize(L);
    opt.layers_g_ffn.resize(L);
    opt.layers_W_up.resize(L);
    opt.layers_W_gate.resize(L);
    opt.layers_W_down.resize(L);
    for (int l = 0; l < L; ++l) {
        init_state(opt.layers_g_attn[l], D);
        init_state(opt.layers_W_Q[l], D * D);
        init_state(opt.layers_W_K[l], D * D);
        init_state(opt.layers_W_V[l], D * D);
        init_state(opt.layers_W_O[l], D * D);
        init_state(opt.layers_g_ffn[l], D);
        init_state(opt.layers_W_up[l],   D * F);
        init_state(opt.layers_W_gate[l], D * F);
        init_state(opt.layers_W_down[l], F * D);
    }
    opt.step = 0;
}

// -----------------------------------------------------------------------------
// Inference forward (unchanged).
// -----------------------------------------------------------------------------
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

    std::vector<int> padded(T_full, 0);
    for (int t = 0; t < T; ++t) padded[t] = ids[t];
    ws.ids.copy_from_host(padded);

    launch_embedding_lookup(ws.ids.ptr, w.E.ptr, ws.X.ptr, T_full, V, D);

    for (int l = 0; l < cfg.n_layers; ++l) {
        const auto& L = w.layers[l];
        launch_rmsnorm(ws.X.ptr, L.g_attn.ptr, ws.U.ptr, T_full, D, cfg.rms_eps);
        launch_matmul(ws.U.ptr, L.W_Q.ptr, ws.Q.ptr, T_full, D, D);
        launch_matmul(ws.U.ptr, L.W_K.ptr, ws.K.ptr, T_full, D, D);
        launch_matmul(ws.U.ptr, L.W_V.ptr, ws.V.ptr, T_full, D, D);
        launch_rope(ws.Q.ptr, T_full, H, dh, cfg.rope_base);
        launch_rope(ws.K.ptr, T_full, H, dh, cfg.rope_base);
        launch_causal_mha(ws.Q.ptr, ws.K.ptr, ws.V.ptr,
                          ws.scores.ptr, ws.attn_out.ptr, T_full, H, dh);
        launch_matmul(ws.attn_out.ptr, L.W_O.ptr, ws.O.ptr, T_full, D, D);
        launch_add(ws.X.ptr, ws.O.ptr, ws.R.ptr, T_full * D);
        launch_rmsnorm(ws.R.ptr, L.g_ffn.ptr, ws.Z.ptr, T_full, D, cfg.rms_eps);
        launch_swiglu(ws.Z.ptr, L.W_up.ptr, L.W_gate.ptr, L.W_down.ptr,
                      ws.ffn_up.ptr, ws.ffn_gate.ptr, ws.ffn_mul.ptr,
                      ws.ffn_out.ptr, T_full, D, F);
        launch_add(ws.R.ptr, ws.ffn_out.ptr, ws.X.ptr, T_full * D);
    }

    launch_rmsnorm(ws.X.ptr, w.g_final.ptr, ws.Y.ptr, T_full, D, cfg.rms_eps);
    launch_logits_tied(ws.Y.ptr, w.E.ptr, ws.logits.ptr, T_full, D, V);

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
        seq.erase(seq.begin(), seq.begin() + (seq.size() - cfg.seq_len));
    }
    std::vector<float> logits;
    for (int step = 0; step < n_new; ++step) {
        if (static_cast<int>(seq.size()) == cfg.seq_len) seq.erase(seq.begin());
        model_forward(w, cfg, ws, seq, logits);
        int best_id = 0;
        float best = logits[0];
        for (int v = 1; v < cfg.vocab_size; ++v) {
            if (logits[v] > best) { best = logits[v]; best_id = v; }
        }
        seq.push_back(best_id);
    }
    return seq;
}

// -----------------------------------------------------------------------------
// Training forward + loss + backward
// -----------------------------------------------------------------------------
void model_forward_train(const ModelWeights& w, const ModelConfig& cfg,
                         TrainWorkspace& tw, const std::vector<int>& ids_padded) {
    const int T = cfg.seq_len;
    const int D = cfg.d_model;
    const int H = cfg.n_heads;
    const int dh = cfg.head_dim;
    const int F = cfg.ffn_dim;
    const int V = cfg.vocab_size;
    if (static_cast<int>(ids_padded.size()) != T)
        die("model_forward_train: ids must be padded to seq_len");

    tw.ids.copy_from_host(ids_padded);

    // X0 = embed(ids)
    launch_embedding_lookup(tw.ids.ptr, w.E.ptr, tw.X0.ptr, T, V, D);

    // Input to layer 0 is X0. We stream X through caches[l].X_in / X_out.
    const float* X_cur = tw.X0.ptr;
    for (int l = 0; l < cfg.n_layers; ++l) {
        auto& c = tw.caches[l];
        const auto& L = w.layers[l];

        // Save X_in (input to this block).
        CUDA_CHECK(cudaMemcpy(c.X_in.ptr, X_cur, sizeof(float) * T * D,
                              cudaMemcpyDeviceToDevice));

        // U = RMSNorm(X_in)
        launch_rmsnorm(c.X_in.ptr, L.g_attn.ptr, c.U.ptr, T, D, cfg.rms_eps);
        // Q, K, V
        launch_matmul(c.U.ptr, L.W_Q.ptr, c.Q_post.ptr, T, D, D);
        launch_matmul(c.U.ptr, L.W_K.ptr, c.K_post.ptr, T, D, D);
        launch_matmul(c.U.ptr, L.W_V.ptr, c.V.ptr, T, D, D);
        // RoPE in-place on Q, K.
        launch_rope(c.Q_post.ptr, T, H, dh, cfg.rope_base);
        launch_rope(c.K_post.ptr, T, H, dh, cfg.rope_base);
        // Attention: c.A holds probs after softmax; c.attn_out is output.
        launch_causal_mha(c.Q_post.ptr, c.K_post.ptr, c.V.ptr,
                          c.A.ptr, c.attn_out.ptr, T, H, dh);
        // O = attn_out @ W_O ; reuse tw.scratch_TD to hold O, then R = X_in + O.
        launch_matmul(c.attn_out.ptr, L.W_O.ptr, tw.scratch_TD.ptr, T, D, D);
        launch_add(c.X_in.ptr, tw.scratch_TD.ptr, c.R.ptr, T * D);
        // Z = RMSNorm(R)
        launch_rmsnorm(c.R.ptr, L.g_ffn.ptr, c.Z.ptr, T, D, cfg.rms_eps);
        // SwiGLU hidden + output into tw.scratch_TD (re-use). We need X_out = R + ffn_out
        launch_swiglu(c.Z.ptr, L.W_up.ptr, L.W_gate.ptr, L.W_down.ptr,
                      c.ffn_up.ptr, c.ffn_gate.ptr, c.ffn_mul.ptr,
                      tw.scratch_TD.ptr, T, D, F);
        launch_add(c.R.ptr, tw.scratch_TD.ptr, c.X_out.ptr, T * D);

        X_cur = c.X_out.ptr;
    }

    // Final RMSNorm + tied logits.
    launch_rmsnorm(X_cur, w.g_final.ptr, tw.Y.ptr, T, D, cfg.rms_eps);
    launch_logits_tied(tw.Y.ptr, w.E.ptr, tw.logits.ptr, T, D, V);
}

float compute_loss(const ModelConfig& cfg, TrainWorkspace& tw,
                   const std::vector<int>& targets_padded) {
    const int T = cfg.seq_len;
    const int V = cfg.vocab_size;
    if (static_cast<int>(targets_padded.size()) != T)
        die("compute_loss: targets must be padded to seq_len");
    tw.targets.copy_from_host(targets_padded);
    launch_cross_entropy(tw.logits.ptr, tw.targets.ptr,
                         tw.loss_scalar.ptr, tw.dlogits.ptr, T, V);
    float host_loss = 0.0f;
    CUDA_CHECK(cudaMemcpy(&host_loss, tw.loss_scalar.ptr, sizeof(float),
                          cudaMemcpyDeviceToHost));
    return host_loss;
}

void model_backward(const ModelWeights& w, const ModelConfig& cfg,
                    TrainWorkspace& tw, ModelGrads& g) {
    const int T = cfg.seq_len;
    const int D = cfg.d_model;
    const int H = cfg.n_heads;
    const int dh = cfg.head_dim;
    const int F = cfg.ffn_dim;
    const int V = cfg.vocab_size;

    // 1) Logits tied: seeds dY and accumulates dE (tied → dE gets logits contrib here
    //    and embedding contrib later).
    // tw.dlogits already holds (softmax - onehot)/n from compute_loss.
    launch_logits_tied_backward(tw.dlogits.ptr, tw.Y.ptr, w.E.ptr,
                                tw.dY.ptr, g.E.ptr, T, D, V);

    // 2) Final RMSNorm backward into dX_next (grad at input of final rmsnorm,
    //    which is X_out of last layer). Uses input X_out (tw.caches.back().X_out).
    const float* X_after_stack = tw.caches.back().X_out.ptr;
    launch_rmsnorm_backward(tw.dY.ptr, X_after_stack, w.g_final.ptr,
                            tw.dX_next.ptr, g.g_final.ptr,
                            T, D, cfg.rms_eps);

    // 3) Walk layers in reverse.
    for (int l = cfg.n_layers - 1; l >= 0; --l) {
        auto& c = tw.caches[l];
        const auto& L = w.layers[l];
        auto& dL = g.layers[l];

        // X_out = R + ffn_out  ==>  dR += dX_next, dffn_out = dX_next
        CUDA_CHECK(cudaMemcpy(tw.dR.ptr, tw.dX_next.ptr,
                              sizeof(float) * T * D, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(tw.dffn_out.ptr, tw.dX_next.ptr,
                              sizeof(float) * T * D, cudaMemcpyDeviceToDevice));

        // SwiGLU backward: dZ, dW_up, dW_gate, dW_down (and dmul/dup/dgate scratch).
        launch_swiglu_backward(c.Z.ptr, L.W_up.ptr, L.W_gate.ptr, L.W_down.ptr,
                               c.ffn_up.ptr, c.ffn_gate.ptr, c.ffn_mul.ptr,
                               tw.dffn_out.ptr,
                               tw.dffn_mul.ptr, tw.dffn_up.ptr, tw.dffn_gate.ptr,
                               tw.dZ.ptr, dL.W_up.ptr, dL.W_gate.ptr, dL.W_down.ptr,
                               T, D, F);

        // RMSNorm2 backward: from dZ → dR (accumulate into existing dR) and dg_ffn.
        // launch_rmsnorm_backward writes dx; we need to accumulate dR. Do it in
        // scratch then axpy.
        launch_rmsnorm_backward(tw.dZ.ptr, c.R.ptr, L.g_ffn.ptr,
                                tw.scratch_TD.ptr, dL.g_ffn.ptr,
                                T, D, cfg.rms_eps);
        launch_axpy(1.0f, tw.scratch_TD.ptr, tw.dR.ptr, T * D);

        // R = X_in + O  ==>  dX_in = dR, dO = dR
        // (We defer storing dX_in into dX_next until we've combined with the
        //  attention branch.)
        CUDA_CHECK(cudaMemcpy(tw.dO.ptr, tw.dR.ptr,
                              sizeof(float) * T * D, cudaMemcpyDeviceToDevice));
        // Initialize dX_next (for this layer's X_in grad) with dR.
        CUDA_CHECK(cudaMemcpy(tw.dX_next.ptr, tw.dR.ptr,
                              sizeof(float) * T * D, cudaMemcpyDeviceToDevice));

        // O = attn_out @ W_O  ==>  dattn_out = dO @ W_Oᵀ,  dW_O += attn_outᵀ @ dO
        launch_matmul_backward(tw.dO.ptr, c.attn_out.ptr, L.W_O.ptr,
                               tw.dattn_out.ptr, dL.W_O.ptr,
                               T, D, D);

        // Attention backward: accumulate into dQ, dK, dV (zero first).
        tw.dQ.zero();
        tw.dK.zero();
        tw.dV_act.zero();
        launch_causal_mha_backward(c.Q_post.ptr, c.K_post.ptr, c.V.ptr,
                                   c.A.ptr, tw.dattn_out.ptr,
                                   tw.dS.ptr,
                                   tw.dQ.ptr, tw.dK.ptr, tw.dV_act.ptr,
                                   T, H, dh);

        // RoPE backward on dQ and dK (in-place).
        launch_rope_backward(tw.dQ.ptr, T, H, dh, cfg.rope_base);
        launch_rope_backward(tw.dK.ptr, T, H, dh, cfg.rope_base);

        // Q = U @ W_Q  ==>  dU += dQ @ W_Qᵀ,  dW_Q += Uᵀ @ dQ
        // K / V analogues. Accumulate dU in scratch_TD starting from the Q branch,
        // then axpy for K and V.
        launch_matmul_backward(tw.dQ.ptr, c.U.ptr, L.W_Q.ptr,
                               tw.dU.ptr, dL.W_Q.ptr, T, D, D);
        launch_matmul_backward(tw.dK.ptr, c.U.ptr, L.W_K.ptr,
                               tw.scratch_TD.ptr, dL.W_K.ptr, T, D, D);
        launch_axpy(1.0f, tw.scratch_TD.ptr, tw.dU.ptr, T * D);
        launch_matmul_backward(tw.dV_act.ptr, c.U.ptr, L.W_V.ptr,
                               tw.scratch_TD.ptr, dL.W_V.ptr, T, D, D);
        launch_axpy(1.0f, tw.scratch_TD.ptr, tw.dU.ptr, T * D);

        // RMSNorm1 backward: dX from dU, wrt X_in and g_attn.
        launch_rmsnorm_backward(tw.dU.ptr, c.X_in.ptr, L.g_attn.ptr,
                                tw.scratch_TD.ptr, dL.g_attn.ptr,
                                T, D, cfg.rms_eps);
        // dX_in (from attention branch) to add to dX_next (which already holds
        // the residual-bypass grad dR).
        launch_axpy(1.0f, tw.scratch_TD.ptr, tw.dX_next.ptr, T * D);
        // Now tw.dX_next is dX at the *input* of this layer == grad to pass to
        // the previous layer's X_out.
    }

    // 4) Embedding backward: tw.dX_next now holds dX0 (grad of embedding output).
    //    g.E was already seeded by the tied-logits path; accumulate scatter-add.
    launch_embedding_backward(tw.ids.ptr, tw.dX_next.ptr, g.E.ptr, T, V, D);
}

void apply_adamw(ModelWeights& w, const ModelGrads& g, OptState& opt,
                 const ModelConfig& cfg,
                 float lr, float beta1, float beta2, float eps, float wd) {
    opt.step += 1;
    int step = opt.step;
    const int V = cfg.vocab_size;
    const int D = cfg.d_model;
    const int F = cfg.ffn_dim;

    // Embeddings + final gain (no weight decay on gains).
    launch_adamw_step(w.E.ptr, g.E.ptr, opt.E.m.ptr, opt.E.v.ptr, V * D,
                      step, lr, beta1, beta2, eps, wd);
    launch_adamw_step(w.g_final.ptr, g.g_final.ptr,
                      opt.g_final.m.ptr, opt.g_final.v.ptr, D,
                      step, lr, beta1, beta2, eps, 0.0f);

    for (int l = 0; l < cfg.n_layers; ++l) {
        auto& L = w.layers[l];
        const auto& dL = g.layers[l];
        auto step_gain = [&](DeviceBuffer<float>& p, const DeviceBuffer<float>& gg,
                             AdamWState& s, int n) {
            launch_adamw_step(p.ptr, gg.ptr, s.m.ptr, s.v.ptr, n,
                              step, lr, beta1, beta2, eps, 0.0f);
        };
        auto step_mat = [&](DeviceBuffer<float>& p, const DeviceBuffer<float>& gg,
                            AdamWState& s, int n) {
            launch_adamw_step(p.ptr, gg.ptr, s.m.ptr, s.v.ptr, n,
                              step, lr, beta1, beta2, eps, wd);
        };
        step_gain(L.g_attn, dL.g_attn, opt.layers_g_attn[l], D);
        step_mat (L.W_Q,   dL.W_Q,   opt.layers_W_Q[l],   D * D);
        step_mat (L.W_K,   dL.W_K,   opt.layers_W_K[l],   D * D);
        step_mat (L.W_V,   dL.W_V,   opt.layers_W_V[l],   D * D);
        step_mat (L.W_O,   dL.W_O,   opt.layers_W_O[l],   D * D);
        step_gain(L.g_ffn, dL.g_ffn, opt.layers_g_ffn[l], D);
        step_mat (L.W_up,   dL.W_up,   opt.layers_W_up[l],   D * F);
        step_mat (L.W_gate, dL.W_gate, opt.layers_W_gate[l], D * F);
        step_mat (L.W_down, dL.W_down, opt.layers_W_down[l], F * D);
    }
}

}  // namespace toyllm
