#pragma once

#include "config.h"
#include "tensor.cuh"
#include <string>
#include <vector>

namespace toyllm {

// Per-layer parameters on device.
struct LayerWeights {
    DeviceBuffer<float> g_attn;   // [D]
    DeviceBuffer<float> W_Q;      // [D, D]
    DeviceBuffer<float> W_K;      // [D, D]
    DeviceBuffer<float> W_V;      // [D, D]
    DeviceBuffer<float> W_O;      // [D, D]
    DeviceBuffer<float> g_ffn;    // [D]
    DeviceBuffer<float> W_up;     // [D, F]
    DeviceBuffer<float> W_gate;   // [D, F]
    DeviceBuffer<float> W_down;   // [F, D]
};

struct ModelWeights {
    DeviceBuffer<float> E;        // [V, D], tied
    DeviceBuffer<float> g_final;  // [D]
    std::vector<LayerWeights> layers;
};

// Per-layer gradient storage (mirrors LayerWeights shapes).
struct LayerGrads {
    DeviceBuffer<float> g_attn;
    DeviceBuffer<float> W_Q;
    DeviceBuffer<float> W_K;
    DeviceBuffer<float> W_V;
    DeviceBuffer<float> W_O;
    DeviceBuffer<float> g_ffn;
    DeviceBuffer<float> W_up;
    DeviceBuffer<float> W_gate;
    DeviceBuffer<float> W_down;
};

struct ModelGrads {
    DeviceBuffer<float> E;
    DeviceBuffer<float> g_final;
    std::vector<LayerGrads> layers;
};

// Activations cached by the forward pass — required by backward.
struct LayerCache {
    DeviceBuffer<float> X_in;        // [T, D] residual stream entering the block
    DeviceBuffer<float> U;           // [T, D] RMSNorm1(X_in)
    DeviceBuffer<float> Q_post;      // [T, D] Q after RoPE
    DeviceBuffer<float> K_post;      // [T, D] K after RoPE
    DeviceBuffer<float> V;           // [T, D]
    DeviceBuffer<float> A;           // [H, T, T] attention probs (post-softmax)
    DeviceBuffer<float> attn_out;    // [T, D]
    DeviceBuffer<float> R;           // [T, D]
    DeviceBuffer<float> Z;           // [T, D] RMSNorm2(R)
    DeviceBuffer<float> ffn_up;      // [T, F]
    DeviceBuffer<float> ffn_gate;    // [T, F]
    DeviceBuffer<float> ffn_mul;     // [T, F]
    DeviceBuffer<float> X_out;       // [T, D] residual after FFN (input of next layer)
};

// Workspace device buffers reused across layers (inference path).
struct Workspace {
    DeviceBuffer<float> X, U, Q, K, V;
    DeviceBuffer<float> scores;
    DeviceBuffer<float> attn_out, O, R, Z;
    DeviceBuffer<float> ffn_up, ffn_gate, ffn_mul, ffn_out;
    DeviceBuffer<float> Y;
    DeviceBuffer<float> logits;
    DeviceBuffer<int>   ids;
};

// Training workspace: activations cached per layer + scratch for backward.
struct TrainWorkspace {
    std::vector<LayerCache> caches;   // size = n_layers
    DeviceBuffer<float> X0;           // [T, D] embedding output (input of layer 0)
    DeviceBuffer<float> Y;            // [T, D] final RMSNorm output
    DeviceBuffer<float> logits;       // [T, V]
    DeviceBuffer<int>   ids;          // [T]
    DeviceBuffer<int>   targets;      // [T]
    DeviceBuffer<float> loss_scalar;  // [1]

    // Gradient buffers (activations-side, sized [T, D] or [T, F]).
    DeviceBuffer<float> dX, dU, dQ, dK, dV_act;
    DeviceBuffer<float> dS;              // [H, T, T]
    DeviceBuffer<float> dattn_out;       // [T, D]
    DeviceBuffer<float> dO;              // [T, D]
    DeviceBuffer<float> dR;              // [T, D]
    DeviceBuffer<float> dZ;              // [T, D]
    DeviceBuffer<float> dffn_up;         // [T, F]
    DeviceBuffer<float> dffn_gate;       // [T, F]
    DeviceBuffer<float> dffn_mul;        // [T, F]
    DeviceBuffer<float> dffn_out;        // [T, D]
    DeviceBuffer<float> dY;              // [T, D]
    DeviceBuffer<float> dlogits;         // [T, V]
    DeviceBuffer<float> dX_next;         // [T, D] working residual grad
    DeviceBuffer<float> scratch_TD;      // [T, D] scratch for add
};

// AdamW state (one entry per parameter tensor, sized to match the param).
struct AdamWState {
    DeviceBuffer<float> m;
    DeviceBuffer<float> v;
};

struct OptState {
    AdamWState E;
    AdamWState g_final;
    std::vector<AdamWState> layers_g_attn;
    std::vector<AdamWState> layers_W_Q;
    std::vector<AdamWState> layers_W_K;
    std::vector<AdamWState> layers_W_V;
    std::vector<AdamWState> layers_W_O;
    std::vector<AdamWState> layers_g_ffn;
    std::vector<AdamWState> layers_W_up;
    std::vector<AdamWState> layers_W_gate;
    std::vector<AdamWState> layers_W_down;
    int step = 0;
};

void init_random_weights(ModelWeights& w, const ModelConfig& cfg);
void alloc_workspace(Workspace& ws, const ModelConfig& cfg);
void alloc_grads(ModelGrads& g, const ModelConfig& cfg);
void zero_grads(ModelGrads& g);
void alloc_train_workspace(TrainWorkspace& tw, const ModelConfig& cfg);
void alloc_optimizer(OptState& opt, const ModelConfig& cfg);

// Inference forward (what was there before).
void model_forward(const ModelWeights& w, const ModelConfig& cfg,
                   Workspace& ws, const std::vector<int>& ids,
                   std::vector<float>& last_logits_host);

// Training forward : pads to T_full, caches activations for backward.
// `ids_padded` must have size cfg.seq_len.
void model_forward_train(const ModelWeights& w, const ModelConfig& cfg,
                         TrainWorkspace& tw, const std::vector<int>& ids_padded);

// Compute loss (mean over valid targets) and seed dlogits.
// targets_padded : size cfg.seq_len; -1 entries are ignored.
// Returns the host-side scalar loss.
float compute_loss(const ModelConfig& cfg, TrainWorkspace& tw,
                   const std::vector<int>& targets_padded);

// Training backward: uses tw.caches + tw.dlogits, accumulates into g.
// Call zero_grads(g) before each step.
void model_backward(const ModelWeights& w, const ModelConfig& cfg,
                    TrainWorkspace& tw, ModelGrads& g);

// Apply one AdamW step to all weights.
void apply_adamw(ModelWeights& w, const ModelGrads& g, OptState& opt,
                 const ModelConfig& cfg,
                 float lr, float beta1, float beta2, float eps, float wd);

// Greedy generation: starts from prompt_ids, generates n_new tokens.
std::vector<int> generate_greedy(const ModelWeights& w, const ModelConfig& cfg,
                                 Workspace& ws,
                                 const std::vector<int>& prompt_ids,
                                 int n_new);

// Checkpoint I/O.
void save_checkpoint(const std::string& path, const ModelWeights& w,
                     const ModelConfig& cfg);
// Loads cfg and weights; `w` and `cfg` are filled in.
void load_checkpoint(const std::string& path, ModelWeights& w, ModelConfig& cfg);

}  // namespace toyllm
