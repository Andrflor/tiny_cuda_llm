#pragma once

#include "config.h"
#include "tensor.cuh"
#include <vector>

namespace toyllm {

// Per-layer parameters on device.
struct LayerWeights {
    // Pre-attention RMSNorm scale [D]
    DeviceBuffer<float> g_attn;
    // QKV projections [D, D]
    DeviceBuffer<float> W_Q;
    DeviceBuffer<float> W_K;
    DeviceBuffer<float> W_V;
    DeviceBuffer<float> W_O;
    // Pre-FFN RMSNorm scale [D]
    DeviceBuffer<float> g_ffn;
    // SwiGLU FFN
    DeviceBuffer<float> W_up;    // [D, F]
    DeviceBuffer<float> W_gate;  // [D, F]
    DeviceBuffer<float> W_down;  // [F, D]
};

struct ModelWeights {
    // Token embeddings [V, D], also used as tied output projection.
    DeviceBuffer<float> E;
    // Final RMSNorm scale [D]
    DeviceBuffer<float> g_final;
    std::vector<LayerWeights> layers;
};

// Workspace device buffers reused across layers.
struct Workspace {
    DeviceBuffer<float> X;            // [T, D]  (residual stream)
    DeviceBuffer<float> U;            // [T, D]  (normed)
    DeviceBuffer<float> Q;            // [T, D]
    DeviceBuffer<float> K;            // [T, D]
    DeviceBuffer<float> V;            // [T, D]
    DeviceBuffer<float> scores;       // [H, T, T]
    DeviceBuffer<float> attn_out;     // [T, D]
    DeviceBuffer<float> O;            // [T, D]
    DeviceBuffer<float> R;            // [T, D]
    DeviceBuffer<float> Z;            // [T, D]
    DeviceBuffer<float> ffn_up;       // [T, F]
    DeviceBuffer<float> ffn_gate;     // [T, F]
    DeviceBuffer<float> ffn_mul;      // [T, F]
    DeviceBuffer<float> ffn_out;      // [T, D]
    DeviceBuffer<float> Y;            // [T, D]  (final normed)
    DeviceBuffer<float> logits;       // [T, V]
    DeviceBuffer<int>   ids;          // [T]
};

void init_random_weights(ModelWeights& w, const ModelConfig& cfg);

void alloc_workspace(Workspace& ws, const ModelConfig& cfg);

// Run forward: ids[0..T-1] → logits[0..T-1, V].
// T_actual ≤ cfg.seq_len; positions ≥ T_actual are padded with 0 (ignored).
void model_forward(const ModelWeights& w, const ModelConfig& cfg,
                   Workspace& ws, const std::vector<int>& ids,
                   std::vector<float>& last_logits_host);

// Greedy generation: starts from prompt_ids, generates n_new tokens, returns the
// concatenated sequence (prompt + generated).
std::vector<int> generate_greedy(const ModelWeights& w, const ModelConfig& cfg,
                                 Workspace& ws,
                                 const std::vector<int>& prompt_ids,
                                 int n_new);

}  // namespace toyllm
