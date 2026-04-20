#include "config.h"
#include "model.cuh"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace toyllm;

int main() {
    ModelConfig cfg;
    cfg.vocab_size = 64;   // small for the smoke test
    cfg.seq_len    = 8;
    cfg.d_model    = 32;
    cfg.n_layers   = 2;
    cfg.n_heads    = 4;
    cfg.head_dim   = 8;    // 4*8 = 32
    cfg.ffn_dim    = 64;
    cfg.seed       = 123ULL;

    ModelWeights w;
    init_random_weights(w, cfg);
    Workspace ws;
    alloc_workspace(ws, cfg);

    std::vector<int> ids = {1, 2, 3, 4};

    std::vector<float> logits;
    model_forward(w, cfg, ws, ids, logits);

    assert(static_cast<int>(logits.size()) == cfg.vocab_size);
    // At least one finite logit.
    bool any_finite = false;
    for (float v : logits) if (std::isfinite(v)) any_finite = true;
    assert(any_finite);

    // Greedy generate a few tokens: must not crash and must produce ids in [0,V).
    auto full = generate_greedy(w, cfg, ws, ids, 4);
    assert(full.size() == ids.size() + 4);
    for (int id : full) {
        assert(id >= 0 && id < cfg.vocab_size);
    }

    std::printf("test_model_smoke: OK (generated %zu tokens)\n", full.size());
    return 0;
}
