#include "model.cuh"
#include "tokenizer.h"
#include "common.h"
#include "config.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace toyllm {

// train(corpus_path, vocab, merges, ckpt_out, steps, [lr]).
//
// At each step: sample a random contiguous window of seq_len+1 token ids from
// the encoded corpus. Input ids = window[:-1], targets = window[1:]. Pad with
// -1 if shorter (never here because we sample exactly seq_len+1). Run forward,
// loss, backward, AdamW step. Log loss every log_every steps.
int cmd_train(int argc, char** argv) {
    if (argc < 7 || argc > 8) {
        std::fprintf(stderr,
            "usage: toy_llm train <corpus> <vocab> <merges> <ckpt-out> <steps> [lr]\n");
        return 1;
    }
    std::string corpus_path = argv[2];
    std::string vocab_path  = argv[3];
    std::string merges_path = argv[4];
    std::string ckpt_path   = argv[5];
    int n_steps = std::atoi(argv[6]);
    float lr = (argc == 8) ? static_cast<float>(std::atof(argv[7])) : 3e-3f;
    if (n_steps <= 0) die("invalid steps");

    Tokenizer tk = load_tokenizer(vocab_path, merges_path);
    std::string text = read_file(corpus_path);
    auto tokens = tk.encode(text);
    std::fprintf(stderr, "[train] corpus bytes=%zu tokens=%zu vocab=%d\n",
                 text.size(), tokens.size(), tk.vocab_size());

    ModelConfig cfg;
    cfg.vocab_size = tk.vocab_size();

    if (static_cast<int>(tokens.size()) < cfg.seq_len + 2) {
        die("corpus too small for seq_len — need at least seq_len+2 tokens");
    }

    ModelWeights w;
    init_random_weights(w, cfg);
    ModelGrads g;
    alloc_grads(g, cfg);
    TrainWorkspace tw;
    alloc_train_workspace(tw, cfg);
    OptState opt;
    alloc_optimizer(opt, cfg);

    const float beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f, wd = 0.01f;
    const int T = cfg.seq_len;
    const int N = static_cast<int>(tokens.size());
    const int log_every = std::max(1, n_steps / 50);

    Xorshift64 rng(cfg.seed ^ 0xDEADBEEFULL);

    std::vector<int> ids_pad(T), tgt_pad(T);
    float ema_loss = -1.0f;

    for (int step = 1; step <= n_steps; ++step) {
        // Sample a window start in [0, N - (T+1)].
        int max_start = N - (T + 1);
        int s = static_cast<int>(rng.next_u64() % static_cast<uint64_t>(max_start + 1));
        for (int t = 0; t < T; ++t) {
            ids_pad[t] = tokens[s + t];
            tgt_pad[t] = tokens[s + t + 1];
        }

        zero_grads(g);
        model_forward_train(w, cfg, tw, ids_pad);
        float loss = compute_loss(cfg, tw, tgt_pad);
        model_backward(w, cfg, tw, g);
        apply_adamw(w, g, opt, cfg, lr, beta1, beta2, eps, wd);

        ema_loss = (ema_loss < 0.0f) ? loss : 0.98f * ema_loss + 0.02f * loss;
        if (step == 1 || step % log_every == 0 || step == n_steps) {
            std::fprintf(stderr, "[train] step %5d/%d  loss=%.4f  ema=%.4f\n",
                         step, n_steps, loss, ema_loss);
        }
    }

    save_checkpoint(ckpt_path, w, cfg);
    std::fprintf(stderr, "[train] saved %s\n", ckpt_path.c_str());
    return 0;
}

}  // namespace toyllm
