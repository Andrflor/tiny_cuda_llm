#pragma once

namespace toyllm {

struct ModelConfig {
    int vocab_size = 512;
    int seq_len   = 32;
    int d_model   = 128;
    int n_layers  = 2;
    int n_heads   = 4;
    int head_dim  = 32;      // d_model = n_heads * head_dim
    int ffn_dim   = 256;
    float rms_eps = 1e-5f;
    float rope_base = 10000.0f;
    unsigned long long seed = 42ULL;
};

}  // namespace toyllm
