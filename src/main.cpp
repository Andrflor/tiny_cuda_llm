#include "config.h"
#include "common.h"
#include "tokenizer.h"
#include "model.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace toyllm;

namespace toyllm { int cmd_train(int argc, char** argv); }

static void usage() {
    std::fprintf(stderr,
        "usage:\n"
        "  toy_llm train-bpe <corpus> <vocab-out> <merges-out> <nb-merges>\n"
        "  toy_llm encode    <vocab> <merges> <text>\n"
        "  toy_llm decode    <vocab> <merges> <id> [id ...]\n"
        "  toy_llm train     <corpus> <vocab> <merges> <ckpt-out> <steps> [lr]\n"
        "  toy_llm generate  <vocab> <merges> <prompt-file> <nb-tokens> [--weights <ckpt>]\n");
}

static int cmd_train_bpe(int argc, char** argv) {
    if (argc != 6) { usage(); return 1; }
    std::string corpus_path = argv[2];
    std::string vocab_path  = argv[3];
    std::string merges_path = argv[4];
    int n_merges = std::atoi(argv[5]);
    if (n_merges <= 0) die("invalid nb-merges");

    std::string text = read_file(corpus_path);
    std::fprintf(stderr, "[bpe] corpus bytes: %zu\n", text.size());
    Tokenizer tk = train_bpe(text, n_merges);
    std::fprintf(stderr, "[bpe] vocab size  : %d (merges: %zu)\n",
                 tk.vocab_size(), tk.merges.size());
    save_tokenizer(tk, vocab_path, merges_path);
    std::fprintf(stderr, "[bpe] saved %s / %s\n",
                 vocab_path.c_str(), merges_path.c_str());
    return 0;
}

static int cmd_encode(int argc, char** argv) {
    if (argc != 5) { usage(); return 1; }
    Tokenizer tk = load_tokenizer(argv[2], argv[3]);
    std::string text = argv[4];
    auto ids = tk.encode(text);
    for (size_t i = 0; i < ids.size(); ++i) {
        std::printf("%s%d", i ? " " : "", ids[i]);
    }
    std::printf("\n");
    return 0;
}

static int cmd_decode(int argc, char** argv) {
    if (argc < 5) { usage(); return 1; }
    Tokenizer tk = load_tokenizer(argv[2], argv[3]);
    std::vector<int> ids;
    ids.reserve(argc - 4);
    for (int i = 4; i < argc; ++i) ids.push_back(std::atoi(argv[i]));
    std::string out = tk.decode(ids);
    std::fwrite(out.data(), 1, out.size(), stdout);
    std::printf("\n");
    return 0;
}

static int cmd_generate(int argc, char** argv) {
    // Positional: <vocab> <merges> <prompt-file> <nb-tokens>
    // Optional: --weights <ckpt>
    if (argc != 6 && argc != 8) { usage(); return 1; }
    std::string vocab_path = argv[2];
    std::string merges_path = argv[3];
    std::string prompt_file = argv[4];
    int n_new = std::atoi(argv[5]);
    if (n_new <= 0) die("invalid nb-tokens");
    std::string weights_path;
    if (argc == 8) {
        if (std::strcmp(argv[6], "--weights") != 0) { usage(); return 1; }
        weights_path = argv[7];
    }

    Tokenizer tk = load_tokenizer(vocab_path, merges_path);
    std::string prompt = read_file(prompt_file);
    while (!prompt.empty() && (prompt.back() == '\n' || prompt.back() == '\r')) {
        prompt.pop_back();
    }

    ModelConfig cfg;
    cfg.vocab_size = tk.vocab_size();

    ModelWeights w;
    if (!weights_path.empty()) {
        load_checkpoint(weights_path, w, cfg);
        if (cfg.vocab_size != tk.vocab_size()) {
            std::fprintf(stderr,
                "[warn] checkpoint vocab=%d but tokenizer vocab=%d\n",
                cfg.vocab_size, tk.vocab_size());
        }
        std::fprintf(stderr, "[gen] loaded weights from %s\n", weights_path.c_str());
    } else {
        init_random_weights(w, cfg);
        std::fprintf(stderr, "[gen] using random weights (seed %llu)\n",
                     (unsigned long long)cfg.seed);
    }

    auto prompt_ids = tk.encode(prompt);
    if (prompt_ids.empty()) die("empty prompt after encoding");
    std::fprintf(stderr, "[gen] prompt ids: %zu, vocab=%d, seq_len=%d\n",
                 prompt_ids.size(), cfg.vocab_size, cfg.seq_len);

    Workspace ws;
    alloc_workspace(ws, cfg);

    auto full = generate_greedy(w, cfg, ws, prompt_ids, n_new);

    std::fprintf(stderr, "[gen] full ids (%zu): ", full.size());
    for (size_t i = 0; i < full.size(); ++i) {
        std::fprintf(stderr, "%s%d", i ? " " : "", full[i]);
    }
    std::fprintf(stderr, "\n");

    std::string text = tk.decode(full);
    std::fwrite(text.data(), 1, text.size(), stdout);
    std::printf("\n");
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    std::string cmd = argv[1];
    if (cmd == "train-bpe") return cmd_train_bpe(argc, argv);
    if (cmd == "encode")    return cmd_encode(argc, argv);
    if (cmd == "decode")    return cmd_decode(argc, argv);
    if (cmd == "train")     return cmd_train(argc, argv);
    if (cmd == "generate")  return cmd_generate(argc, argv);
    usage();
    return 1;
}
