// Pure-CPU tokenizer CLI. No CUDA dependency — for quickly iterating on the
// BPE trainer without needing a GPU.

#include "tokenizer.h"
#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace toyllm;

static void usage() {
    std::fprintf(stderr,
        "usage:\n"
        "  toy_tokenizer train  <corpus> <vocab-out> <merges-out> <nb-merges>\n"
        "  toy_tokenizer encode <vocab> <merges> <text>\n"
        "  toy_tokenizer decode <vocab> <merges> <id> [id ...]\n");
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    std::string cmd = argv[1];

    if (cmd == "train") {
        if (argc != 6) { usage(); return 1; }
        std::string text = read_file(argv[2]);
        int n_merges = std::atoi(argv[5]);
        if (n_merges <= 0) die("invalid nb-merges");
        Tokenizer tk = train_bpe(text, n_merges);
        save_tokenizer(tk, argv[3], argv[4]);
        std::fprintf(stderr, "[bpe] vocab=%d merges=%zu\n",
                     tk.vocab_size(), tk.merges.size());
        return 0;
    }
    if (cmd == "encode") {
        if (argc != 5) { usage(); return 1; }
        Tokenizer tk = load_tokenizer(argv[2], argv[3]);
        auto ids = tk.encode(argv[4]);
        for (size_t i = 0; i < ids.size(); ++i) {
            std::printf("%s%d", i ? " " : "", ids[i]);
        }
        std::printf("\n");
        return 0;
    }
    if (cmd == "decode") {
        if (argc < 5) { usage(); return 1; }
        Tokenizer tk = load_tokenizer(argv[2], argv[3]);
        std::vector<int> ids;
        for (int i = 4; i < argc; ++i) ids.push_back(std::atoi(argv[i]));
        std::string out = tk.decode(ids);
        std::fwrite(out.data(), 1, out.size(), stdout);
        std::printf("\n");
        return 0;
    }
    usage();
    return 1;
}
