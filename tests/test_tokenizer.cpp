#include "tokenizer.h"
#include "common.h"

#include <cassert>
#include <cstdio>
#include <string>

using namespace toyllm;

static Tokenizer make_tk_bytes_only() {
    Tokenizer tk;
    tk.id_to_piece.resize(256);
    for (int i = 0; i < 256; ++i) {
        tk.id_to_piece[i] = std::string(1, static_cast<char>(i));
        tk.piece_to_id[tk.id_to_piece[i]] = i;
    }
    return tk;
}

int main() {
    // 1) Byte-only tokenizer is identity.
    {
        Tokenizer tk = make_tk_bytes_only();
        std::string s = "hello";
        auto ids = tk.encode(s);
        assert(ids.size() == s.size());
        for (size_t i = 0; i < s.size(); ++i) {
            assert(ids[i] == static_cast<unsigned char>(s[i]));
        }
        std::string back = tk.decode(ids);
        assert(back == s);
    }

    // 2) Round-trip through save/load.
    {
        Tokenizer tk = make_tk_bytes_only();
        // Add a fake merge: (104='h', 101='e') -> 256  "he"
        tk.merges.push_back({104, 101});
        tk.id_to_piece.push_back("he");
        tk.piece_to_id["he"] = 256;
        tk.rebuild_pair_index();

        save_tokenizer(tk, "/tmp/toyllm_v.txt", "/tmp/toyllm_m.txt");
        Tokenizer tk2 = load_tokenizer("/tmp/toyllm_v.txt", "/tmp/toyllm_m.txt");
        assert(tk2.vocab_size() == tk.vocab_size());
        assert(tk2.merges.size() == tk.merges.size());
        assert(tk2.merges[0] == tk.merges[0]);

        auto ids = tk2.encode("hello");
        // Must start with the merged "he" token (id 256).
        assert(!ids.empty());
        assert(ids[0] == 256);
        assert(tk2.decode(ids) == "hello");
    }

    std::printf("test_tokenizer: OK\n");
    return 0;
}
