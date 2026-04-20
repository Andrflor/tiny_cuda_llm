#include "tokenizer.h"
#include "common.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

namespace toyllm {

static char nibble_to_hex(int n) {
    return static_cast<char>(n < 10 ? '0' + n : 'a' + (n - 10));
}
static int hex_to_nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}

static std::string bytes_to_hex(const std::string& s) {
    std::string out;
    out.resize(s.size() * 2);
    for (size_t i = 0; i < s.size(); ++i) {
        unsigned char b = static_cast<unsigned char>(s[i]);
        out[2 * i]     = nibble_to_hex(b >> 4);
        out[2 * i + 1] = nibble_to_hex(b & 0xF);
    }
    return out;
}
static std::string hex_to_bytes(const std::string& h) {
    if (h.size() % 2 != 0) die("odd-length hex");
    std::string out;
    out.resize(h.size() / 2);
    for (size_t i = 0; i < out.size(); ++i) {
        int hi = hex_to_nibble(h[2 * i]);
        int lo = hex_to_nibble(h[2 * i + 1]);
        if (hi < 0 || lo < 0) die("invalid hex char");
        out[i] = static_cast<char>((hi << 4) | lo);
    }
    return out;
}

void Tokenizer::rebuild_pair_index() {
    pair_rank.clear();
    pair_rank.reserve(merges.size() * 2);
    for (size_t r = 0; r < merges.size(); ++r) {
        pair_rank[pack_pair(merges[r].first, merges[r].second)] = static_cast<int>(r);
    }
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    // Start as raw bytes.
    std::vector<int> ids;
    ids.reserve(text.size());
    for (unsigned char c : text) ids.push_back(static_cast<int>(c));

    if (merges.empty() || ids.size() < 2) return ids;

    // Iteratively find the adjacent pair with the lowest rank and merge it.
    // Naive O(N·M) but fine at toy scale.
    while (ids.size() >= 2) {
        int best_rank = -1;
        size_t best_pos = 0;
        for (size_t i = 0; i + 1 < ids.size(); ++i) {
            auto it = pair_rank.find(pack_pair(ids[i], ids[i + 1]));
            if (it != pair_rank.end()) {
                if (best_rank < 0 || it->second < best_rank) {
                    best_rank = it->second;
                    best_pos = i;
                    if (best_rank == 0) break;  // can't beat that
                }
            }
        }
        if (best_rank < 0) break;
        const int new_id = 256 + best_rank;
        ids[best_pos] = new_id;
        ids.erase(ids.begin() + best_pos + 1);
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    out.reserve(ids.size() * 2);
    for (int id : ids) {
        if (id < 0 || id >= static_cast<int>(id_to_piece.size())) {
            die("decode: id out of range");
        }
        out += id_to_piece[id];
    }
    return out;
}

void save_tokenizer(const Tokenizer& tk,
                    const std::string& vocab_path,
                    const std::string& merges_path) {
    {
        std::ofstream f(vocab_path);
        if (!f) die("cannot write " + vocab_path);
        f << tk.id_to_piece.size() << "\n";
        for (size_t i = 0; i < tk.id_to_piece.size(); ++i) {
            f << i << " " << bytes_to_hex(tk.id_to_piece[i]) << "\n";
        }
    }
    {
        std::ofstream f(merges_path);
        if (!f) die("cannot write " + merges_path);
        f << tk.merges.size() << "\n";
        for (size_t r = 0; r < tk.merges.size(); ++r) {
            f << r << " " << tk.merges[r].first << " " << tk.merges[r].second << "\n";
        }
    }
}

Tokenizer load_tokenizer(const std::string& vocab_path,
                         const std::string& merges_path) {
    Tokenizer tk;
    {
        std::ifstream f(vocab_path);
        if (!f) die("cannot read " + vocab_path);
        size_t n; f >> n;
        tk.id_to_piece.assign(n, "");
        for (size_t k = 0; k < n; ++k) {
            size_t id; std::string hex;
            f >> id >> hex;
            if (id != k) die("vocab not in order");
            std::string piece = (hex == "") ? std::string() : hex_to_bytes(hex);
            tk.id_to_piece[id] = piece;
            tk.piece_to_id[piece] = static_cast<int>(id);
        }
    }
    {
        std::ifstream f(merges_path);
        if (!f) die("cannot read " + merges_path);
        size_t m; f >> m;
        tk.merges.resize(m);
        for (size_t r = 0; r < m; ++r) {
            size_t rk; int a, b;
            f >> rk >> a >> b;
            if (rk != r) die("merges not in order");
            tk.merges[r] = {a, b};
        }
    }
    tk.rebuild_pair_index();
    return tk;
}

}  // namespace toyllm
