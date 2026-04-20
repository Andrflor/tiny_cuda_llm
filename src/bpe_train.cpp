#include "tokenizer.h"
#include "common.h"

#include <algorithm>
#include <cstdio>
#include <unordered_map>

namespace toyllm {

// Toy BPE trainer.
//
// Algorithm (Sennrich et al., simplified):
//   1. Seed vocab = 256 raw bytes.
//   2. Represent the corpus as a single sequence of ids, broken by whitespace
//      at line boundaries only (we keep it dead-simple: no per-word pre-split,
//      just one big sequence per line to avoid exploding counts).
//   3. Repeat num_merges times:
//        a. count all adjacent (a, b) pairs
//        b. take the most frequent one
//        c. replace every occurrence of (a, b) by new_id = 256 + rank
//        d. append (a, b) to merges, append piece(a) + piece(b) to vocab.

Tokenizer train_bpe(const std::string& corpus, int num_merges) {
    Tokenizer tk;
    tk.id_to_piece.resize(256);
    for (int i = 0; i < 256; ++i) {
        tk.id_to_piece[i] = std::string(1, static_cast<char>(i));
        tk.piece_to_id[tk.id_to_piece[i]] = i;
    }

    // Encode corpus into a vector of "words" (each line is one sequence) to
    // keep merges local and cheap.
    std::vector<std::vector<int>> seqs;
    seqs.reserve(64);
    {
        std::vector<int> cur;
        cur.reserve(128);
        for (unsigned char c : corpus) {
            if (c == '\n') {
                if (!cur.empty()) { seqs.push_back(std::move(cur)); cur.clear(); }
            } else {
                cur.push_back(static_cast<int>(c));
            }
        }
        if (!cur.empty()) seqs.push_back(std::move(cur));
    }

    for (int r = 0; r < num_merges; ++r) {
        // 1. count pairs
        std::unordered_map<uint64_t, long long> pair_counts;
        pair_counts.reserve(1 << 14);
        for (const auto& s : seqs) {
            for (size_t i = 0; i + 1 < s.size(); ++i) {
                uint64_t key = Tokenizer::pack_pair(s[i], s[i + 1]);
                pair_counts[key] += 1;
            }
        }
        if (pair_counts.empty()) break;

        // 2. argmax
        uint64_t best_key = 0;
        long long best_count = -1;
        for (const auto& kv : pair_counts) {
            if (kv.second > best_count ||
                (kv.second == best_count && kv.first < best_key)) {
                best_count = kv.second;
                best_key = kv.first;
            }
        }
        if (best_count < 2) break;  // nothing worth merging

        int a = static_cast<int>(best_key >> 32);
        int b = static_cast<int>(best_key & 0xFFFFFFFFu);
        int new_id = 256 + static_cast<int>(tk.merges.size());

        tk.merges.push_back({a, b});
        std::string piece = tk.id_to_piece[a] + tk.id_to_piece[b];
        tk.id_to_piece.push_back(piece);
        tk.piece_to_id[piece] = new_id;

        // 3. apply merge in every sequence (left-to-right, non-overlapping)
        for (auto& s : seqs) {
            if (s.size() < 2) continue;
            std::vector<int> out;
            out.reserve(s.size());
            size_t i = 0;
            while (i < s.size()) {
                if (i + 1 < s.size() && s[i] == a && s[i + 1] == b) {
                    out.push_back(new_id);
                    i += 2;
                } else {
                    out.push_back(s[i]);
                    i += 1;
                }
            }
            s.swap(out);
        }
    }

    tk.rebuild_pair_index();
    return tk;
}

}  // namespace toyllm
