#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace toyllm {

// Toy char-level BPE tokenizer.
//
// Vocabulary layout:
//   - ids 0..255 correspond to raw bytes (identity)
//   - ids 256..255+M correspond to learned merges, each represented by the
//     concatenation of the two children tokens (as an std::string over bytes)
// Encoding: split string into bytes, then greedily apply learned merges in the
// order they were learned (earliest merge = highest priority).

struct Tokenizer {
    // id -> piece bytes (arbitrary binary string)
    std::vector<std::string> id_to_piece;
    // piece bytes -> id
    std::unordered_map<std::string, int> piece_to_id;
    // Ordered list of merges: (left_id, right_id) -> new_id.
    // Stored by rank; rank = new_id - 256.
    std::vector<std::pair<int, int>> merges;
    // Fast lookup (left_id, right_id) -> rank
    std::unordered_map<uint64_t, int> pair_rank;

    int vocab_size() const { return static_cast<int>(id_to_piece.size()); }

    // Encode a UTF-8 / raw byte string into token ids.
    std::vector<int> encode(const std::string& text) const;

    // Decode token ids back to a raw byte string.
    std::string decode(const std::vector<int>& ids) const;

    // Rebuild pair_rank from merges (needed after loading).
    void rebuild_pair_index();

    static uint64_t pack_pair(int a, int b) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32)
             |  static_cast<uint64_t>(static_cast<uint32_t>(b));
    }
};

// Train a BPE tokenizer on raw text. num_merges = number of merges to learn
// (final vocab size = 256 + num_merges).
Tokenizer train_bpe(const std::string& corpus, int num_merges);

// Save / load to disk.
// vocab file format (one entry per line):
//   <id> <hex-encoded piece bytes>
// merges file format (one entry per line):
//   <rank> <left_id> <right_id>
void save_tokenizer(const Tokenizer& tk,
                    const std::string& vocab_path,
                    const std::string& merges_path);
Tokenizer load_tokenizer(const std::string& vocab_path,
                         const std::string& merges_path);

}  // namespace toyllm
