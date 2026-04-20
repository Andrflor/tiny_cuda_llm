#include "tokenizer.h"

#include <cassert>
#include <cstdio>
#include <string>

using namespace toyllm;

int main() {
    // A corpus where "ab" is clearly the most frequent pair.
    std::string corpus =
        "ab ab ab ab ab ab ab ab ab ab\n"
        "ab ab ab ab ab ab ab ab ab ab\n"
        "xy xy\n";

    Tokenizer tk = train_bpe(corpus, 5);
    assert(tk.vocab_size() > 256);
    assert(tk.merges.size() > 0);

    // The very first merge should be ('a','b') = (97, 98).
    assert(tk.merges[0].first == 97);
    assert(tk.merges[0].second == 98);

    // Encoding "ab" should collapse to a single token after training.
    auto ids = tk.encode("ab");
    assert(ids.size() == 1);
    assert(ids[0] == 256);  // first merge

    // Round-trip.
    std::string back = tk.decode(ids);
    assert(back == "ab");

    std::printf("test_bpe_train: OK (merges=%zu, vocab=%d)\n",
                tk.merges.size(), tk.vocab_size());
    return 0;
}
