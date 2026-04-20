#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace toyllm {

inline void die(const std::string& msg) {
    std::fprintf(stderr, "fatal: %s\n", msg.c_str());
    std::exit(1);
}

inline std::string read_file(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) die("cannot open file: " + path);
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::string s;
    s.resize(static_cast<size_t>(n));
    if (n > 0 && std::fread(&s[0], 1, n, f) != static_cast<size_t>(n)) {
        std::fclose(f);
        die("short read on " + path);
    }
    std::fclose(f);
    return s;
}

// Simple deterministic Xorshift64 RNG for reproducible weight init.
struct Xorshift64 {
    uint64_t state;
    explicit Xorshift64(uint64_t seed) : state(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    uint64_t next_u64() {
        uint64_t x = state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state = x;
        return x;
    }
    // Uniform float in [-1, 1).
    float next_uniform() {
        const uint64_t u = next_u64();
        // 24 random bits → [0,1), then rescale to [-1, 1).
        const float unit = static_cast<float>((u >> 40) & 0xFFFFFFULL) / 16777216.0f;
        return unit * 2.0f - 1.0f;
    }
};

}  // namespace toyllm
