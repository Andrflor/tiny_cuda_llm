#include "model.cuh"
#include "common.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace toyllm {

static constexpr char kMagic[8] = {'T','L','L','M','C','K','P','T'};
static constexpr uint32_t kVersion = 1;

namespace {

void write_exact(FILE* f, const void* p, size_t n, const std::string& path) {
    if (std::fwrite(p, 1, n, f) != n) die("short write on " + path);
}
void read_exact(FILE* f, void* p, size_t n, const std::string& path) {
    if (std::fread(p, 1, n, f) != n) die("short read on " + path);
}

void write_u32(FILE* f, uint32_t v, const std::string& path) {
    write_exact(f, &v, 4, path);
}
uint32_t read_u32(FILE* f, const std::string& path) {
    uint32_t v = 0; read_exact(f, &v, 4, path); return v;
}

void write_tensor(FILE* f, const std::string& name,
                  const DeviceBuffer<float>& buf,
                  const std::vector<uint32_t>& shape,
                  const std::string& path) {
    uint32_t name_len = static_cast<uint32_t>(name.size());
    write_u32(f, name_len, path);
    write_exact(f, name.data(), name_len, path);
    write_u32(f, static_cast<uint32_t>(shape.size()), path);
    for (auto s : shape) write_u32(f, s, path);
    size_t n = 1;
    for (auto s : shape) n *= s;
    if (buf.n != n) die("checkpoint: tensor size mismatch for " + name);
    auto host = buf.to_host();
    write_exact(f, host.data(), n * sizeof(float), path);
}

// Returns the expected element count after reading the header; caller must
// then read n*sizeof(float) bytes and upload to device.
std::vector<float> read_tensor_body(FILE* f, const std::string& expect_name,
                                    const std::vector<uint32_t>& expect_shape,
                                    const std::string& path) {
    uint32_t name_len = read_u32(f, path);
    std::string name(name_len, '\0');
    read_exact(f, name.data(), name_len, path);
    if (name != expect_name) die("checkpoint: expected tensor '" + expect_name + "', got '" + name + "'");
    uint32_t ndim = read_u32(f, path);
    if (ndim != expect_shape.size()) die("checkpoint: ndim mismatch for " + name);
    size_t n = 1;
    for (uint32_t i = 0; i < ndim; ++i) {
        uint32_t s = read_u32(f, path);
        if (s != expect_shape[i]) die("checkpoint: shape mismatch for " + name);
        n *= s;
    }
    std::vector<float> data(n);
    read_exact(f, data.data(), n * sizeof(float), path);
    return data;
}

}  // namespace

void save_checkpoint(const std::string& path, const ModelWeights& w,
                     const ModelConfig& cfg) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) die("cannot open for write: " + path);

    write_exact(f, kMagic, 8, path);
    write_u32(f, kVersion, path);

    // Config.
    write_u32(f, static_cast<uint32_t>(cfg.vocab_size), path);
    write_u32(f, static_cast<uint32_t>(cfg.seq_len),    path);
    write_u32(f, static_cast<uint32_t>(cfg.d_model),    path);
    write_u32(f, static_cast<uint32_t>(cfg.n_layers),   path);
    write_u32(f, static_cast<uint32_t>(cfg.n_heads),    path);
    write_u32(f, static_cast<uint32_t>(cfg.head_dim),   path);
    write_u32(f, static_cast<uint32_t>(cfg.ffn_dim),    path);
    // Floats packaged as raw bytes so endianness matches ingest on same machine.
    write_exact(f, &cfg.rms_eps,   4, path);
    write_exact(f, &cfg.rope_base, 4, path);

    const int V = cfg.vocab_size;
    const int D = cfg.d_model;
    const int F = cfg.ffn_dim;

    // Number of tensors.
    uint32_t n_tensors = 2 + 9 * cfg.n_layers;
    write_u32(f, n_tensors, path);

    write_tensor(f, "E", w.E, {(uint32_t)V, (uint32_t)D}, path);
    write_tensor(f, "g_final", w.g_final, {(uint32_t)D}, path);
    for (int l = 0; l < cfg.n_layers; ++l) {
        const auto& L = w.layers[l];
        auto p = [l](const char* k) { return std::string("layer.") + std::to_string(l) + "." + k; };
        write_tensor(f, p("g_attn"), L.g_attn, {(uint32_t)D}, path);
        write_tensor(f, p("W_Q"),    L.W_Q,   {(uint32_t)D, (uint32_t)D}, path);
        write_tensor(f, p("W_K"),    L.W_K,   {(uint32_t)D, (uint32_t)D}, path);
        write_tensor(f, p("W_V"),    L.W_V,   {(uint32_t)D, (uint32_t)D}, path);
        write_tensor(f, p("W_O"),    L.W_O,   {(uint32_t)D, (uint32_t)D}, path);
        write_tensor(f, p("g_ffn"),  L.g_ffn, {(uint32_t)D}, path);
        write_tensor(f, p("W_up"),   L.W_up,   {(uint32_t)D, (uint32_t)F}, path);
        write_tensor(f, p("W_gate"), L.W_gate, {(uint32_t)D, (uint32_t)F}, path);
        write_tensor(f, p("W_down"), L.W_down, {(uint32_t)F, (uint32_t)D}, path);
    }
    std::fclose(f);
}

void load_checkpoint(const std::string& path, ModelWeights& w, ModelConfig& cfg) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) die("cannot open for read: " + path);

    char magic[8];
    read_exact(f, magic, 8, path);
    if (std::memcmp(magic, kMagic, 8) != 0) die("bad magic in checkpoint: " + path);
    uint32_t ver = read_u32(f, path);
    if (ver != kVersion) die("unsupported checkpoint version: " + path);

    cfg.vocab_size = static_cast<int>(read_u32(f, path));
    cfg.seq_len    = static_cast<int>(read_u32(f, path));
    cfg.d_model    = static_cast<int>(read_u32(f, path));
    cfg.n_layers   = static_cast<int>(read_u32(f, path));
    cfg.n_heads    = static_cast<int>(read_u32(f, path));
    cfg.head_dim   = static_cast<int>(read_u32(f, path));
    cfg.ffn_dim    = static_cast<int>(read_u32(f, path));
    read_exact(f, &cfg.rms_eps,   4, path);
    read_exact(f, &cfg.rope_base, 4, path);

    uint32_t n_tensors = read_u32(f, path);
    uint32_t expect = 2u + 9u * static_cast<uint32_t>(cfg.n_layers);
    if (n_tensors != expect) die("checkpoint: unexpected n_tensors");

    const int V = cfg.vocab_size;
    const int D = cfg.d_model;
    const int F = cfg.ffn_dim;

    auto upload = [&](DeviceBuffer<float>& buf, const std::vector<float>& host) {
        buf.allocate(host.size());
        buf.copy_from_host(host);
    };

    {
        auto e = read_tensor_body(f, "E", {(uint32_t)V, (uint32_t)D}, path);
        upload(w.E, e);
    }
    {
        auto gf = read_tensor_body(f, "g_final", {(uint32_t)D}, path);
        upload(w.g_final, gf);
    }
    w.layers.resize(cfg.n_layers);
    for (int l = 0; l < cfg.n_layers; ++l) {
        auto& L = w.layers[l];
        auto p = [l](const char* k) { return std::string("layer.") + std::to_string(l) + "." + k; };
        upload(L.g_attn, read_tensor_body(f, p("g_attn"), {(uint32_t)D}, path));
        upload(L.W_Q,    read_tensor_body(f, p("W_Q"),    {(uint32_t)D, (uint32_t)D}, path));
        upload(L.W_K,    read_tensor_body(f, p("W_K"),    {(uint32_t)D, (uint32_t)D}, path));
        upload(L.W_V,    read_tensor_body(f, p("W_V"),    {(uint32_t)D, (uint32_t)D}, path));
        upload(L.W_O,    read_tensor_body(f, p("W_O"),    {(uint32_t)D, (uint32_t)D}, path));
        upload(L.g_ffn,  read_tensor_body(f, p("g_ffn"),  {(uint32_t)D}, path));
        upload(L.W_up,   read_tensor_body(f, p("W_up"),   {(uint32_t)D, (uint32_t)F}, path));
        upload(L.W_gate, read_tensor_body(f, p("W_gate"), {(uint32_t)D, (uint32_t)F}, path));
        upload(L.W_down, read_tensor_body(f, p("W_down"), {(uint32_t)F, (uint32_t)D}, path));
    }
    std::fclose(f);
}

}  // namespace toyllm
