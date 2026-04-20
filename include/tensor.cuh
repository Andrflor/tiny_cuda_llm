#pragma once

#include "common.cuh"
#include <cstddef>
#include <vector>

namespace toyllm {

// RAII wrapper for a device allocation. Non-copyable, movable.
template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    size_t n = 0;

    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { allocate(count); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr(o.ptr), n(o.n) {
        o.ptr = nullptr; o.n = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            free();
            ptr = o.ptr; n = o.n;
            o.ptr = nullptr; o.n = 0;
        }
        return *this;
    }
    ~DeviceBuffer() { free(); }

    void allocate(size_t count) {
        free();
        n = count;
        if (count == 0) return;
        CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    }
    void free() {
        if (ptr) { cudaFree(ptr); ptr = nullptr; }
        n = 0;
    }

    void copy_from_host(const T* host, size_t count) {
        CUDA_CHECK(cudaMemcpy(ptr, host, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    void copy_from_host(const std::vector<T>& v) {
        copy_from_host(v.data(), v.size());
    }
    void copy_to_host(T* host, size_t count) const {
        CUDA_CHECK(cudaMemcpy(host, ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    std::vector<T> to_host() const {
        std::vector<T> v(n);
        if (n) copy_to_host(v.data(), n);
        return v;
    }
    void zero() {
        if (n) CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(T)));
    }
};

}  // namespace toyllm
