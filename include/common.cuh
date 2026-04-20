#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                \
                         #expr, __FILE__, __LINE__, cudaGetErrorString(_err));  \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

#define CUDA_CHECK_LAST()                                                       \
    do {                                                                        \
        cudaError_t _err = cudaGetLastError();                                  \
        if (_err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA kernel launch error at %s:%d: %s\n",     \
                         __FILE__, __LINE__, cudaGetErrorString(_err));         \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)
