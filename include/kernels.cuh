#pragma once

namespace toyllm {

// --- Embedding ---
// Gather rows: out[t, d] = E[ids[t], d]
// ids : [T], E : [V, D], out : [T, D]
void launch_embedding_lookup(const int* ids, const float* E, float* out,
                             int T, int V, int D);

// Logits via tied embeddings: logits[t, v] = sum_d X[t, d] * E[v, d]
// X : [T, D], E : [V, D], logits : [T, V]
void launch_logits_tied(const float* X, const float* E, float* logits,
                        int T, int D, int V);

// --- Matmul (naive row-major GEMM) ---
// C[m, n] = A[m, k] @ B[k, n]
void launch_matmul(const float* A, const float* B, float* C,
                   int M, int K, int N);

// --- RMSNorm ---
// out[t, d] = x[t, d] / sqrt(mean(x²) + eps) * g[d]
// x, out : [T, D], g : [D]
void launch_rmsnorm(const float* x, const float* g, float* out,
                    int T, int D, float eps);

// --- Softmax (row-wise on last dim) ---
// x : [rows, cols], in-place or out-of-place, numerically stable (minus max).
void launch_softmax_rows(const float* x, float* out, int rows, int cols);

// --- Causal masked attention ---
// Q, K, V : [T, H, d_h] stored contiguous as [T, H*d_h] (= d_model).
// out     : [T, H, d_h] same layout.
// Computes for each head independently:
//   S = Q K^T / sqrt(d_h) + M   (M causal)
//   A = softmax(S)
//   C = A V
// Uses workspace scores : [H, T, T].
void launch_causal_mha(const float* Q, const float* K, const float* V,
                       float* scores, float* out,
                       int T, int H, int head_dim);

// --- SwiGLU FFN ---
// Z : [T, D], W_up : [D, F], W_gate : [D, F], W_down : [F, D]
// out : [T, D]
// hidden_silu : [T, F], hidden_gate : [T, F], hidden_mul : [T, F] workspaces
void launch_swiglu(const float* Z,
                   const float* W_up, const float* W_gate, const float* W_down,
                   float* hidden_up, float* hidden_gate, float* hidden_mul,
                   float* out,
                   int T, int D, int F);

// --- Residual add ---
// out[i] = a[i] + b[i]   (in-place version: launch_residual_add(x, y, x, n))
void launch_add(const float* a, const float* b, float* out, int n);

}  // namespace toyllm
