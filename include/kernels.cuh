#pragma once

namespace toyllm {

// ============================================================================
// FORWARD KERNELS
// ============================================================================

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

// --- RoPE (rotary position embedding) ---
// Applies in-place rotation to X : [T, H, dh] stored contiguous as [T, H*dh].
// Pairs (x_{2k}, x_{2k+1}) are rotated by t · base^(-2k/dh). Requires dh even.
void launch_rope(float* X, int T, int H, int head_dim, float base);

// --- Causal masked attention ---
// Q, K, V : [T, H, d_h] stored contiguous as [T, H*d_h] (= d_model).
// out     : [T, H, d_h] same layout.
// Computes for each head independently:
//   S = Q K^T / sqrt(d_h) + M   (M causal)
//   A = softmax(S)
//   C = A V
// Uses workspace scores : [H, T, T]. Scores stores the softmax probs A on exit.
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

// ============================================================================
// BACKWARD KERNELS
// ============================================================================

// --- Matmul backward ---
// Given C = A @ B with A[M,K], B[K,N], C[M,N], and dC[M,N]:
//   dA = dC @ Bᵀ     [M, K]
//   dB = Aᵀ @ dC     [K, N]
// Computed as two standard matmuls against transposed operands.
void launch_matmul_backward(const float* dC, const float* A, const float* B,
                            float* dA, float* dB,
                            int M, int K, int N);

// --- RMSNorm backward ---
// Forward: y = x * inv * g,  inv = 1/sqrt(mean(x²) + eps)
// Inputs:  dy [T,D], x [T,D], g [D], eps
// Outputs: dx [T,D], dg [D] (accumulated, set caller to zero first)
void launch_rmsnorm_backward(const float* dy, const float* x, const float* g,
                             float* dx, float* dg,
                             int T, int D, float eps);

// --- Softmax backward (row-wise) ---
// Given y = softmax(x) and dy, computes dx[i] = y[i] * (dy[i] - sum_j(dy[j]*y[j]))
// y, dy, dx : [rows, cols]
void launch_softmax_rows_backward(const float* y, const float* dy, float* dx,
                                  int rows, int cols);

// --- RoPE backward ---
// Inverse rotation by -phi. In-place on dX : [T, H, dh].
void launch_rope_backward(float* dX, int T, int H, int head_dim, float base);

// --- Causal MHA backward ---
// Inputs:
//   Q, K, V : [T, H, dh]  (post-RoPE Q/K same as forward)
//   A       : [H, T, T]   attention probs (stored by forward in `scores`)
//   dout    : [T, H, dh]
// Outputs (must be zeroed by caller):
//   dQ, dK, dV : [T, H, dh]
// Workspace:
//   dS : [H, T, T]
void launch_causal_mha_backward(const float* Q, const float* K, const float* V,
                                const float* A, const float* dout,
                                float* dS,
                                float* dQ, float* dK, float* dV,
                                int T, int H, int head_dim);

// --- SwiGLU backward ---
// Inputs saved from forward:
//   Z [T,D], W_up [D,F], W_gate [D,F], W_down [F,D]
//   hidden_up [T,F] = Z @ W_up
//   hidden_gate [T,F] = Z @ W_gate
//   hidden_mul [T,F]  = silu(hidden_up) * hidden_gate
//   dout [T,D]
// Outputs (must be zeroed by caller for weight grads; dZ is overwritten):
//   dZ [T,D], dW_up [D,F], dW_gate [D,F], dW_down [F,D]
// Workspace scratch: dmul [T,F], dup [T,F], dgate [T,F]
void launch_swiglu_backward(const float* Z,
                            const float* W_up, const float* W_gate, const float* W_down,
                            const float* hidden_up, const float* hidden_gate,
                            const float* hidden_mul,
                            const float* dout,
                            float* dmul, float* dup, float* dgate,
                            float* dZ, float* dW_up, float* dW_gate, float* dW_down,
                            int T, int D, int F);

// --- Embedding backward ---
// Scatter-add: for each t, dE[ids[t], d] += dout[t, d]
// dE must be zero-initialized by the caller (or pre-populated with tied-logits grad).
void launch_embedding_backward(const int* ids, const float* dout, float* dE,
                               int T, int V, int D);

// --- Logits-tied backward ---
// Forward: logits[t, v] = sum_d X[t, d] * E[v, d]
// Given dlogits [T, V], X [T, D], E [V, D]:
//   dX[t, d] = sum_v dlogits[t, v] * E[v, d]        (matmul dlogits @ E)
//   dE[v, d] += sum_t dlogits[t, v] * X[t, d]       (accumulated — tied)
// dE must be pre-zeroed or pre-initialized.
void launch_logits_tied_backward(const float* dlogits, const float* X, const float* E,
                                 float* dX, float* dE,
                                 int T, int D, int V);

// ============================================================================
// LOSS & OPTIMIZER
// ============================================================================

// --- Cross-entropy loss (fused softmax + NLL) ---
// logits : [T, V], targets : [T] with -1 meaning "ignored"
// Output: scalar `loss_out` (device float) = mean NLL over valid positions
//         dlogits [T, V] = (softmax(logits) - onehot(target)) / n_valid,
//                          zeros for ignored rows.
void launch_cross_entropy(const float* logits, const int* targets,
                          float* loss_out, float* dlogits,
                          int T, int V);

// --- AdamW step ---
// Updates `param` in-place using grad, first/second moments m, v.
// `step` is 1-indexed. Applies decoupled weight decay (wd * param) before the
// Adam update so that bias-corrected moments are unchanged by wd.
void launch_adamw_step(float* param, const float* grad,
                       float* m, float* v,
                       int n, int step,
                       float lr, float beta1, float beta2,
                       float eps, float weight_decay);

// --- Fill / scale helpers ---
void launch_fill(float* x, float value, int n);
void launch_axpy(float alpha, const float* x, float* y, int n);  // y += alpha * x

}  // namespace toyllm
