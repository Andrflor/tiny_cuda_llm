# toy_llm_cuda

Un mini LLM **decoder-only** écrit from-scratch en **C++ / CUDA**, sans aucune
dépendance type PyTorch. Le but est double :

1. **Bootstrap (v0, ce repo)** — un *toy LLM* déjà fonctionnel de bout en bout :
   tokenizer BPE entraîné sur un corpus, forward pass Transformer sur GPU,
   génération greedy. **Inference-first**, pas encore d'entraînement.
2. **Target (v1)** — évoluer vers un *tiny SOTA LLM* d'environ **100M params**
   avec GQA + RoPE + attention hybride local/global + FFN mixte dense/MoE.

Tout est écrit à la main. La différenciation (backward), quand elle sera
ajoutée, le sera **à la main** également : pas d'autograd, pas de framework.

---

## État actuel (v0 — bootstrap)

Ce qui tourne **aujourd'hui** :

- Tokenizer BPE trainable (CPU, C++) : train / save / load / encode / decode
- Forward pass CUDA d'un tiny decoder-only Transformer :
  - token embeddings
  - RMSNorm pre-norm
  - causal multi-head self-attention (MHA dense, pas de GQA/RoPE/KV cache)
  - SwiGLU FFN
  - résidus
  - RMSNorm finale
  - projection de sortie en **weight tying** avec l'embedding
- Génération greedy (argmax) sur CPU avec recomputation full-context
- Float32 uniquement

### Config v0

```
vocab_size  = 256 + nb_merges (typiquement ~512)
seq_len     = 32
d_model     = 128
n_layers    = 2
n_heads     = 4
head_dim    = 32        (d_model = n_heads * head_dim)
ffn_dim     = 256
dtype       = float32
```

---

## Équations (v0)

Pour chaque couche ℓ, avec X^(ℓ) ∈ R^(T×d) :

```
U^(ℓ)   = RMSNorm(X^(ℓ))
Q, K, V = U^(ℓ) W_Q, U^(ℓ) W_K, U^(ℓ) W_V        # [T, d]
```

Pour chaque tête h (split de Q,K,V en H blocs de taille d_h) :

```
S_h   = Q_h K_hᵀ / sqrt(d_h) + M      avec M_ij = 0 si j≤i, -∞ sinon
A_h   = softmax(S_h)
C_h   = A_h V_h
C     = concat(C_1, ..., C_H)
O     = C W_O
R^(ℓ) = X^(ℓ) + O
```

Puis SwiGLU :

```
Z^(ℓ)     = RMSNorm(R^(ℓ))
F(z)      = W_down ( SiLU(z W_up) ⊙ (z W_gate) )
X^(ℓ+1)   = R^(ℓ) + F(Z^(ℓ))
```

Logits finaux (tied embeddings) :

```
Y      = RMSNorm(X^(L))
logits = Y · Eᵀ        # [T, V]
```

RMSNorm :  `RMSNorm(x)_i = x_i / sqrt(mean(x²) + ε) * g_i`

SiLU :     `SiLU(x) = x · σ(x)`

---

## Architecture cible (v1 — tiny SOTA ~100M)

La v1 est la cible **finale** du repo. Elle n'est **pas** implémentée ici.
Elle guide les choix de design pour que l'on puisse y migrer sans tout casser.

### Embeddings

- V ≈ 32000, d = 576, tying sortie/embedding : `u_i = z_i^(L) · Eᵀ`

### Attention : GQA + RoPE

- 14 couches
- 12 query heads, 3 KV heads (groupes de 4), head_dim = 48
- RoPE appliqué sur Q et K
- `S_h = (Q̃_h · K̃_g(h)ᵀ) / sqrt(48) + M`

### Attention hybride local / global

Pattern par couche, sur 14 couches : 10 locales + 4 globales.

```
[L, L, L, G, L, L, L, G, L, L, G, L, L, G]
```

Global aux couches ℓ ∈ {4, 8, 11, 14}. Sliding window w = 512 pour les locales.

### FFN : dense + MoE

- d_ff = 1536, SwiGLU
- 10 FFN denses + 4 FFN MoE (4 experts, top-2 routing)
- MoE sur ℓ ∈ {7, 9, 11, 13}

### Répartition finale v1

```
 1 : Local  + Dense
 2 : Local  + Dense
 3 : Local  + Dense
 4 : Global + Dense
 5 : Local  + Dense
 6 : Local  + Dense
 7 : Local  + MoE
 8 : Global + Dense
 9 : Local  + MoE
10 : Local  + Dense
11 : Global + MoE
12 : Local  + Dense
13 : Local  + MoE
14 : Global + Dense
```

### Hyperparamètres annexes v1

- RMSNorm pre-norm, no bias partout
- dropout = 0 en prétraining
- router z-loss + load-balancing loss, capacity factor 1.25–1.5
- RoPE base classique
- FlashAttention possible plus tard, orthogonal à l'archi

Loss totale :

```
L_tot = L_LM + λ_bal · Σ_{ℓ ∈ L_MoE} L_bal^(ℓ)
```

### Roadmap v0 → v1

1. **v0 (ce bootstrap)** — dense MHA + SwiGLU, MNIST-level du NLP.
2. Ajouter KV cache (inférence plus rapide).
3. Remplacer MHA par **GQA** + **RoPE**.
4. Ajouter l'**attention locale** (sliding window) et le pattern hybride.
5. Implémenter le **backward manuel** (autograd à la main) + SGD/AdamW.
6. Scaler à d=576, 14 layers, V=32k.
7. Ajouter les couches **MoE** (4 experts, top-2 + balancing loss).
8. Entraîner sur un corpus réel (WikiText / FineWeb échantillon).

---

## Structure du repo

```
toy_llm_cuda/
  Makefile
  README.md
  CLAUDE.md
  data/
    corpus.txt          # petit corpus pour entraîner le BPE
    prompt.txt          # prompt d'inférence
  include/
    config.h            # hyperparams v0
    common.h            # utils CPU
    common.cuh          # utils CUDA (CUDA_CHECK, etc.)
    tokenizer.h
    tensor.cuh
    model.cuh
    kernels.cuh
  src/
    main.cpp            # CLI end-to-end
    tokenizer.cpp       # encode/decode/save/load
    bpe_train.cpp       # train BPE
    tensor.cu           # alloc/copy device <-> host
    embedding.cu
    matmul.cu           # GEMM naïf row-major
    rmsnorm.cu
    softmax.cu
    attention.cu        # MHA causale dense
    ffn.cu              # SwiGLU
    model.cu            # assemblage forward
  tests/
    test_tokenizer.cpp
    test_bpe_train.cpp
    test_matmul.cu
    test_rmsnorm.cu
    test_attention.cu
    test_model_smoke.cu
```

---

## Build

Prérequis :

- `nvcc` (CUDA ≥ 11)
- `g++` ≥ 9 (C++17)
- GPU NVIDIA dispo pour l'exécution (la compil seule ne l'exige pas)

```
make             # compile tout : toy_llm + tests
make tokenizer   # compile uniquement la partie CPU (pas besoin de CUDA)
make tests       # compile les tests
make clean
```

Variables Make utiles :

- `SM=86` pour cibler une archi GPU spécifique (défaut : sm_80)
- `DEBUG=1` pour `-g -O0`

---

## Utilisation end-to-end

1. Entraîner le tokenizer BPE sur `data/corpus.txt` :

```
./toy_llm train-bpe data/corpus.txt tokenizer.vocab tokenizer.merges 256
```

Arguments : `<corpus> <vocab-out> <merges-out> <nb-merges>`.

2. Encoder un prompt, faire tourner le forward, générer :

```
./toy_llm generate tokenizer.vocab tokenizer.merges data/prompt.txt 16
```

Arguments : `<vocab> <merges> <prompt-file> <nb-tokens-à-générer>`.

**Note** : en v0 les poids du modèle sont initialisés aléatoirement
(seed fixe) à chaque run. La sortie n'a donc aucun sens linguistique ;
ce qu'on vérifie, c'est que **le pipeline tokenizer → CUDA forward → argmax →
detokenizer tourne de bout en bout**.

### Exemples détaillés

```
# Tokenizer seul
./toy_llm encode tokenizer.vocab tokenizer.merges "hello world"
./toy_llm decode tokenizer.vocab tokenizer.merges 1 2 3 4

# Tests unitaires
./tests/test_tokenizer
./tests/test_bpe_train
./tests/test_matmul
./tests/test_rmsnorm
./tests/test_attention
./tests/test_model_smoke
```

---

## Corpus

`data/corpus.txt` contient un petit échantillon suffisant pour apprendre
quelques merges BPE et démontrer le pipeline. Pour aller plus loin, tu peux
remplacer ce fichier par :

- WikiText-103 (brut)
- FineWeb échantillon
- TinyStories
- ton propre corpus

Pour l'entraînement réel du modèle (v1), il faudra un corpus bien plus gros
(des Go de texte) et un pipeline de tokenisation streamée. Ce n'est **pas**
dans le scope de la v0.

---

## Ce qui n'est **pas** dans la v0

Explicitement, et volontairement :

- pas de backward / autograd
- pas d'optimiseur
- pas de GQA, pas de RoPE, pas de KV cache
- pas de MoE, pas d'attention locale/globale
- pas de FlashAttention
- pas de float16 / bfloat16

Tout cela arrive dans la roadmap v1 ci-dessus.
