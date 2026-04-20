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
  - causal multi-head self-attention avec RoPE
  - SwiGLU FFN
  - résidus
  - RMSNorm finale
  - projection de sortie en **weight tying** avec l'embedding
- **Backward manuel** kernel par kernel (matmul, rmsnorm, softmax, rope,
  attention, swiglu, embedding, logits tied, cross-entropy)
- **AdamW** avec weight decay découplé
- **Training** depuis un corpus + save/load de checkpoints (.bin custom,
  sans dépendance)
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

Pour chaque couche $\ell$, avec $X^{(\ell)} \in \mathbb{R}^{T \times d}$ :

$$
U^{(\ell)} = \mathrm{RMSNorm}(X^{(\ell)})
$$

$$
Q = U^{(\ell)} W_Q, \quad K = U^{(\ell)} W_K, \quad V = U^{(\ell)} W_V \in \mathbb{R}^{T \times d}
$$

Pour chaque tête $h$ (split de $Q, K, V$ en $H$ blocs de taille $d_h$), avec le mask causal

$$
M_{ij} = \begin{cases} 0 & \text{si } j \le i \\ -\infty & \text{sinon} \end{cases}
$$

on a

$$
S_h = \frac{Q_h K_h^\top}{\sqrt{d_h}} + M, \quad A_h = \mathrm{softmax}(S_h), \quad C_h = A_h V_h
$$

$$
C = \mathrm{Concat}(C_1, \dots, C_H), \quad O = C W_O, \quad R^{(\ell)} = X^{(\ell)} + O
$$

Puis SwiGLU :

$$
Z^{(\ell)} = \mathrm{RMSNorm}(R^{(\ell)})
$$

$$
F^{(\ell)}(z) = W_{\text{down}}^{(\ell)} \Big( \mathrm{SiLU}(z W_{\text{up}}^{(\ell)}) \odot (z W_{\text{gate}}^{(\ell)}) \Big)
$$

$$
X^{(\ell+1)} = R^{(\ell)} + F^{(\ell)}(Z^{(\ell)})
$$

Logits finaux (tied embeddings) :

$$
Y = \mathrm{RMSNorm}(X^{(L)}), \quad \mathrm{logits} = Y E^\top \in \mathbb{R}^{T \times V}
$$

Définitions auxiliaires :

$$
\mathrm{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\tfrac{1}{d}\sum_{j=1}^{d} x_j^2 + \varepsilon}} \cdot g_i, \qquad \mathrm{SiLU}(x) = x \cdot \sigma(x)
$$

---

## Architecture cible (v1 — tiny SOTA ~100M)

La v1 est la cible **finale** du repo. Elle n'est **pas** implémentée ici.
Elle guide les choix de design pour que l'on puisse y migrer sans tout casser.

### Embeddings

Matrice d'embedding $E \in \mathbb{R}^{V \times d}$ avec $V \approx 32000$ et $d = 576$. Pour une séquence de tokens $(t_0, \dots, t_n)$ :

$$
X^{(0)} = \begin{bmatrix} e_{t_0} \\ \vdots \\ e_{t_n} \end{bmatrix} \in \mathbb{R}^{(n+1) \times 576}
$$

Avec tying en sortie : $u_i = z_i^{(L)} E^\top$.

### Attention : GQA + RoPE

14 couches, 12 query heads, 3 KV heads (groupes de 4), $d_h = 48$. À chaque couche $\ell$ :

$$
U^{(\ell)} = \mathrm{RMSNorm}(X^{(\ell)})
$$

Projections (query par tête, key/value par groupe) :

$$
Q_h^{(\ell)} = U^{(\ell)} W_{Q,h}^{(\ell)}, \quad h = 1, \dots, 12
$$

$$
K_g^{(\ell)} = U^{(\ell)} W_{K,g}^{(\ell)}, \quad V_g^{(\ell)} = U^{(\ell)} W_{V,g}^{(\ell)}, \quad g = 1, 2, 3
$$

RoPE sur $Q, K$ :

$$
\tilde{Q}_h^{(\ell)} = \mathrm{RoPE}(Q_h^{(\ell)}), \quad \tilde{K}_g^{(\ell)} = \mathrm{RoPE}(K_g^{(\ell)})
$$

Pour chaque tête $h$, avec $g(h) \in \{1, 2, 3\}$ son groupe :

$$
S_h^{(\ell)} = \frac{\tilde{Q}_h^{(\ell)} \tilde{K}_{g(h)}^{(\ell) \top}}{\sqrt{48}} + M_h^{(\ell)}, \quad A_h^{(\ell)} = \mathrm{softmax}(S_h^{(\ell)}), \quad C_h^{(\ell)} = A_h^{(\ell)} V_{g(h)}^{(\ell)}
$$

Puis

$$
C^{(\ell)} = \mathrm{Concat}(C_1^{(\ell)}, \dots, C_{12}^{(\ell)}), \quad O^{(\ell)} = C^{(\ell)} W_O^{(\ell)}, \quad R^{(\ell)} = X^{(\ell)} + O^{(\ell)}
$$

### Attention hybride local / global

Pattern par couche, sur 14 couches : 10 locales + 4 globales.

$$
[L, L, L, G, L, L, L, G, L, L, G, L, L, G]
$$

Global aux couches $\ell \in \{4, 8, 11, 14\}$.

**Couche locale** (fenêtre $w$) :

$$
M_{ij}^{(\ell)} = \begin{cases} 0 & \text{si } \max(0, i - w + 1) \le j \le i \\ -\infty & \text{sinon} \end{cases}
$$

**Couche globale** (causal complet) :

$$
M_{ij}^{(\ell)} = \begin{cases} 0 & \text{si } j \le i \\ -\infty & \text{sinon} \end{cases}
$$

Fenêtre recommandée : $w = 512$ (testable $w \in \{512, 1024\}$).

### FFN : dense + MoE

$d_{\text{ff}} = 1536$, SwiGLU. 10 FFN denses + 4 FFN MoE (4 experts, top-2 routing), MoE sur $\ell \in \{7, 9, 11, 13\}$.

**FFN dense.** Après seconde RMSNorm :

$$
Z^{(\ell)} = \mathrm{RMSNorm}(R^{(\ell)})
$$

$$
F^{(\ell)}(z) = W_{\text{down}}^{(\ell)} \Big( \mathrm{SiLU}(z W_{\text{up}}^{(\ell)}) \odot (z W_{\text{gate}}^{(\ell)}) \Big), \quad X^{(\ell+1)} = R^{(\ell)} + F^{(\ell)}(Z^{(\ell)})
$$

**FFN MoE** avec 4 experts, chaque expert

$$
F_e^{(\ell)}(z) = W_{\text{down},e}^{(\ell)} \Big( \mathrm{SiLU}(z W_{\text{up},e}^{(\ell)}) \odot (z W_{\text{gate},e}^{(\ell)}) \Big), \quad e = 1, \dots, 4
$$

Routeur :

$$
a_i^{(\ell)} = z_i^{(\ell)} W_r^{(\ell)} \in \mathbb{R}^4, \quad \mathcal{S}_i^{(\ell)} = \mathrm{Top2}(a_i^{(\ell)})
$$

Poids (softmax restreint aux experts top-2) :

$$
g_{i,e}^{(\ell)} = \begin{cases} \dfrac{\exp(a_{i,e}^{(\ell)})}{\sum_{e' \in \mathcal{S}_i^{(\ell)}} \exp(a_{i,e'}^{(\ell)})} & \text{si } e \in \mathcal{S}_i^{(\ell)} \\ 0 & \text{sinon} \end{cases}
$$

Sortie MoE et résiduel :

$$
Y_i^{(\ell)} = \sum_{e=1}^{4} g_{i,e}^{(\ell)} F_e^{(\ell)}(z_i^{(\ell)}), \quad X_i^{(\ell+1)} = R_i^{(\ell)} + Y_i^{(\ell)}
$$

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

$$
\mathcal{L}_{\text{tot}} = \mathcal{L}_{\text{LM}} + \lambda_{\text{bal}} \sum_{\ell \in \mathcal{L}_{\text{MoE}}} \mathcal{L}_{\text{bal}}^{(\ell)}
$$

### Config cible v1 (résumé)

$$
L = 14, \quad d = 576, \quad H_q = 12, \quad H_{kv} = 3, \quad d_{\text{ff}} = 1536, \quad V = 32\text{k}
$$

$$
10 \text{ local} + 4 \text{ global}, \quad w = 512, \quad 4 \text{ MoE layers}, \quad 4 \text{ experts}, \quad \text{top-2}
$$

Total $\approx 99\text{M}$ paramètres, actif par token nettement inférieur grâce au top-2 MoE.

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
    train.cpp           # boucle d'entraînement
    checkpoint.cpp      # save/load des poids .bin
    tokenizer.cpp       # encode/decode/save/load
    bpe_train.cpp       # train BPE
    tensor.cu           # alloc/copy device <-> host
    embedding.cu        # lookup + logits tied (fwd/bwd)
    matmul.cu           # GEMM naïf row-major (fwd/bwd)
    rmsnorm.cu          # RMSNorm (fwd/bwd)
    softmax.cu          # softmax (fwd/bwd)
    attention.cu        # MHA causale dense (fwd/bwd)
    rope.cu             # RoPE (fwd/bwd inverse rotation)
    ffn.cu              # SwiGLU (fwd/bwd)
    loss.cu             # cross-entropy fused (fwd/bwd)
    optim.cu            # AdamW
    model.cu            # assemblage forward + backward
  tests/
    test_tokenizer.cpp
    test_bpe_train.cpp
    test_matmul.cu      # forward + backward (finite-diff)
    test_rmsnorm.cu     # forward + backward (finite-diff)
    test_attention.cu
    test_rope.cu
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

2. Entraîner le modèle sur le corpus :

```
./toy_llm train data/corpus.txt tokenizer.vocab tokenizer.merges ckpt.bin 2000
# optionnel : learning rate explicite
./toy_llm train data/corpus.txt tokenizer.vocab tokenizer.merges ckpt.bin 2000 3e-3
```

Arguments : `<corpus> <vocab> <merges> <ckpt-out> <steps> [lr]`.
Log la loss toutes les `steps/50` itérations.

3. Génération. Sans checkpoint, les poids sont aléatoires (sortie = bruit) ;
avec `--weights` on charge un checkpoint entraîné :

```
./toy_llm generate tokenizer.vocab tokenizer.merges data/prompt.txt 16
./toy_llm generate tokenizer.vocab tokenizer.merges data/prompt.txt 16 --weights ckpt.bin
```

Arguments : `<vocab> <merges> <prompt-file> <nb-tokens-à-générer> [--weights <ckpt>]`.

**Sanity check du pipeline backward**: le but est de voir la loss descendre
nettement sur quelques centaines / milliers de steps en overfit volontaire
sur `data/corpus.txt`, et que la génération reproduise des n-grammes du
corpus — pas de qualité linguistique attendue sur un tokenizer BPE
byte-level et une archi toy, c'est la preuve que backprop + AdamW +
checkpointing fonctionnent.

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

- pas de GQA, pas de KV cache
- pas de MoE, pas d'attention locale/globale
- pas de FlashAttention
- pas de float16 / bfloat16

Tout cela arrive dans la roadmap v1 ci-dessus.
