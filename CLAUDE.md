# CLAUDE.md

Instructions pour Claude Code quand il travaille sur ce repo.

## Contexte projet

`toy_llm_cuda` est un mini LLM decoder-only écrit from-scratch en **C++ / CUDA**,
sans dépendance PyTorch. Deux étapes :

- **v0 (état actuel)** — toy LLM *inference-first* : BPE tokenizer trainable,
  forward CUDA, génération greedy. Poids aléatoires.
- **v1 (cible)** — tiny SOTA ~100M params : GQA + RoPE + local/global attention
  + dense/MoE. **Backward à la main**, pas d'autograd.

Voir `README.md` pour les équations, la config et la roadmap détaillée.

## Principes de code

1. **Pas de frameworks** : pas de Torch, pas de cuBLAS, pas de cuDNN, pas
   d'Eigen. Kernels CUDA et C++ standard uniquement. Si tu veux ajouter une
   lib, demande d'abord.
2. **Explicit > smart** : code lisible, peu d'abstractions, peu de templates.
   Le but pédagogique prime.
3. **Float32 partout en v0.** Pas de mixed precision.
4. **Row-major** pour tous les tenseurs (convention NumPy / PyTorch).
5. **Noms explicites** : `d_model`, `n_heads`, `seq_len`, `t` pour position.
6. **Pas de commentaires qui paraphrasent le code.** Commentaires seulement
   pour expliquer un *pourquoi* non évident (invariants, formules maths).
7. **Différenciation manuelle** quand elle arrivera : écrire les gradients
   explicitement, kernel par kernel. Pas de tape, pas d'autograd.

## Conventions tenseurs

- Activations : `[T, d]` ou `[B, T, d]` (B=1 pour l'instant).
- Poids linéaires row-major : `W ∈ R^(in × out)`, donc `y = x @ W` avec
  `x ∈ R^(… × in)` et `y ∈ R^(… × out)`.
- Attention head split : vue `[T, H, d_h]` puis permute en `[H, T, d_h]`.
- Scores d'attention : `[H, T, T]` avec mask causal additif.

## Kernels CUDA — règles

- Chaque kernel a une fonction host `launch_xxx(...)` qui gère les dims de
  grid/block et retourne `void`. Aucune erreur CUDA n'est swallowed : utiliser
  `CUDA_CHECK` défini dans `include/common.cuh`.
- Allocation / libération device via `DeviceBuffer<float>` (RAII simple).
- Tous les kernels testables avec un `tests/test_*.cu` indépendant.

## Build

- `make` : tout.
- `make tokenizer` : CPU-only, pas besoin de CUDA installé.
- `make tests` : tous les tests.
- `make clean`.
- Cibler une archi : `make SM=86`.

## Flow de dev conseillé

Quand on modifie un kernel :

1. Écrire / mettre à jour le test dans `tests/test_<kernel>.cu` avec
   référence CPU simple (boucle triple pour matmul, etc.) et `allclose`
   relatif sur float.
2. `make tests && ./tests/test_<kernel>`.
3. Seulement après, brancher dans `src/model.cu`.

## Ne pas faire (v0)

- Pas de backward.
- Pas d'optimiseur.
- Pas de GQA, RoPE, KV cache, MoE, sliding attention, FlashAttention.
- Pas de float16 / bfloat16.
- Pas de dépendance externe.
- Pas de tokenizer byte-level GPT complexe : un BPE char-level simple suffit.

## Commandes de run (sanity check)

```
make
./toy_llm train-bpe data/corpus.txt tokenizer.vocab tokenizer.merges 256
./toy_llm generate  tokenizer.vocab tokenizer.merges data/prompt.txt 16
```

Le résultat textuel est du bruit (poids aléatoires) : c'est **normal** en v0.
Ce qu'on valide, c'est que le pipeline complet tourne sans crash.

## Roadmap (rappel court)

v0 (ici) → KV cache → GQA + RoPE → local/global → backward manuel + AdamW →
scale d=576, 14 layers, V=32k → MoE top-2 → training réel.

Ne pas sauter d'étape.
