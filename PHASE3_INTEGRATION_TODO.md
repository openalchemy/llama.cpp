# Phase 3 — Flash Attention TurboQuant integration (not yet wired)

Phase 1 (CPU reference + tests) and Phase 2 (CUDA encode/decode) landed in
commits `bea0847` and `e2f35ad` on this branch. Phase 3 — making
`flash_attn_ext` actually consume TURBO3/2 K/V tensors during attention — is
NOT yet done; only the safety guard in `llama-kv-cache.cpp` is in place.

The cache will allocate as TURBO3 / TURBO2 (Phase 1 type entries) and the
encode/decode kernels exist (Phase 2), but the graph that runs attention
still expects F16 / F32 K/V tensors. Loading a model with `-ctk turbo3`
today will fail at the first attention step with a type mismatch from
`ggml_flash_attn_ext`.

## Two paths forward

### A. Pre-dequant before attention (~1 day, low risk)

Insert a `ggml_cpy(K_turbo → K_f16_scratch)` and similar for V into the
graph before each `ggml_flash_attn_ext` node. The Phase 2 cpy.cu
dispatcher already routes TURBO3 → F32 correctly; add a TURBO3 → F16
sibling kernel and use it.

Loses the in-attention KV memory locality but gains the 3-4× VRAM
saving the spec promises (because storage stays TURBO; only the working
scratch is F16 and short-lived).

**Code locations:**
- `src/llama-graph.cpp` — graph build, before `ggml_flash_attn_ext` calls
- `ggml/src/ggml-cuda/turboquant.{cu,cuh}` — add f16 → turbo3 + turbo3 → f16
- `ggml/src/ggml-cuda/cpy.cu` — wire those new pairs into the dispatcher

### B. Fused in-tile dequant (~2 days, harder)

Add `type_K = TURBO3 | TURBO2` as a template parameter in
`fattn-tile.cu` so the existing tile loop dequants from shared memory
inline. Q-side gets one FWHT at attention entry; K never gets
inverse-rotated because Hᵀ = H (modulo 1/n which folds into the softmax
scale). This is what the spec's §2.3 sketches.

**Code locations:**
- `ggml/src/ggml-cuda/fattn-tile.cu` — add a `type_K` template branch
  that loads turbo blocks and dequants via the helpers in `turboquant.cuh`
- `ggml/src/ggml-cuda/fattn-common.cuh` — extend the type_traits used
  by the dispatcher
- `ggml/src/ggml-cuda/fattn.cu` — dispatcher entries for new
  `(type_K, type_V)` combinations
- `ggml/src/ggml-cuda/template-instances/fattn-tile*.cu` — re-glob
  picks up new instantiations automatically

## Recommended sequence

1. Path A (1 day) — gets a runnable model so KV memory + perplexity can
   be validated against the paper's numbers.
2. Once Path A's numerical correctness is confirmed (PPL within 0.05 of
   FP16 baseline on wikitext-2 4k ctx), drop in Path B and measure the
   speedup from skipping the pre-dequant pass.

## What's safe to ship right now

- The `head_dim == 128` runtime guard in `llama-kv-cache.cpp` prevents
  silent failures on incompatible models.
- `-ctk fp16 -ctv fp16` still works exactly as before.
- The CPU reference path (Phase 1) round-trips correctly and is what the
  CPU backend will dispatch on (via the type_traits hooked up in
  `ggml.c`), so a CPU-only build with `-ctk turbo3` will work end-to-end
  if anyone wants to test the numerics without a GPU.

— Spec: `doc/specs/2026-05-30-turboquant-kv-cache.md` in the engine
  workspace.
