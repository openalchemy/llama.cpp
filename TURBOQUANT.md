# TurboQuant — Compressed KV cache for llama.cpp

**What this fork adds**: two new `ggml_type` slots, `GGML_TYPE_TURBO3` and
`GGML_TYPE_TURBO2`, that store the attention KV cache at 3 or 2 bits per
coordinate using the TurboQuant algorithm from
[Zandieh et al., "TurboQuant: Online Vector Quantization with Near-Optimal
Distortion Rate" (Google Research, ICLR 2026)](https://arxiv.org/abs/2504.19874).

The KV cache is the dominant runtime VRAM consumer for long-context inference
on a single GPU. Compressing it lets you either run a bigger model, a longer
context, or a larger batch on the same card.

## Practical impact (measured on RTX 5080, 16 GB)

| Model | KV type | n_ctx | VRAM used | Δ |
|---|---|---|---|---|
| **Qwen2.5-Coder-14B Q4_K_M** | FP16  | 32 768 | 15 656 MiB | — |
| **Qwen2.5-Coder-14B Q4_K_M** | TURBO3 | 32 768 | 11 238 MiB | **−4 418 MiB (−4.3 GB)** |
| Qwen3.5-9B Q4_K_M (head_dim 256) | FP16   | 32 768 | 7 968 MiB | — |
| Qwen3.5-9B Q4_K_M (head_dim 256) | TURBO3 | 32 768 | 7 140 MiB | −828 MiB |
| Qwen2.5-Coder-14B Q4_K_M | TURBO3 | 32 768 | — | **prompt 1.24×, gen 1.47×** vs FP16 |

The speed-up on Qwen2.5-Coder-14B comes from the cache being small enough to
stay in L2 / register-tile-resident during attention, removing global-memory
traffic that the FP16 cache forces.

## How it works in one paragraph

The TurboQuant paper's core insight is that a random orthogonal rotation of
each 128-coord KV head spreads coordinate magnitudes into a near-uniform
distribution that scalar quantization handles near-optimally. We use the Fast
Walsh-Hadamard Transform (Hadamard ±1 entries) instead of a true Haar-random
matrix — same statistical property in expectation, ~10× faster kernel, and
the rotation is self-inverse so decode is free of an extra matmul. Per
128-coord block we store one FP16 L2 norm plus 128 packed 3- or 2-bit
codebook indices. At attention time the K and V tensors are cast back to F16
just before flash-attention (Path A); a future Path B will fuse the
dequant into the attention kernel.

Full design + algorithm + correctness validation in
[`docs/turboquant.md`](docs/turboquant.md).

## How to use it from the CLI

```bash
./llama-cli \
  -m Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf \
  -ngl 99 -fa 1 -c 32768 \
  -ctk turbo3 -ctv turbo3 \
  -p "def fibonacci(n):" -n 64
```

`-ctk` and `-ctv` accept `turbo3` and `turbo2` alongside `fp16`, `q8_0`,
`q4_0`, etc.

## Compatibility

- **Requires `n_embd_head_k` to be a positive multiple of 128** (the FWHT
  block size). This covers Llama 3.x, Qwen 2.5 / 3 / 3.5 / 3.6, Mistral 7B,
  Mixtral, Gemma — almost every modern attention model. Multi-Token
  Prediction (MTP) heads in the qwen35 architecture work because they reuse
  the base model's head_dim.
- TurboQuant kernels currently target **CUDA**. CPU fallback exists for
  correctness tests but is not optimised. ROCm/Metal/Vulkan TBD.
- Flash Attention (`-fa 1`) is **required** today — the slow-path attention
  kernels don't carry the TURBO3/TURBO2 dispatch yet.

## What's in the fork (vs upstream)

```
ggml/include/ggml.h            new GGML_TYPE_TURBO3 / TURBO2 enum values
ggml/src/ggml-common.h         block_turbo3 / block_turbo2 structs
ggml/src/ggml-quants.{c,h}     CPU encode / decode + Lloyd-Max codebook
ggml/src/ggml.c                type_traits entries for the new types
ggml/src/ggml-cpu/*            CPU dispatcher hooks + thin wrappers
ggml/src/ggml-cuda/
  turboquant.{cu,cuh}          new — shared-mem CUDA encode/decode kernels
  cpy.cu / cpy-utils.cuh       turbo3/2 ↔ F16/F32 dispatch + per-block encode
  set-rows.cu                  KV cache write path uses turbo encode
  ggml-cuda.cu                 supports_op() whitelists the new pairs
src/llama-graph.cpp            ggml_cast TURBO3/2 → F16 before flash_attn_ext
src/llama-kv-cache.cpp         head_dim multiple-of-128 hard guard at construction
common/arg.cpp                 -ctk turbo3 / -ctv turbo3 parser
tests/test-turboquant.cpp      CPU round-trip + MSE + norm-preservation tests
tests/test-backend-ops.cpp     turbo3/2 added to the CPY all_types iterator
tools/turboquant/
  gen_codebook.py              regen Lloyd-Max codebooks for arbitrary bit widths
```

## Roadmap

- **Path B**: fused in-tile dequant inside the flash-attention kernel, so K
  never gets materialised in F16. Removes the only meaningful Path A overhead.
- **Block size variants**: 256-coord blocks for models like Gemma 4 where
  head_dim is exactly 256; smaller blocks for models where 128 is too coarse.
- **Outlier channels**: paper §5.2 reports significant quality gains by
  bumping a handful of outlier channels to 4 bits while keeping the rest at
  2.5 bits. Not yet implemented.
- **Upstream merge**: opening a PR against `ggml-org/llama.cpp` once the
  set-rows + cpy + fattn paths are fully merged into the upstream style.

## License

[MIT](LICENSE) — inherited from upstream llama.cpp. All TurboQuant patches
in this fork are released under the same license.

## Citation

If you use this fork in research, please cite the original paper:

```bibtex
@inproceedings{zandieh2026turboquant,
  title={TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  booktitle={ICLR},
  year={2026}
}
```
