// TurboQuant CUDA kernels — declarations.
//
// Spec doc/specs/2026-05-30-turboquant-kv-cache.md in the engine workspace.
// CPU reference lives in ggml-quants.c; these are the dispatch entry points
// used by cpy.cu when KV cache types K or V are GGML_TYPE_TURBO3/2.

#pragma once

#include "common.cuh"

// F32 source → packed block_turbo3 destination.
// ne must be a multiple of QK_TURBO (= 128). One thread block per source
// block; 128 threads per block (one per coordinate).
void ggml_cpy_f32_turbo3_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream);

// Packed block_turbo3 → F32. Inverse of the above.
void ggml_cpy_turbo3_f32_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream);

// F32 → block_turbo2.
void ggml_cpy_f32_turbo2_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream);

// block_turbo2 → F32.
void ggml_cpy_turbo2_f32_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream);

// FWHT a single 128-element vector held in shared memory by a 128-thread block.
// Declared here so fattn-tile.cu (Phase 3) can include this header and reuse
// the same shared-memory butterfly without pulling in the rest of the file.
__device__ __forceinline__ void turboq_fwht128_smem(float * shared);

// Codebook lookup helpers (per-coord, branchless).
__device__ __forceinline__ int  turboq_pick_3bit_dev(float v);
__device__ __forceinline__ int  turboq_pick_2bit_dev(float v);
__device__ __forceinline__ float turboq_decode_3bit_dev(int idx);
__device__ __forceinline__ float turboq_decode_2bit_dev(int idx);
