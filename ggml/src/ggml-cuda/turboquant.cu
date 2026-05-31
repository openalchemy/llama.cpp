// TurboQuant CUDA kernels.
//
// Spec doc/specs/2026-05-30-turboquant-kv-cache.md in the engine workspace.
// Companion to the CPU reference in ggml-quants.c.
//
// Block layout per turbo type matches block_turbo3 / block_turbo2 from
// ggml-common.h: an FP16 norm followed by packed 3- or 2-bit indices.
//
// Kernel shape: one thread block per (head_dim = 128) source block, one
// thread per coordinate. Shared memory holds the 128 floats during FWHT
// and norm reduction.

#include "turboquant.cuh"

// QK_TURBO is defined by ggml-common.h via common.cuh's includes.
#ifndef QK_TURBO
#error "QK_TURBO must be visible — check ggml-common.h include order"
#endif

// =============================================================================
// Codebook (must match the CPU constants in ggml-quants.c exactly)
// =============================================================================

__device__ __constant__ float c_turbo3_codebook[8] = {
    -2.1519f * 0.08838834764831845f,
    -1.3439f * 0.08838834764831845f,
    -0.7560f * 0.08838834764831845f,
    -0.2451f * 0.08838834764831845f,
    +0.2451f * 0.08838834764831845f,
    +0.7560f * 0.08838834764831845f,
    +1.3439f * 0.08838834764831845f,
    +2.1519f * 0.08838834764831845f,
};

__device__ __constant__ float c_turbo3_thresholds[7] = {
    (-2.1519f + -1.3439f) * 0.5f * 0.08838834764831845f,
    (-1.3439f + -0.7560f) * 0.5f * 0.08838834764831845f,
    (-0.7560f + -0.2451f) * 0.5f * 0.08838834764831845f,
    (-0.2451f + +0.2451f) * 0.5f * 0.08838834764831845f,
    (+0.2451f + +0.7560f) * 0.5f * 0.08838834764831845f,
    (+0.7560f + +1.3439f) * 0.5f * 0.08838834764831845f,
    (+1.3439f + +2.1519f) * 0.5f * 0.08838834764831845f,
};

__device__ __constant__ float c_turbo2_codebook[4] = {
    -1.5104f * 0.08838834764831845f,
    -0.4528f * 0.08838834764831845f,
    +0.4528f * 0.08838834764831845f,
    +1.5104f * 0.08838834764831845f,
};

__device__ __constant__ float c_turbo2_thresholds[3] = {
    (-1.5104f + -0.4528f) * 0.5f * 0.08838834764831845f,
    (-0.4528f + +0.4528f) * 0.5f * 0.08838834764831845f,
    (+0.4528f + +1.5104f) * 0.5f * 0.08838834764831845f,
};

__device__ __forceinline__ int turboq_pick_3bit_dev(float v) {
    int idx = 0;
    #pragma unroll
    for (int i = 0; i < 7; ++i) {
        idx += (v >= c_turbo3_thresholds[i]) ? 1 : 0;
    }
    return idx;
}

__device__ __forceinline__ int turboq_pick_2bit_dev(float v) {
    int idx = 0;
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        idx += (v >= c_turbo2_thresholds[i]) ? 1 : 0;
    }
    return idx;
}

__device__ __forceinline__ float turboq_decode_3bit_dev(int idx) {
    return c_turbo3_codebook[idx & 7];
}

__device__ __forceinline__ float turboq_decode_2bit_dev(int idx) {
    return c_turbo2_codebook[idx & 3];
}

// =============================================================================
// FWHT specialised for n = 128 with one thread per coordinate
//
// Seven butterfly stages with __syncthreads() between them. Each thread holds
// exactly one element of the input vector; partner index per stage is
// (lane ^ stride). All threads must execute in lockstep — caller must ensure
// blockDim.x == 128.
// =============================================================================

__device__ __forceinline__ void turboq_fwht128_smem(float * shared) {
    const int lane = threadIdx.x;  // 0..127
    #pragma unroll
    for (int stride = 1; stride < QK_TURBO; stride <<= 1) {
        const int partner = lane ^ stride;
        const float a = shared[lane];
        const float b = shared[partner];
        __syncthreads();
        if ((lane & stride) == 0) {
            shared[lane]    = a + b;
            shared[partner] = a - b;
        }
        __syncthreads();
    }
}

// =============================================================================
// Encode: F32 → block_turbo3 / block_turbo2
//
// One CUDA block per QK_TURBO source coordinates. blockDim.x = QK_TURBO.
// Strided iteration over the destination via i_block = blockIdx.x.
// =============================================================================

template <bool IS_3BIT>
static __global__ void cpy_f32_turbo_kernel(
    const char * __restrict__ cx, char * __restrict__ cdst,
    const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13)
{
    const int64_t i_block = blockIdx.x;
    const int64_t block_start = i_block * QK_TURBO;
    if (block_start >= ne) return;

    // Compute strided element index for the source — same arithmetic as the
    // existing cpy_blck_* kernels. The CPU reference assumes a contiguous
    // layout, which is what KV-cache writes give us; we keep the strided
    // path so this kernel can also serve non-cache copies.
    const int64_t i03 = block_start / (ne00 * ne01 * ne02);
    const int64_t i02 = (block_start - i03*ne00*ne01*ne02) / (ne00*ne01);
    const int64_t i01 = (block_start - i03*ne00*ne01*ne02 - i02*ne01*ne00) / ne00;
    const int64_t i00 =  block_start - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
    const int64_t x_offset = i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;

    // Destination block index in the packed array.
    const int64_t i13 = i_block / (ne10/QK_TURBO * ne11 * ne12);
    const int64_t i12 = (i_block - i13*(ne10/QK_TURBO)*ne11*ne12) / ((ne10/QK_TURBO)*ne11);
    const int64_t i11 = (i_block - i13*(ne10/QK_TURBO)*ne11*ne12 - i12*(ne10/QK_TURBO)*ne11) / (ne10/QK_TURBO);
    const int64_t i10 =  i_block - i13*(ne10/QK_TURBO)*ne11*ne12 - i12*(ne10/QK_TURBO)*ne11 - i11*(ne10/QK_TURBO);
    const int64_t dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

    __shared__ float s_buf[QK_TURBO];
    __shared__ float s_warp_sums[QK_TURBO / 32];  // = 4 warps

    const int lane = threadIdx.x;
    const int warp = lane >> 5;
    const int lane_in_warp = lane & 31;

    // 1. Load source coord.
    const float * src = (const float *)(cx + x_offset);
    s_buf[lane] = src[lane];
    __syncthreads();

    // 2. FWHT in shared memory.
    turboq_fwht128_smem(s_buf);

    // 3. L2 norm: warp reduction → per-warp partial → block reduce.
    const float v = s_buf[lane];
    float local = v * v;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local += __shfl_xor_sync(0xFFFFFFFF, local, off);
    }
    if (lane_in_warp == 0) {
        s_warp_sums[warp] = local;
    }
    __syncthreads();
    // Thread 0 sums the 4 warp partials.
    if (lane == 0) {
        float t = 0.0f;
        #pragma unroll
        for (int w = 0; w < QK_TURBO / 32; ++w) t += s_warp_sums[w];
        s_warp_sums[0] = t;  // reuse slot 0 to broadcast
    }
    __syncthreads();
    const float norm = sqrtf(s_warp_sums[0]);

    // 4. Per-coord pick of nearest codebook index.
    const float inv_norm = (norm > 0.0f) ? (1.0f / norm) : 0.0f;
    const float vn = v * inv_norm;
    const int idx = IS_3BIT ? turboq_pick_3bit_dev(vn) : turboq_pick_2bit_dev(vn);

    // 5. Pack indices into the block. Use a shared scratch + thread-0 writes
    //    for the final byte stream so we don't issue 128 single-byte global
    //    stores per block. Same encoding as turboq_pack_3bit / 2bit in
    //    ggml-quants.c.
    __shared__ uint8_t s_idx[QK_TURBO];
    s_idx[lane] = (uint8_t) idx;
    __syncthreads();

    if (lane == 0) {
        if (IS_3BIT) {
            block_turbo3 * dst = (block_turbo3 *)(cdst + dst_offset);
            dst->norm = __float2half(norm);
            for (int b = 0; b < QK_TURBO / 8; ++b) {
                const uint32_t i0 = s_idx[b*8 + 0] & 7u;
                const uint32_t i1 = s_idx[b*8 + 1] & 7u;
                const uint32_t i2 = s_idx[b*8 + 2] & 7u;
                const uint32_t i3 = s_idx[b*8 + 3] & 7u;
                const uint32_t i4 = s_idx[b*8 + 4] & 7u;
                const uint32_t i5 = s_idx[b*8 + 5] & 7u;
                const uint32_t i6 = s_idx[b*8 + 6] & 7u;
                const uint32_t i7 = s_idx[b*8 + 7] & 7u;
                const uint32_t lo = i0 | (i1 << 3) | (i2 << 6) | (i3 << 9) |
                                    (i4 << 12) | (i5 << 15) | (i6 << 18) | (i7 << 21);
                dst->qs[b*3 + 0] = (uint8_t)( lo        & 0xFFu);
                dst->qs[b*3 + 1] = (uint8_t)((lo >> 8 ) & 0xFFu);
                dst->qs[b*3 + 2] = (uint8_t)((lo >> 16) & 0xFFu);
            }
        } else {
            block_turbo2 * dst = (block_turbo2 *)(cdst + dst_offset);
            dst->norm = __float2half(norm);
            for (int i = 0; i < QK_TURBO / 4; ++i) {
                dst->qs[i] = (uint8_t)(
                    ((s_idx[i*4 + 0] & 3u)     ) |
                    ((s_idx[i*4 + 1] & 3u) << 2) |
                    ((s_idx[i*4 + 2] & 3u) << 4) |
                    ((s_idx[i*4 + 3] & 3u) << 6)
                );
            }
        }
    }
}

// =============================================================================
// Decode: block_turbo3 / block_turbo2 → F32
// =============================================================================

// =============================================================================
// F16-emitting decoder (Phase 3 Path A)
//
// Same algorithm as cpy_turbo_f32_kernel — unpack idx, codebook × norm,
// FWHT, 1/n — but writes __half. Used by graph nodes that pre-dequant
// turbo KV into an F16 scratch immediately before Flash Attention.
// =============================================================================

template <bool IS_3BIT>
static __global__ void cpy_turbo_f16_kernel(
    const char * __restrict__ cx, char * __restrict__ cdst,
    const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13)
{
    const int64_t i_block = blockIdx.x;
    const int64_t block_start = i_block * QK_TURBO;
    if (block_start >= ne) return;

    const int64_t i03 = i_block / ((ne00/QK_TURBO) * ne01 * ne02);
    const int64_t i02 = (i_block - i03*(ne00/QK_TURBO)*ne01*ne02) / ((ne00/QK_TURBO)*ne01);
    const int64_t i01 = (i_block - i03*(ne00/QK_TURBO)*ne01*ne02 - i02*(ne00/QK_TURBO)*ne01) / (ne00/QK_TURBO);
    const int64_t i00 =  i_block - i03*(ne00/QK_TURBO)*ne01*ne02 - i02*(ne00/QK_TURBO)*ne01 - i01*(ne00/QK_TURBO);
    const int64_t x_offset = i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;

    const int64_t i13 = block_start / (ne10 * ne11 * ne12);
    const int64_t i12 = (block_start - i13*ne10*ne11*ne12) / (ne10*ne11);
    const int64_t i11 = (block_start - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
    const int64_t i10 =  block_start - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
    const int64_t dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

    const int lane = threadIdx.x;
    __shared__ float s_buf[QK_TURBO];

    int idx;
    float norm;
    if (IS_3BIT) {
        const block_turbo3 * src = (const block_turbo3 *)(cx + x_offset);
        norm = __half2float(src->norm);
        const int chunk = lane >> 3;
        const int off   = lane & 7;
        const uint32_t lo = (uint32_t) src->qs[chunk*3 + 0]
                          | ((uint32_t) src->qs[chunk*3 + 1] << 8)
                          | ((uint32_t) src->qs[chunk*3 + 2] << 16);
        idx = (int)((lo >> (off * 3)) & 7u);
        s_buf[lane] = turboq_decode_3bit_dev(idx) * norm;
    } else {
        const block_turbo2 * src = (const block_turbo2 *)(cx + x_offset);
        norm = __half2float(src->norm);
        const int byte = lane >> 2;
        const int off  = lane & 3;
        idx = (int)((src->qs[byte] >> (off * 2)) & 3u);
        s_buf[lane] = turboq_decode_2bit_dev(idx) * norm;
    }
    __syncthreads();

    turboq_fwht128_smem(s_buf);
    const float scaled = s_buf[lane] / (float) QK_TURBO;

    __half * dst = (__half *)(cdst + dst_offset);
    dst[lane] = __float2half(scaled);
}

template <bool IS_3BIT>
static __global__ void cpy_turbo_f32_kernel(
    const char * __restrict__ cx, char * __restrict__ cdst,
    const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13)
{
    const int64_t i_block = blockIdx.x;
    const int64_t block_start = i_block * QK_TURBO;
    if (block_start >= ne) return;

    // Source block offset.
    const int64_t i03 = i_block / ((ne00/QK_TURBO) * ne01 * ne02);
    const int64_t i02 = (i_block - i03*(ne00/QK_TURBO)*ne01*ne02) / ((ne00/QK_TURBO)*ne01);
    const int64_t i01 = (i_block - i03*(ne00/QK_TURBO)*ne01*ne02 - i02*(ne00/QK_TURBO)*ne01) / (ne00/QK_TURBO);
    const int64_t i00 =  i_block - i03*(ne00/QK_TURBO)*ne01*ne02 - i02*(ne00/QK_TURBO)*ne01 - i01*(ne00/QK_TURBO);
    const int64_t x_offset = i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;

    // Destination element index.
    const int64_t i13 = block_start / (ne10 * ne11 * ne12);
    const int64_t i12 = (block_start - i13*ne10*ne11*ne12) / (ne10*ne11);
    const int64_t i11 = (block_start - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
    const int64_t i10 =  block_start - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
    const int64_t dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

    const int lane = threadIdx.x;
    __shared__ float s_buf[QK_TURBO];

    // 1. Unpack idx and look up codebook.
    int idx;
    float norm;
    if (IS_3BIT) {
        const block_turbo3 * src = (const block_turbo3 *)(cx + x_offset);
        norm = __half2float(src->norm);
        // 3-bit unpack: 8 indices per 3-byte chunk. Compute which chunk + offset.
        const int chunk = lane >> 3;          // 0..15
        const int off   = lane & 7;
        const uint32_t lo = (uint32_t) src->qs[chunk*3 + 0]
                          | ((uint32_t) src->qs[chunk*3 + 1] << 8)
                          | ((uint32_t) src->qs[chunk*3 + 2] << 16);
        idx = (int)((lo >> (off * 3)) & 7u);
        s_buf[lane] = turboq_decode_3bit_dev(idx) * norm;
    } else {
        const block_turbo2 * src = (const block_turbo2 *)(cx + x_offset);
        norm = __half2float(src->norm);
        const int byte = lane >> 2;
        const int off  = lane & 3;
        idx = (int)((src->qs[byte] >> (off * 2)) & 3u);
        s_buf[lane] = turboq_decode_2bit_dev(idx) * norm;
    }
    __syncthreads();

    // 2. Inverse FWHT (= FWHT again) + 1/n scaling.
    turboq_fwht128_smem(s_buf);
    const float scaled = s_buf[lane] / (float) QK_TURBO;

    // 3. Scatter to destination.
    float * dst = (float *)(cdst + dst_offset);
    dst[lane] = scaled;
}

// =============================================================================
// Host-side launchers (called from cpy.cu)
// =============================================================================

void ggml_cpy_f32_turbo3_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream)
{
    GGML_ASSERT(ne % QK_TURBO == 0);
    const int64_t num_blocks = ne / QK_TURBO;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_turbo_kernel<true><<<(unsigned)num_blocks, QK_TURBO, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
         ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

void ggml_cpy_turbo3_f32_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream)
{
    GGML_ASSERT(ne % QK_TURBO == 0);
    const int64_t num_blocks = ne / QK_TURBO;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_turbo_f32_kernel<true><<<(unsigned)num_blocks, QK_TURBO, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
         ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

void ggml_cpy_f32_turbo2_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream)
{
    GGML_ASSERT(ne % QK_TURBO == 0);
    const int64_t num_blocks = ne / QK_TURBO;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_turbo_kernel<false><<<(unsigned)num_blocks, QK_TURBO, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
         ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

void ggml_cpy_turbo2_f32_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream)
{
    GGML_ASSERT(ne % QK_TURBO == 0);
    const int64_t num_blocks = ne / QK_TURBO;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_turbo_f32_kernel<false><<<(unsigned)num_blocks, QK_TURBO, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
         ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

void ggml_cpy_turbo3_f16_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream)
{
    GGML_ASSERT(ne % QK_TURBO == 0);
    const int64_t num_blocks = ne / QK_TURBO;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_turbo_f16_kernel<true><<<(unsigned)num_blocks, QK_TURBO, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
         ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

void ggml_cpy_turbo2_f16_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream)
{
    GGML_ASSERT(ne % QK_TURBO == 0);
    const int64_t num_blocks = ne / QK_TURBO;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_turbo_f16_kernel<false><<<(unsigned)num_blocks, QK_TURBO, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
         ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}
