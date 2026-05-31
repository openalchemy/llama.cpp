#pragma once

#include "ggml-common.h"
#include "convert.cuh"

static __device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

// Single-thread TurboQuant encode. Called by set-rows.cu via the
// set_rows_cuda_quant template — one thread per block_turbo3 / block_turbo2.
// Performance is fine for KV cache writes (few rows per token) even though
// the shared-mem multi-thread variant in turboquant.cu is faster on bulk cpy.
// Matches the CPU reference in ggml-quants.c byte-for-byte. Codebook
// constants are duplicated here to keep this header standalone — same
// values as TURBOQ_INV_SQRT_QK * Lloyd-Max table.

static __device__ void quantize_f32_turbo3_block(const float * __restrict__ x, block_turbo3 * __restrict__ y) {
    constexpr int      QK = 128;
    constexpr float    INV_SQRT_QK = 0.08838834764831845f;
    constexpr float    LV[8] = {
        -2.1519f * INV_SQRT_QK, -1.3439f * INV_SQRT_QK,
        -0.7560f * INV_SQRT_QK, -0.2451f * INV_SQRT_QK,
        +0.2451f * INV_SQRT_QK, +0.7560f * INV_SQRT_QK,
        +1.3439f * INV_SQRT_QK, +2.1519f * INV_SQRT_QK,
    };
    constexpr float    TH[7] = {
        (LV[0] + LV[1]) * 0.5f, (LV[1] + LV[2]) * 0.5f,
        (LV[2] + LV[3]) * 0.5f, (LV[3] + LV[4]) * 0.5f,
        (LV[4] + LV[5]) * 0.5f, (LV[5] + LV[6]) * 0.5f,
        (LV[6] + LV[7]) * 0.5f,
    };

    float buf[QK];
    #pragma unroll 32
    for (int j = 0; j < QK; ++j) buf[j] = x[j];

    // FWHT in registers / local mem. 7 stages for QK=128.
    for (int h = 1; h < QK; h <<= 1) {
        for (int i = 0; i < QK; i += h * 2) {
            for (int j = i; j < i + h; ++j) {
                const float a = buf[j];
                const float b = buf[j + h];
                buf[j]     = a + b;
                buf[j + h] = a - b;
            }
        }
    }

    // L2 norm + normalise.
    float sumsq = 0.0f;
    #pragma unroll 32
    for (int j = 0; j < QK; ++j) sumsq += buf[j] * buf[j];
    const float norm = sqrtf(sumsq);
    y->norm = __float2half(norm);
    const float inv = norm > 0.0f ? (1.0f / norm) : 0.0f;

    uint8_t idx[QK];
    #pragma unroll 32
    for (int j = 0; j < QK; ++j) {
        const float v = buf[j] * inv;
        int k = 0;
        #pragma unroll
        for (int t = 0; t < 7; ++t) k += (v >= TH[t]) ? 1 : 0;
        idx[j] = (uint8_t) k;
    }

    // Pack 128 × 3-bit → 48 bytes.
    for (int b = 0; b < QK / 8; ++b) {
        const uint32_t lo =
              (idx[b*8 + 0] & 7u)
            | ((idx[b*8 + 1] & 7u) << 3)
            | ((idx[b*8 + 2] & 7u) << 6)
            | ((idx[b*8 + 3] & 7u) << 9)
            | ((idx[b*8 + 4] & 7u) << 12)
            | ((idx[b*8 + 5] & 7u) << 15)
            | ((idx[b*8 + 6] & 7u) << 18)
            | ((idx[b*8 + 7] & 7u) << 21);
        y->qs[b*3 + 0] = (uint8_t)(lo        & 0xFFu);
        y->qs[b*3 + 1] = (uint8_t)((lo >> 8 ) & 0xFFu);
        y->qs[b*3 + 2] = (uint8_t)((lo >> 16) & 0xFFu);
    }
}

static __device__ void quantize_f32_turbo2_block(const float * __restrict__ x, block_turbo2 * __restrict__ y) {
    constexpr int      QK = 128;
    constexpr float    INV_SQRT_QK = 0.08838834764831845f;
    constexpr float    LV[4] = {
        -1.5104f * INV_SQRT_QK, -0.4528f * INV_SQRT_QK,
        +0.4528f * INV_SQRT_QK, +1.5104f * INV_SQRT_QK,
    };
    constexpr float    TH[3] = {
        (LV[0] + LV[1]) * 0.5f,
        (LV[1] + LV[2]) * 0.5f,
        (LV[2] + LV[3]) * 0.5f,
    };

    float buf[QK];
    #pragma unroll 32
    for (int j = 0; j < QK; ++j) buf[j] = x[j];

    for (int h = 1; h < QK; h <<= 1) {
        for (int i = 0; i < QK; i += h * 2) {
            for (int j = i; j < i + h; ++j) {
                const float a = buf[j];
                const float b = buf[j + h];
                buf[j]     = a + b;
                buf[j + h] = a - b;
            }
        }
    }

    float sumsq = 0.0f;
    #pragma unroll 32
    for (int j = 0; j < QK; ++j) sumsq += buf[j] * buf[j];
    const float norm = sqrtf(sumsq);
    y->norm = __float2half(norm);
    const float inv = norm > 0.0f ? (1.0f / norm) : 0.0f;

    uint8_t idx[QK];
    #pragma unroll 32
    for (int j = 0; j < QK; ++j) {
        const float v = buf[j] * inv;
        int k = 0;
        #pragma unroll
        for (int t = 0; t < 3; ++t) k += (v >= TH[t]) ? 1 : 0;
        idx[j] = (uint8_t) k;
    }

    for (int i = 0; i < QK / 4; ++i) {
        y->qs[i] = (uint8_t)(
              ((idx[i*4 + 0] & 3u)     )
            | ((idx[i*4 + 1] & 3u) << 2)
            | ((idx[i*4 + 2] & 3u) << 4)
            | ((idx[i*4 + 3] & 3u) << 6)
        );
    }
}

static __device__ void quantize_f32_q4_0_block(const float * __restrict__ x, block_q4_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK4_0/2 + j]*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q4_1_block(const float * __restrict__ x, block_q4_1 * __restrict__ y) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = x[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q5_0_block(const float * __restrict__ x, block_q5_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q5_1_block(const float * __restrict__ x, block_q5_1 * __restrict__ y) {
    float min = x[0];
    float max = x[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = x[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (x[0       + j] - min)*id;
        const float x1 = (x[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q8_0_block(const float * __restrict__ x, block_q8_0 * __restrict__ y) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = x[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = x[j]*id;
        y->qs[j] = roundf(x0);
    }
}

static __device__ void quantize_f32_iq4_nl_block(const float * __restrict__ x, block_iq4_nl * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = x[0        + j]*id;
        const float x1 = x[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = x[0        + j]*x[0        + j];
        const float w1 = x[QK4_NL/2 + j]*x[QK4_NL/2 + j];
        sumqx += w0*v0*x[j] + w1*v1*x[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    y->d = sumq2 > 0 ? sumqx/sumq2 : d;
}

// Wrapper functions for cpy.cu compatibility
static __device__ void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    quantize_f32_q4_0_block((const float *)cxi, (block_q4_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    quantize_f32_q4_1_block((const float *)cxi, (block_q4_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    quantize_f32_q5_0_block((const float *)cxi, (block_q5_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    quantize_f32_q5_1_block((const float *)cxi, (block_q5_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    quantize_f32_q8_0_block((const float *)cxi, (block_q8_0 *)cdsti);
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    quantize_f32_iq4_nl_block((const float *)cxi, (block_iq4_nl *)cdsti);
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_scalar(const char * cxi, char * cdsti) {
    *(dst_t *) cdsti = ggml_cuda_cast<dst_t>(*(const src_t *) cxi);
}
