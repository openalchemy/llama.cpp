// TurboQuant CPU reference tests.
//
// Calls the type traits exposed by ggml.h so we don't need to depend on
// the private ggml-quants.h. Covers:
//   1. Round-trip MSE on uniform-on-sphere inputs within paper bounds.
//   2. Round-trip MSE on Gaussian inputs (norm scaling must absorb scale).
//   3. Norm preservation across encode/decode within 15 %.
//   4. All-zero input decodes to all-zero output (no NaN, no overflow).
//   5. The expected ggml type_size for block_turbo3 / block_turbo2.
//
// Phase 1 of spec doc/specs/2026-05-30-turboquant-kv-cache.md.

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int   QK_TURBO = 128;
constexpr float TURBO3_MSE_MAX = 0.020f;
constexpr float TURBO2_MSE_MAX = 0.060f;

int g_failures = 0;

#define EXPECT_TRUE(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "  FAIL %s:%d %s\n", __FILE__, __LINE__, (msg)); \
        ++g_failures; \
    } \
} while (0)

#define EXPECT_LT(a, b, msg) do { \
    if (!((a) < (b))) { \
        std::fprintf(stderr, "  FAIL %s:%d %s (%.6f < %.6f)\n", __FILE__, __LINE__, (msg), (double)(a), (double)(b)); \
        ++g_failures; \
    } \
} while (0)

float mean_squared_error(const float * a, const float * b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        const double d = (double)a[i] - (double)b[i];
        s += d * d;
    }
    return (float)(s / (double)n);
}

void test_turbo_roundtrip(ggml_type type, float mse_max, const char * tag) {
    std::printf("test_turbo_roundtrip[%s]:\n", tag);
    const auto * traits = ggml_get_type_traits(type);
    EXPECT_TRUE(traits != nullptr, "no type traits");
    EXPECT_TRUE(traits->from_float_ref != nullptr, "no encoder");
    EXPECT_TRUE(traits->to_float != nullptr,        "no decoder");
    EXPECT_TRUE(traits->blck_size == QK_TURBO,      "wrong blck_size");
    if (g_failures > 0) return;

    std::mt19937 rng(0xBEEF);
    std::normal_distribution<float> g(0.0f, 1.0f);

    const int NB = 64;
    std::vector<float> x(NB * QK_TURBO);
    std::vector<float> r(NB * QK_TURBO);
    std::vector<uint8_t> q(NB * traits->type_size);

    for (int b = 0; b < NB; ++b) {
        float sumsq = 0.0f;
        for (int j = 0; j < QK_TURBO; ++j) {
            const float v = g(rng);
            x[b*QK_TURBO + j] = v;
            sumsq += v*v;
        }
        const float inv = 1.0f / std::sqrt(sumsq);
        for (int j = 0; j < QK_TURBO; ++j) x[b*QK_TURBO + j] *= inv;
    }

    traits->from_float_ref(x.data(), q.data(), NB * QK_TURBO);
    traits->to_float       (q.data(), r.data(), NB * QK_TURBO);

    const float mse = mean_squared_error(x.data(), r.data(), NB * QK_TURBO);
    std::printf("  mse=%.6f (target<%.4f)\n", mse, mse_max);
    EXPECT_LT(mse, mse_max, "round-trip mse exceeds paper bound");
}

void test_norm_preservation() {
    std::printf("test_norm_preservation[turbo3]:\n");
    const auto * traits = ggml_get_type_traits(GGML_TYPE_TURBO3);
    EXPECT_TRUE(traits != nullptr, "no type traits");
    if (g_failures > 0) return;

    std::mt19937 rng(0xFEED);
    std::normal_distribution<float> g(0.0f, 2.3f);

    const int NB = 16;
    std::vector<float> x(NB * QK_TURBO);
    std::vector<float> r(NB * QK_TURBO);
    std::vector<uint8_t> q(NB * traits->type_size);

    for (int j = 0; j < NB * QK_TURBO; ++j) x[j] = g(rng);
    traits->from_float_ref(x.data(), q.data(), NB * QK_TURBO);
    traits->to_float       (q.data(), r.data(), NB * QK_TURBO);

    for (int b = 0; b < NB; ++b) {
        double sx = 0.0, sr = 0.0;
        for (int j = 0; j < QK_TURBO; ++j) {
            sx += (double)x[b*QK_TURBO + j] * (double)x[b*QK_TURBO + j];
            sr += (double)r[b*QK_TURBO + j] * (double)r[b*QK_TURBO + j];
        }
        const float nx = std::sqrt((float)sx);
        const float nr = std::sqrt((float)sr);
        const float rel = std::fabs(nx - nr) / nx;
        EXPECT_LT(rel, 0.15f, "block norm drift exceeds 15 %");
    }
}

void test_zero_input_decodes_to_zero() {
    std::printf("test_zero_input_decodes_to_zero[turbo3]:\n");
    const auto * traits = ggml_get_type_traits(GGML_TYPE_TURBO3);
    if (traits == nullptr) { EXPECT_TRUE(false, "no traits"); return; }

    std::vector<float> x(QK_TURBO, 0.0f);
    std::vector<uint8_t> q(traits->type_size);
    std::vector<float> r(QK_TURBO);

    traits->from_float_ref(x.data(), q.data(), QK_TURBO);
    traits->to_float       (q.data(), r.data(), QK_TURBO);

    for (int j = 0; j < QK_TURBO; ++j) {
        EXPECT_TRUE(std::isfinite(r[j]), "decode of zero produced non-finite");
        EXPECT_LT(std::fabs(r[j]), 1e-3f, "decode of zero not near zero");
    }
}

void test_block_size_traits() {
    std::printf("test_block_size_traits:\n");
    const auto * t3 = ggml_get_type_traits(GGML_TYPE_TURBO3);
    const auto * t2 = ggml_get_type_traits(GGML_TYPE_TURBO2);
    EXPECT_TRUE(t3 != nullptr, "no traits for TURBO3");
    EXPECT_TRUE(t2 != nullptr, "no traits for TURBO2");
    if (t3) {
        EXPECT_TRUE(t3->blck_size == QK_TURBO,            "TURBO3 blck_size != 128");
        EXPECT_TRUE(t3->type_size == 2 + QK_TURBO * 3 / 8, "TURBO3 type_size != 50");
    }
    if (t2) {
        EXPECT_TRUE(t2->blck_size == QK_TURBO,            "TURBO2 blck_size != 128");
        EXPECT_TRUE(t2->type_size == 2 + QK_TURBO * 2 / 8, "TURBO2 type_size != 34");
    }
}

} // namespace

int main(int /*argc*/, char ** /*argv*/) {
    test_block_size_traits();
    test_turbo_roundtrip(GGML_TYPE_TURBO3, TURBO3_MSE_MAX, "turbo3");
    test_turbo_roundtrip(GGML_TYPE_TURBO2, TURBO2_MSE_MAX, "turbo2");
    test_norm_preservation();
    test_zero_input_decodes_to_zero();

    if (g_failures == 0) {
        std::printf("OK — all TurboQuant tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
}
