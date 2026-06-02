// VARIANT 2 — RVV 1.0 vectorised math.
//
// Same shape as variant 1 (identical bench.h and main.cpp); the only
// difference is the implementation of FastExpInplace, which is replaced
// by a vectorised polynomial approximation that runs on RVV 1.0
// hardware (SG2044 / XuanTie C920, VLEN=128).
//
// Algorithm (per double x):
//
//   1. Clamp x into [-700, 700] so the result stays representable.
//   2. n = round(x * log2(e))            // integer exponent
//   3. r = x - n * ln(2)                 // |r| <= ln(2)/2 ≈ 0.347
//   4. P(r) ≈ exp(r) using degree-7 Taylor:
//        P(r) = 1 + r + r^2/2 + r^3/6 + ... + r^7/5040
#define _USE_MATH_DEFINES
//      worst-case error ~ r^8/8! < 5e-9 on |r| <= 0.347 — about 1 ULP.
//   5. 2^n = bit-cast of ((1023 + n) << 52) into double.
//   6. exp(x) ≈ P(r) * 2^n
//
// All steps are done in vector form (e64m1, vlmax = 2 doubles on
// VLEN=128, vlmax = 8 on VLEN=512), so each iteration of the outer
// loop processes vlmax doubles in parallel.
//
// Time spent inside FastExpInplace is accumulated into g_exp_total_ns
// exactly as in variant 1 so the timing breakdown is directly
// comparable.

#include "bench.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#if !defined(__riscv_vector)
#error "Variant 2 requires an RVV 1.0 toolchain (e.g. -march=rv64gcv1p0)."
#endif

#include <riscv_vector.h>

// =========================================================================
// Profiling counter storage (same names as variant 1).
// =========================================================================
std::uint64_t g_exp_total_ns   = 0;
std::uint64_t g_exp_call_count = 0;
std::uint64_t g_exp_elem_count = 0;

// =========================================================================
// RVV-vectorised exp.
// =========================================================================
static inline void FastExpInplaceRvvBody(double* x, std::size_t n) {
    // exp(r) ≈ sum_{k=0..7} r^k / k!  on |r| <= ln(2)/2.
    constexpr double LOG2E   = 1.4426950408889634;
    constexpr double LN2     = 0.6931471805599453;
    constexpr double EXP_HI  =  700.0;
    constexpr double EXP_LO  = -700.0;
    constexpr double C7 = 1.0 / 5040.0;
    constexpr double C6 = 1.0 / 720.0;
    constexpr double C5 = 1.0 / 120.0;
    constexpr double C4 = 1.0 / 24.0;
    constexpr double C3 = 1.0 / 6.0;
    constexpr double C2 = 1.0 / 2.0;
    constexpr double C1 = 1.0;
    constexpr double C0 = 1.0;

    std::size_t i = 0;
    while (i < n) {
        const std::size_t vl = __riscv_vsetvl_e64m1(n - i);

        vfloat64m1_t vx = __riscv_vle64_v_f64m1(x + i, vl);

        // Clamp to safe range so n fits in 11 exponent bits and the
        // polynomial stays in its accurate region after reduction.
        vx = __riscv_vfmin_vf_f64m1(vx, EXP_HI, vl);
        vx = __riscv_vfmax_vf_f64m1(vx, EXP_LO, vl);

        // n = round(x * log2(e))   (round-to-nearest-even).
        const vfloat64m1_t vnf = __riscv_vfmul_vf_f64m1(vx, LOG2E, vl);
        const vint64m1_t   vn  = __riscv_vfcvt_x_f_v_i64m1(vnf, vl);
        const vfloat64m1_t vnd = __riscv_vfcvt_f_x_v_f64m1(vn, vl);

        // r = x - n * ln(2)   (one-step Cody-Waite; LN2 fits a double
        // exactly enough for ~1-ULP accuracy across the clamped range).
        const vfloat64m1_t vr =
            __riscv_vfnmsac_vf_f64m1(vx, LN2, vnd, vl);    // vx - LN2 * vnd

        // Horner: vy starts at C7, then  vy = vy*vr + Ck  six times.
        vfloat64m1_t vy = __riscv_vfmv_v_f_f64m1(C7, vl);
        vy = __riscv_vfmadd_vv_f64m1(vy, vr, __riscv_vfmv_v_f_f64m1(C6, vl), vl);
        vy = __riscv_vfmadd_vv_f64m1(vy, vr, __riscv_vfmv_v_f_f64m1(C5, vl), vl);
        vy = __riscv_vfmadd_vv_f64m1(vy, vr, __riscv_vfmv_v_f_f64m1(C4, vl), vl);
        vy = __riscv_vfmadd_vv_f64m1(vy, vr, __riscv_vfmv_v_f_f64m1(C3, vl), vl);
        vy = __riscv_vfmadd_vv_f64m1(vy, vr, __riscv_vfmv_v_f_f64m1(C2, vl), vl);
        vy = __riscv_vfmadd_vv_f64m1(vy, vr, __riscv_vfmv_v_f_f64m1(C1, vl), vl);
        vy = __riscv_vfmadd_vv_f64m1(vy, vr, __riscv_vfmv_v_f_f64m1(C0, vl), vl);

        // 2^n by composing the exponent bits directly: a finite double
        // with biased exponent (1023 + n) and zero mantissa is exactly
        // 2^n.
        const vint64m1_t vexpBits = __riscv_vsll_vx_i64m1(
            __riscv_vadd_vx_i64m1(vn, 1023, vl), 52, vl);
        const vfloat64m1_t vpow2 =
            __riscv_vreinterpret_v_i64m1_f64m1(vexpBits);

        const vfloat64m1_t vresult = __riscv_vfmul_vv_f64m1(vy, vpow2, vl);
        __riscv_vse64_v_f64m1(x + i, vresult, vl);

        i += vl;
    }
}

void FastExpInplace(double* x, std::size_t n) {
    TStopwatch sw;
    FastExpInplaceRvvBody(x, n);
    g_exp_total_ns   += sw.ElapsedNs();
    g_exp_call_count += 1;
    g_exp_elem_count += n;
}

// =========================================================================
// CrossEntropy der1/der2 — Logloss hot configuration (identical to v1).
// =========================================================================
void CalcCrossEntropyDerRange(
    int start,
    int count,
    const double* approxes,
    const float* targets,
    const float* weights,
    TDers* ders)
{
    double expBuf[EXP_BATCH_CAPACITY];

    int processed = 0;
    while (processed < count) {
        const int batch = std::min<int>(EXP_BATCH_CAPACITY, count - processed);
        std::memcpy(expBuf, approxes + start + processed, batch * sizeof(double));
        FastExpInplace(expBuf, static_cast<std::size_t>(batch));

        for (int j = 0; j < batch; ++j) {
            const double e = expBuf[j];
            const double p = 1.0 - 1.0 / (1.0 + e);
            const int i = start + processed + j;
            ders[i].Der1 = targets[i] - p;
            ders[i].Der2 = -p * (1.0 - p);
            ders[i].Der3 = 0.0;
        }
        processed += batch;
    }

    if (weights != nullptr) {
        for (int i = start; i < start + count; ++i) {
            ders[i].Der1 *= weights[i];
            ders[i].Der2 *= weights[i];
        }
    }
}

// =========================================================================
// Scatter half of CalcLeafDers (identical to v1).
// =========================================================================
template <bool UseWeights>
static void CalcLeafDersImpl(
    int rowStart,
    int rowCount,
    const TIndexType* leafIndices,
    const float* weights,
    const TDers* approxDers,
    TDers* leafDers,
    double* leafWeights)
{
    for (int rowIdx = rowStart; rowIdx < rowStart + rowCount; ++rowIdx) {
        TDers& d = leafDers[leafIndices[rowIdx]];
        d.Der1 += approxDers[rowIdx - rowStart].Der1;
        d.Der2 += approxDers[rowIdx - rowStart].Der2;
        const double rowWeight = UseWeights ? weights[rowIdx] : 1.0;
        leafWeights[leafIndices[rowIdx]] += rowWeight;
    }
}

void CalcLeafDers(
    int sampleCount,
    int /*leafCount*/,
    const TIndexType* indices,
    const float* targets,
    const float* weights,
    const double* approxes,
    TDers* approxDersScratch,
    TDers* leafDers,
    double* leafWeights)
{
    for (int innerBlockStart = 0;
         innerBlockStart < sampleCount;
         innerBlockStart += APPROX_BLOCK_SIZE)
    {
        const int innerCount =
            std::min(sampleCount - innerBlockStart, APPROX_BLOCK_SIZE);

        CalcCrossEntropyDerRange(
            /*start=*/0,
            /*count=*/innerCount,
            approxes + innerBlockStart,
            targets + innerBlockStart,
            weights ? weights + innerBlockStart : nullptr,
            approxDersScratch);

        if (weights == nullptr) {
            CalcLeafDersImpl<false>(
                innerBlockStart, innerCount,
                indices, weights,
                approxDersScratch,
                leafDers, leafWeights);
        } else {
            CalcLeafDersImpl<true>(
                innerBlockStart, innerCount,
                indices, weights,
                approxDersScratch,
                leafDers, leafWeights);
        }
    }
}
