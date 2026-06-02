// VARIANT 1 — BASE.
//
// Mirrors what catboost runs on a stock RISC-V build for the Logloss hot
// path inside CalcLeafDers:
//
//   * FastExpInplace falls through to libm's scalar std::exp, exactly as
//     catboost/libs/helpers/math_utils.cpp does when neither AVX2 nor
//     SSE2 is detected.
//   * CalcCrossEntropyDerRange replays the (!UseExpApprox, !HasDelta,
//     UseTDers, !CalcThirdDer) instantiation of
//     CalcCrossEntropyDerRangeImpl from
//     catboost/private/libs/algo_helpers/error_functions.cpp — including
//     the 16-element TExpForwardView batching of the exp call.
//   * CalcLeafDers wires the two together with the same APPROX_BLOCK_SIZE
//     inner blocking that catboost's approx_calcer.cpp uses, only
//     stripped of the multi-thread MapMerge step that the benchmark
//     does not need.
//
// Each FastExpInplace call accumulates wall-clock time into the
// g_exp_total_ns counter so that main() can break down the total
// CalcLeafDers cost into "exp" and "everything else".

#include "bench.h"

#include <algorithm>
#include <cmath>
#include <cstring>

// =========================================================================
// Profiling counter storage.
// =========================================================================
std::uint64_t g_exp_total_ns   = 0;
std::uint64_t g_exp_call_count = 0;
std::uint64_t g_exp_elem_count = 0;

// =========================================================================
// Scalar reference exp — what the RISC-V build of catboost uses today.
// =========================================================================
void FastExpInplace(double* x, std::size_t n) {
    TStopwatch sw;
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = std::exp(x[i]);
    }
    g_exp_total_ns   += sw.ElapsedNs();
    g_exp_call_count += 1;
    g_exp_elem_count += n;
}

// =========================================================================
// CrossEntropy der1/der2 — Logloss hot configuration.
// =========================================================================
void CalcCrossEntropyDerRange(
    int start,
    int count,
    const double* approxes,
    const float* targets,
    const float* weights,
    TDers* ders)
{
    // Re-create the lazy 16-element exp view: copy a chunk of approxes,
    // call FastExpInplace, consume.  This is exactly what
    // TExpForwardView<16> does in catboost.
    double expBuf[EXP_BATCH_CAPACITY];

    int processed = 0;
    while (processed < count) {
        const int batch = std::min<int>(EXP_BATCH_CAPACITY, count - processed);
        std::memcpy(expBuf, approxes + start + processed, batch * sizeof(double));
        FastExpInplace(expBuf, static_cast<std::size_t>(batch));

        for (int j = 0; j < batch; ++j) {
            const double e = expBuf[j];
            const double p = 1.0 - 1.0 / (1.0 + e);          // sigmoid
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
// Scatter half of CalcLeafDers — Der1/Der2 into per-leaf accumulators.
// Faithful to CalcLeafDersImpl in approx_calcer.cpp.
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

// =========================================================================
// Single-thread CalcLeafDers — mirrors catboost approx_calcer.cpp:164 but
// without the outer ExecRange.
// =========================================================================
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

        // Math (exp + sigmoid + d1/d2) — this is where ~44% of the time
        // lives in the user's profile.
        CalcCrossEntropyDerRange(
            /*start=*/0,
            /*count=*/innerCount,
            approxes + innerBlockStart,
            targets + innerBlockStart,
            weights ? weights + innerBlockStart : nullptr,
            approxDersScratch);

        // Scatter — the other ~44%.
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
