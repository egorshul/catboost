// Common types, timers and kernel declarations for the CalcLeafDers
// microbenchmark.  Identical between the base (scalar) and the RVV
// variants — only kernel.cpp differs.

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>

// Mirror of catboost's TDers (third derivative unused for Logloss but kept
// for layout-faithfulness).
struct TDers {
    double Der1;
    double Der2;
    double Der3;
};

// Mirror of catboost's TIndexType (leaf id).
using TIndexType = std::uint32_t;

// Matches catboost::APPROX_BLOCK_SIZE.
constexpr int APPROX_BLOCK_SIZE = 500;

// Capacity of catboost's TExpForwardView<16> — exp is materialised in
// chunks of 16 doubles.
constexpr int EXP_BATCH_CAPACITY = 16;

// =========================================================================
// Profiling counters.  Each FastExpInplace call adds to these globals so
// that main() can subtract them from the full-function elapsed time.
// =========================================================================
extern std::uint64_t g_exp_total_ns;
extern std::uint64_t g_exp_call_count;
extern std::uint64_t g_exp_elem_count;

inline void ResetExpCounters() {
    g_exp_total_ns = 0;
    g_exp_call_count = 0;
    g_exp_elem_count = 0;
}

// =========================================================================
// Kernel API (implementation lives in kernel.cpp — that's the only file
// that differs between the two variants).
// =========================================================================

// In-place vector exp.  For variant 1 this is a scalar std::exp loop; for
// variant 2 it is the RVV-vectorised polynomial approximation.  Time spent
// here is accumulated into g_exp_total_ns.
void FastExpInplace(double* x, std::size_t n);

// CrossEntropy (Logloss) der1/der2 over [start, start+count).  Mirrors
// catboost's CalcCrossEntropyDerRangeImpl<false, true, false, false> — the
// hot configuration for the user's bench (Logloss, !UseExpApprox,
// !HasDelta).
void CalcCrossEntropyDerRange(
    int start,
    int count,
    const double* approxes,
    const float* targets,
    const float* weights,         // may be nullptr
    TDers* ders);

// Single-threaded equivalent of catboost's CalcLeafDers for the same hot
// configuration.  Returns through `leafDers` / `leafWeights` the
// per-leaf sums of Der1, Der2 and rowWeight.  `approxDersScratch` must
// hold at least APPROX_BLOCK_SIZE elements.
void CalcLeafDers(
    int sampleCount,
    int leafCount,
    const TIndexType* indices,
    const float* targets,
    const float* weights,         // may be nullptr
    const double* approxes,
    TDers* approxDersScratch,
    TDers* leafDers,
    double* leafWeights);

// Convenience wall-clock helper.
struct TStopwatch {
    using Clock = std::chrono::steady_clock;
    Clock::time_point Start = Clock::now();
    std::uint64_t ElapsedNs() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - Start).count();
    }
};
