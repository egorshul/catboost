// CalcLeafDers benchmark driver — generates synthetic Logloss inputs of
// the size used by catboost's CalcLeafDers (5 M docs × 256 leaves at
// depth 8) and reports:
//
//   * total elapsed time inside CalcLeafDers,
//   * total elapsed time inside the FastExpInplace calls it makes
//     (accumulated through the g_exp_total_ns counter),
//   * the implied "everything else" (scatter + sigmoid + der1/der2)
//     budget.
//
// Build (cross to RISC-V; same command for both variants):
//   riscv64-linux-gnu-g++ -march=rv64gcv1p0 -std=c++17 -O3 -static \
//     bench.h kernel.cpp main.cpp -o bench
//
// Run on SG2044 directly, or under QEMU:
//   qemu-riscv64 -cpu rv64,v=true,vlen=128,elen=64,vext_spec=v1.0 ./bench

#include "bench.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

struct TDataset {
    int SampleCount;
    int LeafCount;
    std::vector<TIndexType> Indices;
    std::vector<float>      Targets;
    std::vector<float>      Weights;     // empty == "no weights"
    std::vector<double>     Approxes;
};

TDataset GenerateDataset(int sampleCount, int leafCount, bool useWeights, unsigned seed) {
    TDataset ds;
    ds.SampleCount = sampleCount;
    ds.LeafCount   = leafCount;
    ds.Indices.resize(sampleCount);
    ds.Targets.resize(sampleCount);
    ds.Approxes.resize(sampleCount);
    if (useWeights) {
        ds.Weights.resize(sampleCount);
    }

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> leafDist(0, leafCount - 1);
    std::uniform_real_distribution<double> approxDist(-4.0, 4.0);
    std::bernoulli_distribution targetDist(0.3);
    std::uniform_real_distribution<float> weightDist(0.1f, 2.0f);

    for (int i = 0; i < sampleCount; ++i) {
        ds.Indices[i]  = static_cast<TIndexType>(leafDist(rng));
        ds.Targets[i]  = targetDist(rng) ? 1.0f : 0.0f;
        ds.Approxes[i] = approxDist(rng);
        if (useWeights) {
            ds.Weights[i] = weightDist(rng);
        }
    }
    return ds;
}

double Sum(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s;
}

double SumDer1(const std::vector<TDers>& v) {
    double s = 0.0;
    for (const auto& d : v) s += d.Der1;
    return s;
}

double SumDer2(const std::vector<TDers>& v) {
    double s = 0.0;
    for (const auto& d : v) s += d.Der2;
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    // Defaults sized to look like the user's bench (depth 8 → 256 leaves,
    // 5 M docs per CalcLeafDers call) but configurable from the CLI.
    int sampleCount = 5'000'000;
    int leafCount   = 256;
    int repeats     = 3;
    bool useWeights = false;
    if (argc > 1) sampleCount = std::atoi(argv[1]);
    if (argc > 2) leafCount   = std::atoi(argv[2]);
    if (argc > 3) repeats     = std::atoi(argv[3]);
    if (argc > 4) useWeights  = (std::atoi(argv[4]) != 0);

    std::printf("Config: sampleCount=%d, leafCount=%d, useWeights=%d, repeats=%d\n\n",
        sampleCount, leafCount, (int)useWeights, repeats);

    const TDataset ds = GenerateDataset(
        sampleCount, leafCount, useWeights, /*seed=*/0xC4'7B'00'57u);

    std::vector<TDers>  approxDersScratch(APPROX_BLOCK_SIZE);
    std::vector<TDers>  leafDers(leafCount);
    std::vector<double> leafWeights(leafCount);

    // Warm-up run (cache, page faults, dynamic-linker work).
    std::fill(leafDers.begin(),    leafDers.end(),    TDers{0, 0, 0});
    std::fill(leafWeights.begin(), leafWeights.end(), 0.0);
    ResetExpCounters();
    CalcLeafDers(
        sampleCount, leafCount,
        ds.Indices.data(),
        ds.Targets.data(),
        useWeights ? ds.Weights.data() : nullptr,
        ds.Approxes.data(),
        approxDersScratch.data(),
        leafDers.data(),
        leafWeights.data());

    const double checksumDer1   = SumDer1(leafDers);
    const double checksumDer2   = SumDer2(leafDers);
    const double checksumWeight = Sum(leafWeights);
    std::printf("Warmup checksums: sum(Der1)=%.6f  sum(Der2)=%.6f  sum(weight)=%.6f\n\n",
        checksumDer1, checksumDer2, checksumWeight);

    std::puts("repeat    full(ms)     exp(ms)   exp-calls   exp-elems   other(ms)");
    std::puts("------    --------   ---------   ---------   ---------   ---------");

    std::uint64_t bestFullNs = ~0ull;
    std::uint64_t bestExpNs  = ~0ull;

    for (int r = 0; r < repeats; ++r) {
        std::fill(leafDers.begin(),    leafDers.end(),    TDers{0, 0, 0});
        std::fill(leafWeights.begin(), leafWeights.end(), 0.0);
        ResetExpCounters();

        TStopwatch sw;
        CalcLeafDers(
            sampleCount, leafCount,
            ds.Indices.data(),
            ds.Targets.data(),
            useWeights ? ds.Weights.data() : nullptr,
            ds.Approxes.data(),
            approxDersScratch.data(),
            leafDers.data(),
            leafWeights.data());
        const std::uint64_t fullNs = sw.ElapsedNs();

        if (fullNs < bestFullNs) bestFullNs = fullNs;
        if (g_exp_total_ns < bestExpNs) bestExpNs = g_exp_total_ns;

        const std::uint64_t otherNs =
            fullNs > g_exp_total_ns ? fullNs - g_exp_total_ns : 0;

        std::printf("%6d  %10.3f  %10.3f  %10" PRIu64 "  %10" PRIu64 "  %10.3f\n",
            r,
            fullNs           / 1.0e6,
            g_exp_total_ns   / 1.0e6,
            g_exp_call_count,
            g_exp_elem_count,
            otherNs          / 1.0e6);
    }

    std::printf("\nBest:   full=%.3f ms   exp=%.3f ms   (exp %.1f%% of full)\n",
        bestFullNs / 1.0e6,
        bestExpNs  / 1.0e6,
        bestFullNs ? (100.0 * bestExpNs / bestFullNs) : 0.0);

    return 0;
}
