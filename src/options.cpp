/*
 * Copyright (c) 2026 Alexander Vieth
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cellclustering/options.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <thread>

namespace {
    using namespace cellclustering;

    // An enum can hold values outside its enumerators (e.g. static_cast<Metric>(7)).
    // The switches in neighbors.cpp have no default, so such a value would leave a builder
    // null and be dereferenced. Reject it here instead.
    bool isValid(const Metric m) noexcept {
        switch (m) {
        case Metric::Euclidean:
        case Metric::Cosine:
        case Metric::Manhattan:
            return true;
        }
        return false;
    }

    bool isValid(const NNMethod m) noexcept {
        switch (m) {
        case NNMethod::HNSW:
        case NNMethod::Annoy:
        case NNMethod::KMKNN:
        case NNMethod::BruteForce:
            return true;
        }
        return false;
    }

    bool isValid(const ClusteringAlgo a) noexcept {
        switch (a) {
        case ClusteringAlgo::Louvain:
            return true;
        }
        return false;
    }

} // namespace


namespace cellclustering {

    std::string_view toString(const Metric m) noexcept {
        switch (m) {
        case Metric::Euclidean:
            return "euclidean";
        case Metric::Cosine:
            return "cosine";
        case Metric::Manhattan:
            return "manhattan";
        }
        return "unknown";
    }

    std::string_view toString(const NNMethod m) noexcept {
        switch (m) {
        case NNMethod::HNSW:
            return "hnsw";
        case NNMethod::Annoy:
            return "annoy";
        case NNMethod::KMKNN:
            return "kmknn";
        case NNMethod::BruteForce:
            return "brute force";
        }
        return "unknown";
    }

    std::string_view toString(const ClusteringAlgo a) noexcept {
        switch (a) {
        case ClusteringAlgo::Louvain:
            return "louvain";
        }
        return "unknown";
    }

    unsigned int getNumThreads(const int nJobs) {
        const unsigned int hw = std::max(1u, std::thread::hardware_concurrency());

        if (nJobs < 0)
            return hw;

        if (nJobs == 0)
            return 1;

        return std::min(static_cast<unsigned int>(nJobs), hw);
    }

    void validate(const Options& opt) {
        const auto require = [](const bool ok, const char* msg) {
            if (!ok)
                throw std::invalid_argument(msg);
        };

        // ---- community detection ----
        require(isValid(opt.clusteringAlgo), "clusteringAlgo is not a valid ClusteringAlgo");

        const auto& lv = opt.louvainParams;
        require(std::isfinite(lv.qTol) && lv.qTol >= 0, "louvainParams.qTol must be finite and >= 0");
        require(lv.timeLimit > 0, "louvainParams.timeLimit must be > 0"); // false for NaN; +inf means no limit
        require(lv.maxRuns >= 1, "louvainParams.maxRuns must be >= 1");
        require(lv.patience >= 1, "louvainParams.patience must be >= 1");

        // directed, prune, minClusterSize, jaccard, returnGraph: every value is meaningful.
        // (prune = true overrides directed, so that combination is not an error.)

        // ---- knn ----
        require(opt.k >= 1, "k must be >= 1");
        require(isValid(opt.nnMetric), "nnMetric is not a valid Metric");
        require(isValid(opt.nnMethod), "nnMethod is not a valid NNMethod");

        require(opt.hnsw.numLinks >= 2, "hnsw.numLinks must be >= 2");
        require(opt.hnsw.efConstruction >= 1, "hnsw.efConstruction must be >= 1");
        require(opt.hnsw.efSearch >= 1, "hnsw.efSearch must be >= 1");

        require(opt.annoy.nTrees >= 1, "annoy.nTrees must be >= 1");
        // -1 selects the Annoy default; otherwise it must be a positive, finite number.
        require(opt.annoy.searchK == -1 || (std::isfinite(opt.annoy.searchK) && opt.annoy.searchK > 0),
                "annoy.searchK must be -1 (default) or a finite value > 0");

        require(std::isfinite(opt.kmknn.power) && opt.kmknn.power > 0 && opt.kmknn.power <= 1,
                "kmknn.power must be in (0, 1]");

        // ---- extras ----
        // seed: any value (or none) is valid. verbose: bool.
        // nJobs: any int is valid; see getNumThreads (< 0 all threads, 0 one thread, > 0 capped at hardware).
    }

} // namespace cellclustering
