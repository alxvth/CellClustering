/*
 * Copyright (c) 2026 Alexander Vieth
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef CELLCLUSTERING_OPTIONS_HPP
#define CELLCLUSTERING_OPTIONS_HPP

#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>

namespace cellclustering {

    // Distance used to define nearest neighbours
    enum class Metric { Euclidean, Cosine, Manhattan };

    // Nearest-neighbour backend
    enum class NNMethod { HNSW, Annoy, KMKNN, BruteForce };

    // Community detection algorithm
    // TODO: add Leiden
    enum class ClusteringAlgo { Louvain };

    std::string_view toString(Metric m) noexcept;
    std::string_view toString(NNMethod m) noexcept;
    std::string_view toString(ClusteringAlgo a) noexcept;

    struct HnswParams {
        int numLinks       = 16; // equivalent to the `M` parameter
        int efConstruction = 200;
        int efSearch       = 10;
    };

    struct AnnoyParams {
        int nTrees = 50;
        // -1 = Annoy default (n_neighbours * n_trees).
        double searchK = -1;
    };

    struct KMKNNParams {
        double power = 0.5; // Power of the number of observations, to define the number of cluster centers
    };

    struct LouvainParams {
        double qTol      = 1e-3;
        double timeLimit = 2000; // seconds
        int maxRuns      = 100;
        int patience     = 20;
    };

    // Main settings
    struct Options {
        // ---- Community detection ----
        ClusteringAlgo clusteringAlgo = ClusteringAlgo::Louvain;
        LouvainParams louvainParams{};
        bool directed              = false; // keep the asymmetric kNN graph
        bool prune                 = false; // symmetrize with product instead of average (forces directed=false)
        std::size_t minClusterSize = 10;    // clusters with size <= this get label std::numeric_limits<size_t>::max
        bool jaccard               = true;  // Jaccard weights; false = Gaussian kernel (sigma = 1)
        bool returnGraph           = true;

        // ---- knn ----
        int k             = 30; // neighbours in the first graph-construction step
        Metric nnMetric   = Metric::Euclidean;
        NNMethod nnMethod = NNMethod::HNSW;
        HnswParams hnsw{};
        AnnoyParams annoy{};
        KMKNNParams kmknn{};

        // ---- extras ----
        std::optional<std::uint64_t> seed; // seeds ANN builds and Louvain restarts
        bool verbose = false;              // progress messages on stderr
        int nJobs    = -1;                 // -1 all cores; < -1: n_cpus + 1 + n_jobs
    };

    constexpr std::size_t NOT_ASSIGNED = std::numeric_limits<std::size_t>::max();

    // nJobs    Behavior
    // < 0	    Use all available hardware threads
    // == 0     Use a single thread
    // > 0      Use the smaller of nJobs and available hardware threads
    unsigned int getNumThreads(int nJobs);

    // Throws std::invalid_argument on inconsistent settings.
    void validate(const Options& opt);

} // namespace cellclustering

#endif // CELLCLUSTERING_OPTIONS_HPP
