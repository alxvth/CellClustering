/*
 * Copyright (c) 2026 Alexander Vieth
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cellclustering/phenograph.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <map>

#include "cellclustering/graph.hpp"
#include "cellclustering/louvain.hpp"
#include "cellclustering/neighbors.hpp"

namespace cellclustering {

    // Relabel clusters by descending size (labels 0..). Clusters whose size is
    // <= min_size become -1 (this mirrors the reference code, whose docstring says "smaller").
    std::vector<IndexType> sortBySize(const std::vector<IndexType>& clusters, const IndexType minSize) {
        std::map<IndexType, IndexType> sizes;
        for (auto c : clusters)
            ++sizes[c];

        std::vector<std::pair<IndexType, IndexType>> order(sizes.begin(), sizes.end());
        std::ranges::stable_sort(order, [](const auto& a, const auto& b) { return a.second > b.second; });

        std::map<IndexType, IndexType> relabel;
        IndexType next = 0;
        for (auto& [label, size] : order)
            relabel[label] = size > minSize ? next++ : std::numeric_limits<IndexType>::max();

        std::vector<IndexType> out(clusters.size());
        for (IndexType i = 0; i < clusters.size(); ++i)
            out[i] = relabel[clusters[i]];

        return out;
    }

    PhenographResult phenograph(const KnnResult& knn, const Options& in_opt) {
        using clock = std::chrono::steady_clock;
        validate(in_opt);
        Options opt = in_opt;
        if (opt.prune)
            opt.directed = false; // prune takes precedence
        auto log = [&](const char* what, clock::time_point since) {
            if (opt.verbose)
                std::cout << what << " in " << std::chrono::duration<double>(clock::now() - since).count()
                          << " seconds\n";
        };

        auto tic = clock::now();
        BGraph graph = opt.jaccard ? jaccardKernel(knn, opt.nJobs) : gaussianKernel(knn, 1.0);
        log(opt.jaccard ? "Jaccard graph constructed" : "Gaussian kernel graph constructed", tic);
        if (!opt.directed)
            graph = symmetrize(graph, opt.prune);

        PhenographResult res;
        res.Q = std::numeric_limits<double>::quiet_NaN();
        if (opt.clusteringAlgo == ClusteringAlgo::Louvain) {
            tic = clock::now();
            const LouvainResult lr = runLouvain(graph, opt);
            log("Louvain completed", tic);
            res.communities = sortBySize(lr.communities, opt.minClusterSize);
            res.Q = lr.Q;
        }

        if (opt.returnGraph) {
            res.graph = std::move(graph);
        }

        return res;
    }

    PhenographResult phenograph(const DenseMatrixView& data, const Options& opt) {
        validate(opt);
        const auto tic = std::chrono::steady_clock::now();
        const KnnResult knn = findNeighbors(data, opt);
        if (opt.verbose)
            std::cerr << "Neighbors computed in "
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - tic).count() << " seconds\n";
        return phenograph(knn, opt);
    }

} // namespace cellclustering
