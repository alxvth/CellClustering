/*
 * Copyright (c) 2026 Alexander Vieth
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cellclustering/louvain.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/louvain_clustering.hpp>

#include <chrono>
#include <random>

namespace cellclustering {

    constexpr double kMinImprovement = 1e-6;

    LouvainResult runLouvain(const BGraph& graph, const Options& opt) {
        using clock = std::chrono::steady_clock;
        const auto tic = clock::now();
        const std::size_t n = boost::num_vertices(graph);

        LouvainResult best;
        const auto& params = opt.louvainParams;

         // modularity is undefined; every vertex is alone
         if (boost::num_edges(graph) == 0) {
            best.communities.resize(n);
            for (std::size_t i = 0; i < n; ++i)
                best.communities[i] = i;
            return best;
        }

        std::mt19937_64 master(opt.seed ? *opt.seed : std::random_device{}());
        const auto weights = boost::get(boost::edge_weight, graph);
        std::vector<IndexType> comm(n);

        double qRef = 0.0; // best accepted Q (reference starts at 0)
        int run = 0, updated = 0;
        auto elapsed = [&] { return std::chrono::duration<double>(clock::now() - tic).count(); };
        // The first run is unconditional so a result always exists.
        while (run == 0 ||
               (run - updated < params.patience && run < params.maxRuns && elapsed() < params.timeLimit)) {
            std::mt19937_64 gen(master());
            auto cmap = boost::make_iterator_property_map(comm.begin(), boost::get(boost::vertex_index, graph));
            double q = boost::louvain_clustering(graph, cmap, weights, gen, boost::newman_and_girvan{}, kMinImprovement,
                                                 kMinImprovement);
            ++run;
            if (best.communities.empty()) { // always keep something, even if q <= q_tol
                best.communities.assign(comm.begin(), comm.end());
                best.Q = q;
            }
            if (q - qRef > params.qTol) {
                qRef = q;
                updated = run;
                best.communities.assign(comm.begin(), comm.end());
                best.Q = q;
            }
        }
        best.runs = run;
        return best;
    }

} // namespace cellclustering
