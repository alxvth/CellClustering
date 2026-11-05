/*
 * Copyright (c) 2026 Alexander Vieth
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "cellclustering/options.hpp"
#include "cellclustering/types.hpp"

namespace cellclustering {

    // Repeated randomized Louvain (Boost.Graph `louvain_clustering`), keeping the best
    // partition. Same restart policy as the reference `runlouvain`: a restart is accepted if
    // it beats the best Q by more than q_tol; stop after `patience` runs without
    // acceptance, `max_runs` runs, or `time_limit` seconds. Each stored entry of `graph`
    // is an undirected edge (parallel entries add up). Weights must be >= 0.
    LouvainResult runLouvain(const BGraph& graph, const Options& opt);

} // namespace cellclustering
