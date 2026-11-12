/*
 * Copyright (c) 2026 Alexander Vieth
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef CELLCLUSTERING_GRAPH_HPP
#define CELLCLUSTERING_GRAPH_HPP

#include "cellclustering/types.hpp"

namespace cellclustering {

    // Jaccard coefficient |N(i) INTERSECT N(j)| / |N(i) UNION N(j)| for every kNN edge i->j.
    // Entries with zero overlap are dropped; output is row-major in neighbour order.
    BGraph jaccardKernel(const KnnResult& knn, int nJobs = -1);

    // Gaussian kernel of the kNN distances: exp(-0.5 (d/sigma)^2) / (sigma sqrt(2 pi)).
    // Only positive sigma are valid, this implementation chooses to return BGraph(n) otherwises
    BGraph gaussianKernel(const KnnResult& knn, double sigma = 1.0);

    // Symmetrize a directed graph and keep the strict lower triangle (row > col),
    // sorted by (row, col). prune=false: (G + G^T) / 2. prune=true: G ∘ G^T.
    BGraph symmetrize(const BGraph& graph, bool prune);

} // namespace cellclustering

#endif // __CELLCLUSTERING_GRAPH_HPP
