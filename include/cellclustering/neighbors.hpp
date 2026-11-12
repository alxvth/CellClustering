/*
 * Copyright (c) 2026 Alexander Vieth
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef CELLCLUSTERING_NEIGHBORS_HPP
#define CELLCLUSTERING_NEIGHBORS_HPP

#include "cellclustering/options.hpp"
#include "cellclustering/types.hpp"

namespace cellclustering {

    // Find the k nearest neighbours (self excluded),
    KnnResult findNeighbors(const DenseMatrixView& data, const Options& opt);

} // namespace cellclustering

#endif // CELLCLUSTERING_NEIGHBORS_HPP
