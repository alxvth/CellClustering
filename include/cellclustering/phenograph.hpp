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

    // PhenoGraph on a dense n x d matrix.
    PhenographResult phenograph(const DenseMatrixView& data, const Options& opt = {});

    // PhenoGraph from an already computed KnnResult.
    PhenographResult phenograph(const KnnResult& knn, const Options& opt = {});

    // Relabel clusters by descending size (labels 0..). Clusters whose size is
    // <= min_size become -1 (this mirrors the reference code, whose docstring says "smaller").
    std::vector<IndexType> sortBySize(const std::vector<IndexType>& clusters, const IndexType minSize);

} // namespace cellclustering
