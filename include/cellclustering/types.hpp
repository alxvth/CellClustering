/*
 * Copyright (c) 2026 Alexander Vieth
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cassert>
#include <cstdint>
#include <optional>
#include <vector>

#include <boost/graph/adjacency_list.hpp>

namespace cellclustering {

    using IndexType     = std::size_t;
    using DataType      = float;
    using DistanceType  = double;
    using EdgeEntry     = std::pair<IndexType, DistanceType>;
    using EdgeList      = std::vector<EdgeEntry>;
    using IndexPair     = std::pair<IndexType, IndexType>;
    using IndexPairList = std::vector<IndexPair>;

    using BGraph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property,
                                         boost::property<boost::edge_weight_t, DistanceType>>;

    // Non-owning view of a dense, row-major (n_rows x n_cols) matrix.
    class DenseMatrixView {
    public:
        DenseMatrixView() = delete;
        explicit DenseMatrixView(const DataType* data, const std::size_t rows, const std::size_t cols)
            : dataPtr(data), nRows(rows), nCols(cols) {}

        explicit DenseMatrixView(const std::vector<DataType>& data, const std::size_t rows, const std::size_t cols)
            : dataPtr(data.data()), nRows(rows), nCols(cols) {
            assert(data.size() == rows * cols);
        }

        const DataType* data() const noexcept {
            return dataPtr;
        }
        std::size_t rows() const noexcept {
            return nRows;
        }
        std::size_t cols() const noexcept {
            return nCols;
        }
        const DataType* row(const std::size_t i) const noexcept {
            return dataPtr + i * nCols;
        }

    private:
        const DataType* dataPtr = nullptr;
        std::size_t nRows{};
        std::size_t nCols{};
    };

    // k-nearest-neighbour result, self excluded. Row-major n_rows x k.
    // Rows are ordered by increasing distance.
    struct KnnResult {
        std::size_t k = 0;
        std::vector<EdgeList> neighborList;

        std::size_t nRows() const {
            return neighborList.size();
        };
    };

    struct LouvainResult {
        std::vector<IndexType> communities; // labels 0..C-1
        double Q = 0;                       // modularity value
        int runs = 0;
    };

    struct PhenographResult {
        // One label per input data point.
        // std::numeric_limits<IndexType>::max marks outliers (members of small clusters).
        // Empty when clustering_algo == None.
        std::vector<IndexType> communities;
        // Graph used for community detection (lower triangle if undirected).
        std::optional<BGraph> graph;
        // Modularity of the (unfiltered) Louvain partition on `graph`; NaN if no clustering was run.
        double Q = 0;
    };

} // namespace cellclustering
