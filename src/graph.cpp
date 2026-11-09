/*
 * Copyright (c) 2026 Alexander Vieth
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cellclustering/graph.hpp"

#include "cellclustering/options.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace {
    using namespace cellclustering;

    // Builds the graph in one shot using the (first, last, weights, n) constructor.
    BGraph makeGraph(const std::size_t n, const IndexPairList& edges, const std::vector<DistanceType>& w) {
        return {edges.begin(), edges.end(), w.begin(), n};
    }

    template <typename S, typename T> bool isEqual(const S x, const T y) {
        constexpr double absEpsilon = 1e-12;
        constexpr double relEpsilon = 1e-5;

        const double dx = static_cast<double>(x);
        const double dy = static_cast<double>(y);

        return std::abs(dx - dy) <= std::max(absEpsilon, relEpsilon * std::max(std::abs(dx), std::abs(dy)));
    }
} // namespace

namespace cellclustering {

    BGraph jaccardKernel(const KnnResult& knn, int nJobs) {
        const std::size_t n = knn.nRows();
        const double twoK = 2.0 * static_cast<double>(knn.k);

        if (isEqual(n, 0) || isEqual(twoK, 0))
            return BGraph(n);

        // Check all points have k neighbors
        assert(knn.neighborList.front().size() == knn.k);
        assert(std::ranges::all_of(knn.neighborList,
                                   [&](const auto& row) { return row.size() == knn.neighborList.front().size(); }));

        const unsigned int numThreads = getNumThreads(nJobs);
        const std::size_t total = n * knn.k;

        // Sorte, neighbour sets
        std::vector<std::vector<IndexType>> neighborsSets(n);
#pragma omp parallel for num_threads(numThreads) schedule(dynamic, 1)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
            const auto& neighbors = knn.neighborList[i];
            neighborsSets[i].reserve(knn.k);
            for (auto const& j : std::views::keys(neighbors)) {
                neighborsSets[i].push_back(j);
            }
            std::ranges::sort(neighborsSets[i]);
        }

        // Check all sets have k neighbors
        assert(neighborsSets.front().size() == knn.k);
        assert(std::ranges::all_of(neighborsSets,
                                   [&](const auto& row) { return row.size() == neighborsSets.front().size(); }));

        // Compute Jaccard similarity
        std::vector<DistanceType> s(total);

#pragma omp parallel for num_threads(numThreads) schedule(dynamic, 1)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
            const auto& a = neighborsSets[i];
            const auto& row = knn.neighborList[i];

            for (std::size_t j = 0; j < row.size(); ++j) {
                const auto& b = neighborsSets[static_cast<std::size_t>(row[j].first)];

                // Compute |N(i) INTERSECT N(j)|
                std::size_t shared = 0;
                for (auto p = a.begin(), q = b.begin(); p != a.end() && q != b.end();) {
                    if (*p < *q)
                        ++p;
                    else if (*q < *p)
                        ++q;
                    else {
                        ++shared;
                        ++p;
                        ++q;
                    }
                }

                // |N(i) INTERSECT N(j)| / |N(i) UNION N(j)| = |N(i) INTERSECT N(j)| / (2k - |N(i) INTERSECT N(j)|)
                s[i * knn.k + j] = static_cast<double>(shared) / (twoK - static_cast<double>(shared));
            }
        }

        IndexPairList edges;
        std::vector<DistanceType> w;
        edges.reserve(total);
        w.reserve(total);
        for (std::size_t i = 0; i < n; ++i) {
            const auto& row = knn.neighborList[i];
            for (std::size_t j = 0; j < row.size(); ++j) {
                if (const DistanceType v = s[i * knn.k + j]; v > 0) {
                    edges.emplace_back(i, row[j].first);
                    w.push_back(v);
                }
            }
        }

        return makeGraph(n, edges, w);
    }

    BGraph gaussianKernel(const KnnResult& knn, double sigma) {
        const std::size_t n = knn.nRows();

        if (isEqual(n, 0) || !(sigma > 0))
            return BGraph(n);

        const double norm = 1.0 / (sigma * std::sqrt(2.0 * std::numbers::pi_v<double>));

        // Check all points have k neighbors
        assert(knn.neighborList.front().size() == knn.k);
        assert(std::ranges::all_of(knn.neighborList,
                                   [&](const auto& row) { return row.size() == knn.neighborList.front().size(); }));

        const std::size_t total = n * knn.k;

        IndexPairList edges;
        std::vector<DistanceType> w;
        edges.reserve(total);
        w.reserve(total);
        for (std::size_t i = 0; i < n; ++i) {
            for (const auto& [j, dist] : knn.neighborList[i]) {
                const double x = dist / sigma;
                edges.emplace_back(static_cast<IndexType>(i), j);
                w.push_back(norm * std::exp(-0.5 * x * x));
            }
        }

        return makeGraph(n, edges, w);
    }

    BGraph symmetrize(const BGraph& g, bool prune) {
        struct Entry {
            IndexType hi, lo;
            bool lower;
            DistanceType w;
        };
        std::vector<Entry> e;
        e.reserve(num_edges(g));

        const auto wmap = get(boost::edge_weight, g);
        for (auto [it, end] = boost::edges(g); it != end; ++it) {
            // source()/target() reflect insertion order, i.e. original i -> j
            const IndexType i = boost::source(*it, g);
            const IndexType j = boost::target(*it, g);
            if (i == j)
                continue;
            e.push_back(i > j ? Entry{i, j, true, wmap[*it]} : Entry{j, i, false, wmap[*it]});
        }
        std::ranges::sort(e, [](const Entry& a, const Entry& b) { return a.hi != b.hi ? a.hi < b.hi : a.lo < b.lo; });

        IndexPairList edges;
        std::vector<DistanceType> w;
        for (std::size_t p = 0; p < e.size();) {
            std::size_t q = p;
            DistanceType lower = 0, upper = 0;
            for (; q < e.size() && e[q].hi == e[p].hi && e[q].lo == e[p].lo; ++q)
                (e[q].lower ? lower : upper) += e[q].w;
            const DistanceType val = prune ? lower * upper : 0.5 * (lower + upper);
            if (!isEqual(val, 0)) {
                edges.emplace_back(e[p].hi, e[p].lo);
                w.push_back(val);
            }
            p = q;
        }
        return makeGraph(boost::num_vertices(g), edges, w);
    }

} // namespace cellclustering