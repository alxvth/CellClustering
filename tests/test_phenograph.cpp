#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <numbers>
#include <tuple>
#include <vector>

#include "cellclustering/graph.hpp"
#include "cellclustering/phenograph.hpp"
#include "cellclustering/types.hpp"

#include "test_utils.hpp"

using Catch::Approx;
using namespace cellclustering;

namespace {

    using WEdge = std::tuple<IndexType, IndexType, DistanceType>; // (source, target, weight)

    // Edges in the order the graph iterates them (= insertion order for vecS edge lists).
    std::vector<WEdge> edgesInOrder(const BGraph& g) {
        std::vector<WEdge> out;
        const auto wmap = get(boost::edge_weight, g);
        for (auto [it, end] = boost::edges(g); it != end; ++it)
            out.emplace_back(boost::source(*it, g), boost::target(*it, g), wmap[*it]);
        return out;
    }

    std::vector<WEdge> sorted(std::vector<WEdge> e) {
        std::ranges::sort(e);
        return e;
    }

    BGraph buildGraph(const std::size_t n, const std::vector<WEdge>& es) {
        IndexPairList pairs;
        std::vector<DistanceType> w;
        for (const auto& [a, b, x] : es) {
            pairs.emplace_back(a, b);
            w.push_back(x);
        }
        return BGraph(pairs.begin(), pairs.end(), w.begin(), n);
    }

    // Order-independent comparison (parallel edges are allowed, so compare sorted lists).
    void requireSameEdges(const std::vector<WEdge>& actual, const std::vector<WEdge>& expected, const double margin = 1e-9) {
        const auto a = sorted(actual);
        const auto e = sorted(expected);
        REQUIRE(a.size() == e.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            INFO("edge index " << i);
            CHECK(std::get<0>(a[i]) == std::get<0>(e[i]));
            CHECK(std::get<1>(a[i]) == std::get<1>(e[i]));
            CHECK(std::get<2>(a[i]) == Approx(std::get<2>(e[i])).margin(margin));
        }
    }

    KnnResult makeKnn(const std::size_t k, std::vector<EdgeList> rows) {
        KnnResult r;
        r.k = k;
        r.neighborList = std::move(rows);
        return r;
    }

    double gauss(const double d, const double sigma) {
        const double x = d / sigma;
        return std::exp(-0.5 * x * x) / (sigma * std::sqrt(2.0 * std::numbers::pi_v<double>));
    }

} // namespace

TEST_CASE("jaccardKernel reproduces the reference affinity graph", "[core][jaccard]") {
    const auto nJobs = GENERATE(1, 4);
    const BGraph g = jaccardKernel(test::fixedKnn(), nJobs);

    const auto affinityValuesSize = test::affinityValues.size();

    REQUIRE(test::affinityRow.size() == affinityValuesSize);
    REQUIRE(test::affinityCol.size() == affinityValuesSize);
    REQUIRE(boost::num_edges(g) == affinityValuesSize);

    std::vector<WEdge> expected;
    expected.reserve(affinityValuesSize);
    for (std::size_t t = 0; t < affinityValuesSize; ++t)
        expected.emplace_back(static_cast<IndexType>(test::affinityRow[t]),
                              static_cast<IndexType>(test::affinityCol[t]), test::affinityValues[t]);

    requireSameEdges(edgesInOrder(g), expected, 1e-7);
}

TEST_CASE("jaccardKernel drops zero overlap and de-duplicates neighbour sets", "[core][jaccard]") {
    // 0 <-> 1 share {2}; 2 and 3 point only at each other / duplicate entries.
    const auto knn = makeKnn(2, {
                                    {{1, 0.0}, {2, 0.0}}, // N(0) = {1, 2}
                                    {{0, 0.0}, {2, 0.0}}, // N(1) = {0, 2}
                                    {{3, 0.0}, {3, 0.0}}, // N(2) = {3}   after de-duplication
                                    {{2, 0.0}, {2, 0.0}}, // N(3) = {2}   after de-duplication
                                });
    const BGraph g = jaccardKernel(knn);
    CHECK(boost::num_vertices(g) == 4);

    const auto edges = edgesInOrder(g);
    // N(0)={1,2}, N(1)={0,2}: shared {2} -> 1/(4-1). Everything else has no overlap.
    requireSameEdges(edges, {{0, 1, 1.0 / 3.0}, {1, 0, 1.0 / 3.0}});

    for (const auto& [a, b, w] : edges)
        CHECK(w > 0);
    // N(3) = {2}, N(2) = {3}: no overlap -> edge 3->2 dropped.
    for (const auto& [a, b, w] : edges)
        CHECK_FALSE((a == 3 && b == 2));
}

TEST_CASE("gaussianKernel", "[core][gaussian]") {
    // Two points, k = 2 (parallel kNN entries are kept as parallel edges).
    const auto knn = makeKnn(2, {
                                    {{1, 0.0}, {1, 1.0}},
                                    {{0, 2.0}, {0, 3.0}},
                                });

    SECTION("sigma = 1: nothing is dropped, weights follow the kernel") {
        const BGraph g = gaussianKernel(knn, 1.0);
        CHECK(boost::num_vertices(g) == 2);
        REQUIRE(boost::num_edges(g) == 4);
        requireSameEdges(
            edgesInOrder(g),
            {{0, 1, gauss(0.0, 1.0)}, {0, 1, gauss(1.0, 1.0)}, {1, 0, gauss(2.0, 1.0)}, {1, 0, gauss(3.0, 1.0)}});
    }

    SECTION("neighbour order is kept (insertion order = row-major)") {
        const auto edges = edgesInOrder(gaussianKernel(knn, 1.0));
        REQUIRE(edges.size() == 4);
        CHECK(std::get<2>(edges[0]) == Approx(gauss(0.0, 1.0)));
        CHECK(std::get<2>(edges[1]) == Approx(gauss(1.0, 1.0)));
        CHECK(std::get<2>(edges[3]) == Approx(gauss(3.0, 1.0)));
    }

    SECTION("sigma = 2") {
        const double c = 1.0 / std::sqrt(2.0 * std::numbers::pi_v<double>);
        const auto edges = edgesInOrder(gaussianKernel(knn, 2.0));
        REQUIRE(edges.size() == 4);
        CHECK(std::get<2>(edges[1]) == Approx(c / 2 * std::exp(-0.5 * 0.25)));
    }

    SECTION("default sigma is 1") {
        requireSameEdges(edgesInOrder(gaussianKernel(knn)), edgesInOrder(gaussianKernel(knn, 1.0)));
    }

    SECTION("non-positive or NaN sigma returns an edgeless graph with all vertices") {
        for (const double sigma : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN()}) {
            INFO("sigma = " << sigma);
            const BGraph g = gaussianKernel(knn, sigma);
            CHECK(boost::num_vertices(g) == 2);
            CHECK(boost::num_edges(g) == 0);
        }
    }
}

TEST_CASE("symmetrize keeps the strict lower triangle", "[core][symmetrize]") {
    // 0->1 (0.5), 1->0 (0.3), 2->0 (0.4), and a diagonal entry 1->1 (9.0)
    const BGraph g = buildGraph(3, {{0, 1, 0.5}, {1, 0, 0.3}, {2, 0, 0.4}, {1, 1, 9.0}});

    SECTION("average") {
        const BGraph avg = symmetrize(g, /*prune=*/false);
        CHECK(boost::num_vertices(avg) == 3);
        const auto edges = edgesInOrder(avg); // sorted by (row, col)
        REQUIRE(edges.size() == 2);
        CHECK(std::get<0>(edges[0]) == 1);
        CHECK(std::get<1>(edges[0]) == 0);
        CHECK(std::get<0>(edges[1]) == 2);
        CHECK(std::get<1>(edges[1]) == 0);
        CHECK(std::get<2>(edges[0]) == Approx(0.4)); // (0.5 + 0.3) / 2
        CHECK(std::get<2>(edges[1]) == Approx(0.2)); // one direction only -> halved
    }

    SECTION("prune keeps only mutual pairs") {
        const BGraph prod = symmetrize(g, /*prune=*/true);
        const auto edges = edgesInOrder(prod);
        REQUIRE(edges.size() == 1);
        CHECK(std::get<0>(edges[0]) == 1);
        CHECK(std::get<1>(edges[0]) == 0);
        CHECK(std::get<2>(edges[0]) == Approx(0.15)); // 0.5 * 0.3
    }
}

// NOTE: sortBySize is not part of graph.hpp and its API was not shown. This test is unchanged
// from the old version; adjust the name/include if the function was renamed or moved.
TEST_CASE("sortBySize relabels by descending size", "[core][sort]") {
    // sizes: label 5 -> 4, label 2 -> 3, label 9 -> 1
    const std::vector<IndexType> in = {2, 5, 5, 9, 2, 5, 2, 5};
    CHECK(sortBySize(in, 0) == std::vector<IndexType>{1, 0, 0, 2, 1, 0, 1, 0});
    // Clusters with size <= min_size are outliers (strict '>' as in the reference).
    CHECK(sortBySize(in, 3) ==
          std::vector<IndexType>{NOT_ASSIGNED, 0, 0, NOT_ASSIGNED, NOT_ASSIGNED, 0, NOT_ASSIGNED, 0});
    CHECK(sortBySize(in, 4) == std::vector<IndexType>(8, NOT_ASSIGNED));
    CHECK(sortBySize({}, 10).empty());
}

TEST_CASE("modularity", "[core][modularity]") {
    // two disjoint triangles: {0,1,2} and {3,4,5}
    const BGraph g = buildGraph(6, {{1, 0, 1.0}, {2, 0, 1.0}, {2, 1, 1.0}, {4, 3, 1.0}, {5, 3, 1.0}, {5, 4, 1.0}});
    CHECK(modularity(g, {0, 0, 0, 1, 1, 1}) == Approx(0.5));
    CHECK(modularity(g, {0, 0, 0, 0, 0, 0}) == Approx(0.0).margin(1e-12));
    CHECK_THROWS_AS(modularity(g, {0, 1}), std::invalid_argument);
}

TEST_CASE("fixed dataset: reference communities have the reference modularity", "[core][modularity]") {
    // fixedAffinity() builds the BGraph from the reference affinity arrays.
    CHECK(modularity(test::fixedAffinity(), test::communities) == Approx(test::kQLouvain).margin(1e-6));
}
