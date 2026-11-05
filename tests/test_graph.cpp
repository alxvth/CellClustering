#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numbers>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "cellclustering/graph.hpp"
#include "cellclustering/types.hpp"

using namespace cellclustering;
using Catch::Approx;

namespace {

    using WEdge = std::tuple<IndexType, IndexType, DistanceType>; // (source, target, weight)

    KnnResult makeKnn(const std::size_t k, std::vector<EdgeList> rows) {
        KnnResult r;
        r.k = k;
        r.neighborList = std::move(rows);
        return r;
    }

    // All edges of g as (source, target, weight), sorted. source/target keep insertion order.
    std::vector<WEdge> allEdgesOf(const BGraph& g) {
        std::vector<WEdge> out;
        const auto wmap = get(boost::edge_weight, g);
        for (auto [it, end] = boost::edges(g); it != end; ++it)
            out.emplace_back(boost::source(*it, g), boost::target(*it, g), wmap[*it]);
        std::ranges::sort(out);
        return out;
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

    void requireEdges(const std::vector<WEdge>& actual, const std::vector<WEdge>& expected) {
        REQUIRE(actual.size() == expected.size());
        for (std::size_t i = 0; i < actual.size(); ++i) {
            INFO("edge index " << i);
            CHECK(std::get<0>(actual[i]) == std::get<0>(expected[i]));
            CHECK(std::get<1>(actual[i]) == std::get<1>(expected[i]));
            CHECK(std::get<2>(actual[i]) == Approx(std::get<2>(expected[i])).margin(1e-12));
        }
    }

    // k-NN of `cliques` disjoint cliques of `size` nodes each (k = size - 1).
    KnnResult cliqueKnn(const std::size_t cliques, const std::size_t size) {
        std::vector<EdgeList> rows(cliques * size);
        for (std::size_t c = 0; c < cliques; ++c)
            for (std::size_t i = 0; i < size; ++i)
                for (std::size_t j = 0; j < size; ++j)
                    if (i != j)
                        rows[c * size + i].emplace_back(c * size + j, 1.0);
        return makeKnn(size - 1, std::move(rows));
    }

    double gauss(const double d, const double sigma) {
        const double x = d / sigma;
        return std::exp(-0.5 * x * x) / (sigma * std::sqrt(2.0 * std::numbers::pi_v<double>));
    }

} // namespace

// ---------------------------------------------------------------------------------------------
// jaccardKernel
// ---------------------------------------------------------------------------------------------

TEST_CASE("jaccardKernel: empty input", "[graph][jaccard]") {
    SECTION("no rows") {
        const BGraph g = jaccardKernel(makeKnn(3, {}));
        CHECK(boost::num_vertices(g) == 0);
        CHECK(boost::num_edges(g) == 0);
    }
    SECTION("k == 0 keeps the vertices but has no edges") {
        const BGraph g = jaccardKernel(makeKnn(0, std::vector<EdgeList>(3)));
        CHECK(boost::num_vertices(g) == 3);
        CHECK(boost::num_edges(g) == 0);
    }
}

TEST_CASE("jaccardKernel: complete graph on 4 nodes (k = 3)", "[graph][jaccard]") {
    // N(i) = all other nodes. For edge i->j: |N(i) AND N(j)| = 2, so w = 2 / (2*3 - 2) = 0.5.
    const BGraph g = jaccardKernel(cliqueKnn(1, 4));
    CHECK(boost::num_vertices(g) == 4);
    REQUIRE(boost::num_edges(g) == 12); // directed kNN edges stay separate (no merging)
    for (const auto& [a, b, w] : allEdgesOf(g)) {
        CHECK(a != b);
        CHECK(w == Approx(0.5));
    }
}

TEST_CASE("jaccardKernel: ring with k = 2 drops zero-overlap edges", "[graph][jaccard]") {
    // N(i) = {i+1, i+2} mod 5. Only i -> i+1 shares one neighbour (i+2); i -> i+2 shares none.
    const auto knn = makeKnn(2, {
                                    {{1, 0.1}, {2, 0.2}},
                                    {{2, 0.1}, {3, 0.2}},
                                    {{3, 0.1}, {4, 0.2}},
                                    {{4, 0.1}, {0, 0.2}},
                                    {{0, 0.1}, {1, 0.2}},
                                });
    const BGraph g = jaccardKernel(knn);
    CHECK(boost::num_vertices(g) == 5);
    // 1 / (2*2 - 1) = 1/3
    requireEdges(allEdgesOf(g), {{0, 1, 1.0 / 3}, {1, 2, 1.0 / 3}, {2, 3, 1.0 / 3}, {3, 4, 1.0 / 3}, {4, 0, 1.0 / 3}});
}

TEST_CASE("jaccardKernel: no overlap gives an edgeless graph with all vertices", "[graph][jaccard]") {
    const auto knn = makeKnn(1, {{{1, 1.0}}, {{0, 1.0}}, {{3, 1.0}}, {{2, 1.0}}});
    const BGraph g = jaccardKernel(knn);
    CHECK(boost::num_vertices(g) == 4);
    CHECK(boost::num_edges(g) == 0);
}

TEST_CASE("jaccardKernel: edge direction is preserved (source = row, target = neighbour)", "[graph][jaccard]") {
    const BGraph g = jaccardKernel(cliqueKnn(1, 3)); // k = 2, shared = 1, w = 1/3
    const auto edges = allEdgesOf(g);
    REQUIRE(edges.size() == 6);
    // both directions of every pair exist as separate edges
    for (IndexType i = 0; i < 3; ++i)
        for (IndexType j = 0; j < 3; ++j)
            if (i != j) {
                const auto n = std::ranges::count_if(
                    edges, [&](const WEdge& e) { return std::get<0>(e) == i && std::get<1>(e) == j; });
                CHECK(n == 1);
            }
}

TEST_CASE("jaccardKernel: weights are independent of neighbour order and distances", "[graph][jaccard]") {
    const auto a = makeKnn(2, {{{1, 0.1}, {2, 0.2}}, {{0, 0.1}, {2, 0.2}}, {{0, 0.1}, {1, 0.2}}});
    const auto b = makeKnn(2, {{{2, 9.0}, {1, 9.5}}, {{2, 1.0}, {0, 2.0}}, {{1, 3.0}, {0, 4.0}}});
    requireEdges(allEdgesOf(jaccardKernel(a)), allEdgesOf(jaccardKernel(b)));
}

TEST_CASE("jaccardKernel: weights lie in (0, 1]", "[graph][jaccard]") {
    const BGraph g = jaccardKernel(cliqueKnn(2, 6));
    REQUIRE(boost::num_edges(g) > 0);
    for (const auto& [a, b, w] : allEdgesOf(g)) {
        CHECK(w > 0.0);
        CHECK(w <= 1.0);
    }
}

// ---------------------------------------------------------------------------------------------
// gaussianKernel
// ---------------------------------------------------------------------------------------------

TEST_CASE("gaussianKernel: invalid sigma returns graph without edges", "[graph][gaussian]") {
    const auto knn = makeKnn(1, {{{1, 1.0}}, {{0, 1.0}}});

    const BGraph g1 = gaussianKernel(knn, 0.0);
    CHECK(boost::num_vertices(g1) == knn.nRows());
    CHECK(boost::num_edges(g1) == 0);

    const BGraph g2 = gaussianKernel(knn, -1.0);
    CHECK(boost::num_vertices(g2) == knn.nRows());
    CHECK(boost::num_edges(g2) == 0);

    const BGraph g3 = gaussianKernel(knn, std::numeric_limits<double>::quiet_NaN());
    CHECK(boost::num_vertices(g3) == knn.nRows());
    CHECK(boost::num_edges(g3) == 0);
}

TEST_CASE("gaussianKernel: empty input", "[graph][gaussian]") {
    const BGraph g = gaussianKernel(makeKnn(2, {}), 1.0);
    CHECK(boost::num_vertices(g) == 0);
    CHECK(boost::num_edges(g) == 0);
}

TEST_CASE("gaussianKernel: one edge per kNN entry with the expected weight", "[graph][gaussian]") {
    const auto knn = makeKnn(2, {
                                    {{1, 0.0}, {2, 1.0}},
                                    {{0, 0.5}, {2, 2.0}},
                                    {{1, 1.0}, {0, 3.0}},
                                });

    for (const double sigma : {0.5, 1.0, 2.5}) {
        INFO("sigma = " << sigma);
        const BGraph g = gaussianKernel(knn, sigma);
        CHECK(boost::num_vertices(g) == 3);
        requireEdges(allEdgesOf(g), {{0, 1, gauss(0.0, sigma)},
                                  {0, 2, gauss(1.0, sigma)},
                                  {1, 0, gauss(0.5, sigma)},
                                  {1, 2, gauss(2.0, sigma)},
                                  {2, 0, gauss(3.0, sigma)},
                                  {2, 1, gauss(1.0, sigma)}});
    }
}

TEST_CASE("gaussianKernel: default sigma is 1", "[graph][gaussian]") {
    const auto knn = makeKnn(1, {{{1, 0.7}}, {{0, 0.7}}});
    requireEdges(allEdgesOf(gaussianKernel(knn)), allEdgesOf(gaussianKernel(knn, 1.0)));
}

TEST_CASE("gaussianKernel: zero distance gives the peak 1 / (sigma sqrt(2 pi))", "[graph][gaussian]") {
    const auto knn = makeKnn(1, {{{1, 0.0}}, {{0, 0.0}}});
    const auto edges = allEdgesOf(gaussianKernel(knn, 2.0));
    REQUIRE(edges.size() == 2);
    for (const auto& e : edges)
        CHECK(std::get<2>(e) == Approx(1.0 / (2.0 * std::sqrt(2.0 * std::numbers::pi_v<double>))));
}

TEST_CASE("gaussianKernel: weight decreases with distance and stays positive", "[graph][gaussian]") {
    const auto knn = makeKnn(3, {{{1, 0.1}, {2, 1.0}, {3, 5.0}},
                                 {{0, 1.0}, {2, 1.0}, {3, 1.0}},
                                 {{0, 1.0}, {1, 1.0}, {3, 1.0}},
                                 {{0, 1.0}, {1, 1.0}, {2, 1.0}}});
    const auto edges = allEdgesOf(gaussianKernel(knn, 1.0));
    std::map<IndexType, double> fromZero;
    for (const auto& [a, b, w] : edges) {
        CHECK(w > 0.0);
        if (a == 0)
            fromZero[b] = w;
    }
    REQUIRE(fromZero.size() == 3);
    CHECK(fromZero[1] > fromZero[2]);
    CHECK(fromZero[2] > fromZero[3]);
}

TEST_CASE("gaussianKernel: larger sigma flattens the weights", "[graph][gaussian]") {
    const auto knn = makeKnn(1, {{{1, 3.0}}, {{0, 3.0}}});
    const double narrow = std::get<2>(allEdgesOf(gaussianKernel(knn, 0.5)).front());
    const double wide = std::get<2>(allEdgesOf(gaussianKernel(knn, 5.0)).front());
    // at d = 3 the wide kernel is much larger even though its peak is lower
    CHECK(wide > narrow);
}

// ---------------------------------------------------------------------------------------------
// symmetrize
// ---------------------------------------------------------------------------------------------

TEST_CASE("symmetrize: empty graph", "[graph][symmetrize]") {
    for (const bool prune : {false, true}) {
        const BGraph g = symmetrize(BGraph(0), prune);
        CHECK(boost::num_vertices(g) == 0);
        CHECK(boost::num_edges(g) == 0);
    }
}

TEST_CASE("symmetrize: vertex count is preserved, including isolated vertices", "[graph][symmetrize]") {
    const BGraph in = buildGraph(10, {{0, 1, 1.0}, {1, 0, 1.0}});
    CHECK(boost::num_vertices(symmetrize(in, false)) == 10);
    CHECK(boost::num_vertices(symmetrize(in, true)) == 10);
}

TEST_CASE("symmetrize (average): mutual edges are averaged", "[graph][symmetrize]") {
    const BGraph in = buildGraph(3, {{0, 1, 2.0}, {1, 0, 4.0}});
    // stored once at (row > col) = (1, 0), weight (2 + 4) / 2
    requireEdges(allEdgesOf(symmetrize(in, false)), {{1, 0, 3.0}});
}

TEST_CASE("symmetrize (average): one-directional edges are halved", "[graph][symmetrize]") {
    const BGraph in = buildGraph(3, {{0, 1, 2.0}, {2, 1, 6.0}});
    // 0->1 is upper (0 < 1): stored at (1, 0) with 0.5 * 2. 2->1 is lower: stored at (2, 1) with 0.5 * 6.
    requireEdges(allEdgesOf(symmetrize(in, false)), {{1, 0, 1.0}, {2, 1, 3.0}});
}

TEST_CASE("symmetrize (prune): only mutual edges survive, with product weight", "[graph][symmetrize]") {
    const BGraph in = buildGraph(4, {{0, 1, 2.0}, {1, 0, 4.0}, {2, 3, 5.0}});
    requireEdges(allEdgesOf(symmetrize(in, true)), {{1, 0, 8.0}});
}

TEST_CASE("symmetrize: self-loops are removed", "[graph][symmetrize]") {
    const BGraph in = buildGraph(3, {{0, 0, 5.0}, {1, 1, 5.0}, {0, 2, 1.0}, {2, 0, 1.0}});
    for (const bool prune : {false, true}) {
        const auto edges = allEdgesOf(symmetrize(in, prune));
        REQUIRE(edges.size() == 1);
        CHECK(std::get<0>(edges[0]) == 2);
        CHECK(std::get<1>(edges[0]) == 0);
    }
}

TEST_CASE("symmetrize: output is the strict lower triangle with one edge per pair", "[graph][symmetrize]") {
    const BGraph in = buildGraph(
        5, {{0, 1, 1.0}, {1, 0, 1.0}, {1, 2, 1.0}, {3, 2, 1.0}, {2, 3, 1.0}, {4, 0, 1.0}, {0, 4, 1.0}, {4, 4, 1.0}});
    for (const bool prune : {false, true}) {
        const auto edges = allEdgesOf(symmetrize(in, prune));
        std::map<std::pair<IndexType, IndexType>, int> count;
        for (const auto& [a, b, w] : edges) {
            CHECK(a > b);
            ++count[{a, b}];
        }
        for (const auto& kv : count)
            CHECK(kv.second == 1);
    }
}

TEST_CASE("symmetrize: repeated entries for the same pair are summed per side", "[graph][symmetrize]") {
    // two upper entries (2 + 3) and one lower entry (4)
    const BGraph in = buildGraph(2, {{0, 1, 2.0}, {0, 1, 3.0}, {1, 0, 4.0}});
    requireEdges(allEdgesOf(symmetrize(in, false)), {{1, 0, 0.5 * (4.0 + 5.0)}});
    requireEdges(allEdgesOf(symmetrize(in, true)), {{1, 0, 4.0 * 5.0}});
}

TEST_CASE("symmetrize: zero-weight results are dropped", "[graph][symmetrize]") {
    const BGraph in = buildGraph(2, {{0, 1, 0.0}, {1, 0, 0.0}});
    CHECK(boost::num_edges(symmetrize(in, false)) == 0);
    CHECK(boost::num_edges(symmetrize(in, true)) == 0);
}

TEST_CASE("symmetrize: does not modify its input", "[graph][symmetrize]") {
    const BGraph in = buildGraph(3, {{0, 1, 2.0}, {1, 0, 4.0}, {2, 1, 1.0}});
    const auto before = allEdgesOf(in);
    (void)symmetrize(in, false);
    (void)symmetrize(in, true);
    requireEdges(allEdgesOf(in), before);
}

TEST_CASE("symmetrize: result is symmetric in the input direction", "[graph][symmetrize]") {
    // G and G^T must give the same symmetrized graph.
    const std::vector<WEdge> fwd = {{0, 1, 1.0}, {1, 2, 2.0}, {2, 0, 3.0}, {1, 0, 0.5}};
    std::vector<WEdge> rev;
    for (const auto& [a, b, w] : fwd)
        rev.emplace_back(b, a, w);
    for (const bool prune : {false, true})
        requireEdges(allEdgesOf(symmetrize(buildGraph(3, fwd), prune)), allEdgesOf(symmetrize(buildGraph(3, rev), prune)));
}

// ---------------------------------------------------------------------------------------------
// modularity
// ---------------------------------------------------------------------------------------------

TEST_CASE("modularity: communities size must match vertex count", "[graph][modularity]") {
    const BGraph g = buildGraph(3, {{0, 1, 1.0}});
    CHECK_THROWS_AS(modularity(g, {0, 0}), std::invalid_argument);
    CHECK_THROWS_AS(modularity(g, {0, 0, 0, 0}), std::invalid_argument);
}

TEST_CASE("modularity: graph without edges has modularity 0", "[graph][modularity]") {
    CHECK(modularity(BGraph(0), {}) == Approx(0.0));
    CHECK(modularity(BGraph(3), {0, 1, 2}) == Approx(0.0));
}

TEST_CASE("modularity: two disjoint triangles", "[graph][modularity]") {
    const BGraph g = buildGraph(6, {{0, 1, 1.0}, {1, 2, 1.0}, {2, 0, 1.0}, {3, 4, 1.0}, {4, 5, 1.0}, {5, 3, 1.0}});
    // 2m = 12; in_total = 12; each community has degree sum 6 -> Q = 1 - 2 * (6/12)^2 = 0.5
    CHECK(modularity(g, {0, 0, 0, 1, 1, 1}) == Approx(0.5));
    // everything in one community -> Q = 0
    CHECK(modularity(g, {0, 0, 0, 0, 0, 0}) == Approx(0.0).margin(1e-12));
}

TEST_CASE("modularity: path graph with known value", "[graph][modularity]") {
    const BGraph g = buildGraph(4, {{0, 1, 1.0}, {1, 2, 1.0}, {2, 3, 1.0}});
    // 2m = 6, in_total = 4, tot = {3, 3} -> Q = 4/6 - 2 * (3/6)^2 = 1/6
    CHECK(modularity(g, {0, 0, 1, 1}) == Approx(1.0 / 6.0));
    // singletons: Q = -sum (d_i / 2m)^2 = -(1 + 4 + 4 + 1) / 36
    CHECK(modularity(g, {0, 1, 2, 3}) == Approx(-10.0 / 36.0));
}

TEST_CASE("modularity: two nodes with a single edge", "[graph][modularity]") {
    const BGraph g = buildGraph(2, {{0, 1, 2.0}});
    CHECK(modularity(g, {0, 0}) == Approx(0.0).margin(1e-12));
    CHECK(modularity(g, {0, 1}) == Approx(-0.5));
}

TEST_CASE("modularity: community labels are arbitrary", "[graph][modularity]") {
    const BGraph g = buildGraph(6, {{0, 1, 1.0}, {1, 2, 1.0}, {2, 0, 1.0}, {3, 4, 1.0}, {4, 5, 1.0}, {5, 3, 1.0}});
    CHECK(modularity(g, {7, 7, 7, 42, 42, 42}) == Approx(modularity(g, {0, 0, 0, 1, 1, 1})));
    CHECK(modularity(g, {1, 1, 1, 0, 0, 0}) == Approx(modularity(g, {0, 0, 0, 1, 1, 1})));
}

TEST_CASE("modularity: invariant under global weight scaling", "[graph][modularity]") {
    const std::vector<WEdge> base = {{0, 1, 1.0}, {1, 2, 2.0}, {2, 3, 1.0}, {3, 0, 0.5}, {1, 3, 0.25}};
    std::vector<WEdge> scaled;
    for (const auto& [a, b, w] : base)
        scaled.emplace_back(a, b, 10.0 * w);
    const std::vector<IndexType> comm = {0, 0, 1, 1};
    CHECK(modularity(buildGraph(4, base), comm) == Approx(modularity(buildGraph(4, scaled), comm)));
}

TEST_CASE("modularity: parallel edges behave like one edge with the summed weight", "[graph][modularity]") {
    const BGraph multi = buildGraph(3, {{0, 1, 1.0}, {0, 1, 1.0}, {1, 2, 1.0}});
    const BGraph merged = buildGraph(3, {{0, 1, 2.0}, {1, 2, 1.0}});
    CHECK(modularity(multi, {0, 0, 1}) == Approx(modularity(merged, {0, 0, 1})));
}

TEST_CASE("modularity: a good partition scores higher than a bad one", "[graph][modularity]") {
    const BGraph g =
        buildGraph(6, {{0, 1, 1.0}, {1, 2, 1.0}, {2, 0, 1.0}, {3, 4, 1.0}, {4, 5, 1.0}, {5, 3, 1.0}, {2, 3, 0.1}});
    CHECK(modularity(g, {0, 0, 0, 1, 1, 1}) > modularity(g, {0, 1, 0, 1, 0, 1}));
}

// ---------------------------------------------------------------------------------------------
// pipeline: kernel -> symmetrize -> modularity
// ---------------------------------------------------------------------------------------------

TEST_CASE("pipeline: two cliques are recovered with the expected modularity", "[graph][pipeline]") {
    const auto knn = cliqueKnn(2, 4); // nodes 0-3 and 4-7, k = 3

    const BGraph jac = jaccardKernel(knn); // 24 directed edges, each 0.5
    REQUIRE(boost::num_edges(jac) == 24);

    for (const bool prune : {false, true}) {
        INFO("prune = " << prune);
        const BGraph sym = symmetrize(jac, prune);
        // 6 undirected pairs per clique. Average: 0.5. Prune: 0.5 * 0.5 = 0.25.
        REQUIRE(boost::num_edges(sym) == 12);
        const double expectedW = prune ? 0.25 : 0.5;
        for (const auto& [a, b, w] : allEdgesOf(sym)) {
            CHECK(a > b);
            CHECK((a / 4) == (b / 4)); // no cross-clique edges
            CHECK(w == Approx(expectedW));
        }
        // 2m = 12 * w * 2; in_total = 2m; two equal communities -> Q = 1 - 2 * (1/2)^2 = 0.5
        CHECK(modularity(sym, {0, 0, 0, 0, 1, 1, 1, 1}) == Approx(0.5));
        CHECK(modularity(sym, {0, 1, 0, 1, 0, 1, 0, 1}) < 0.0);
    }
}

TEST_CASE("pipeline: gaussian kernel on a symmetric kNN graph", "[graph][pipeline]") {
    const auto knn = cliqueKnn(2, 3); // all distances == 1, k = 2
    const BGraph sym = symmetrize(gaussianKernel(knn, 1.0), false);
    REQUIRE(boost::num_edges(sym) == 6);
    for (const auto& [a, b, w] : allEdgesOf(sym))
        CHECK(w == Approx(gauss(1.0, 1.0)));
    CHECK(modularity(sym, {0, 0, 0, 1, 1, 1}) == Approx(0.5));
}