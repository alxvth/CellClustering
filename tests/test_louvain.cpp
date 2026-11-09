#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tuple>
#include <vector>

#include "cellclustering/louvain.hpp"
#include "cellclustering/options.hpp"
#include "cellclustering/types.hpp"

#include "test_utils.hpp"

using Catch::Approx;
using namespace cellclustering;

namespace {

    using WEdge = std::tuple<IndexType, IndexType, DistanceType>;

    BGraph buildGraph(const std::size_t n, const std::vector<WEdge>& es) {
        IndexPairList pairs;
        std::vector<DistanceType> w;
        for (const auto& [a, b, x] : es) {
            pairs.emplace_back(a, b);
            w.push_back(x);
        }
        return BGraph(pairs.begin(), pairs.end(), w.begin(), n);
    }

    // Two cliques of `size` vertices (weight 1) joined by a single weak bridge (weight 0.1).
    // Vertices 0..size-1 and size..2*size-1. `extraVertices` adds isolated vertices at the end.
    BGraph twoCliques(const std::size_t size, const std::size_t extraVertices = 0) {
        std::vector<WEdge> es;
        for (std::size_t c = 0; c < 2; ++c)
            for (std::size_t i = 0; i < size; ++i)
                for (std::size_t j = 0; j < i; ++j)
                    es.emplace_back(c * size + i, c * size + j, 1.0);
        es.emplace_back(size, 0, 0.1); // single weak bridge
        return buildGraph(2 * size + extraVertices, es);
    }

} // namespace

TEST_CASE("runLouvain on the reference affinity graph", "[louvain]") {
    Options opt = test::makeOptions(1);
    opt.louvainParams.qTol = 1e-3;
    opt.louvainParams.timeLimit = 2000;

    // fixedAffinity() must now return a BGraph (see note on the reference graph).
    const BGraph g = test::fixedAffinity();
    const LouvainResult r = runLouvain(g, opt);

    const std::vector<IndexType>& communities = test::communities;

    const double modularity = test::computeModularity(g, r.communities);

    CHECK(test::samePartition(r.communities, communities));
    CHECK(r.Q == Approx(test::kQLouvain).margin(1e-6));
    CHECK(r.Q == Approx(modularity).margin(1e-9));
    CHECK(r.runs >= 1);
}

TEST_CASE("runLouvain finds planted communities", "[louvain]") {
    const Options opt = test::makeOptions(3);
    const BGraph g = twoCliques(12);
    const LouvainResult r = runLouvain(g, opt);

    const double modularity = test::computeModularity(g, r.communities);

    REQUIRE(r.communities.size() == 24);
    CHECK(test::nDistinct(r.communities) == 2);
    for (std::size_t i = 1; i < 12; ++i) {
        CHECK(r.communities[i] == r.communities[0]);
        CHECK(r.communities[12 + i] == r.communities[12]);
    }
    CHECK(r.communities[0] != r.communities[12]);
    CHECK(r.Q > 0.4);
    CHECK(r.Q == Approx(modularity).margin(1e-9));
}

TEST_CASE("runLouvain is reproducible for a fixed seed", "[louvain]") {
    const Options opt = test::makeOptions(99);
    const BGraph g = twoCliques(10);
    const auto a = runLouvain(g, opt);
    const auto b = runLouvain(g, opt);
    CHECK(a.communities == b.communities);
    CHECK(a.Q == Approx(b.Q));
    CHECK(a.runs == b.runs);
}

TEST_CASE("runLouvain restart policy", "[louvain]") {
    const BGraph g = twoCliques(8);

    SECTION("maxRuns caps the number of restarts") {
        Options opt = test::makeOptions(5);
        opt.louvainParams.maxRuns = 3;
        const auto r = runLouvain(g, opt);
        CHECK(r.runs >= 1);
        CHECK(r.runs <= 3);
    }

    SECTION("patience stops after that many runs without improvement") {
        Options opt = test::makeOptions(5);
        opt.louvainParams.maxRuns = 100;
        opt.louvainParams.patience = 2;
        const auto r = runLouvain(g, opt);
        CHECK(r.runs >= 1);
        CHECK(r.runs <= 100);
        // The first run is accepted (Q > qTol) and later restarts find the same optimum,
        // so the loop ends after 1 + patience runs.
        CHECK(r.runs == 1 + opt.louvainParams.patience);
    }

    SECTION("timeLimit = 0 still makes exactly one run") {
        Options opt = test::makeOptions(5);
        opt.louvainParams.timeLimit = 0;
        const auto r = runLouvain(g, opt);
        CHECK(r.runs == 1);
        CHECK(r.communities.size() == 16);
    }

    SECTION("maxRuns = 1 makes exactly one run") {
        Options opt = test::makeOptions(5);
        opt.louvainParams.maxRuns = 1;
        CHECK(runLouvain(g, opt).runs == 1);
    }
}

TEST_CASE("runLouvain edge cases", "[louvain]") {
    SECTION("graph without vertices") {
        const auto r = runLouvain(BGraph(0), Options{});
        CHECK(r.communities.empty());
    }

    SECTION("graph without edges: every vertex alone") {
        const auto r = runLouvain(BGraph(4), Options{});
        CHECK(r.communities == std::vector<IndexType>{0, 1, 2, 3});
    }

    SECTION("isolated vertices stay singletons") {
        const Options opt = test::makeOptions(2);
        const BGraph g = twoCliques(6, 2); // vertices 12, 13 have no edges
        const auto r = runLouvain(g, opt);
        REQUIRE(r.communities.size() == 14);
        CHECK(r.communities[12] != r.communities[13]);
        CHECK(test::nDistinct(r.communities) == 4);
    }

    SECTION("zero-weight edges are accepted and do not change the planted partition") {
        const Options opt = test::makeOptions(7);
        std::vector<WEdge> es;
        for (std::size_t c = 0; c < 2; ++c)
            for (std::size_t i = 0; i < 5; ++i)
                for (std::size_t j = 0; j < i; ++j)
                    es.emplace_back(c * 5 + i, c * 5 + j, 1.0);
        es.emplace_back(5, 0, 0.1);
        es.emplace_back(1, 7, 0.0); // zero-weight edge across the cliques
        const BGraph g = buildGraph(10, es);

        LouvainResult r;
        REQUIRE_NOTHROW(r = runLouvain(g, opt));
        REQUIRE(r.communities.size() == 10);
        CHECK(test::nDistinct(r.communities) == 2);
        CHECK(r.communities[0] != r.communities[5]);
    }
}
