#include "cellclustering/options.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <chrono>
#include <random>
#include <vector>

#include "cellclustering/types.hpp"
#include "cellclustering/neighbors.hpp"

#include "test_utils.hpp"

using namespace cellclustering;

namespace {
    
    // Helper to create synthetic high-dimensional data in row-major layout
    std::vector<DataType> generateRandomData(const std::size_t nRows, const std::size_t nCols, std::mt19937& rng) {
        std::normal_distribution<DataType> dist(0.0f, 1.0f);
        std::vector<DataType> buffer(nRows * nCols);
        for (auto& val : buffer) {
            val = dist(rng);
        }
        return buffer;
    }

} // namespace

TEST_CASE("findNeighbors: Invariant and Contract Checks", "[knn][invariants]") {
    // 1. Seed from current epoch time and log it for reproducibility
    const auto seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    INFO("PRNG Seed: " << seed);
    std::mt19937 rng(seed);

    constexpr std::size_t nRows = 150;
    constexpr std::size_t nCols = 64;
    constexpr std::size_t k = 10;

    const auto dataBuffer = generateRandomData(nRows, nCols, rng);
    const DenseMatrixView dataView{dataBuffer.data(), nRows, nCols};

    // Test across methods and metrics
    auto method = GENERATE(NNMethod::HNSW, NNMethod::Annoy, NNMethod::KMKNN, NNMethod::BruteForce);
    auto metric = GENERATE(Metric::Euclidean, Metric::Cosine, Metric::Manhattan);

    Options opt;
    opt.nnMethod = method;
    opt.nnMetric = metric;
    opt.k = k;

    KnnResult res = findNeighbors(dataView, opt);

    SECTION("Structural and dimensional validity") {
        REQUIRE(res.nRows() == nRows);
        REQUIRE(res.k == k);

        for (IndexType rowIdx = 0; rowIdx < res.nRows(); ++rowIdx) {
            const auto& neighbors = res.neighborList[rowIdx];

            // Exactly k neighbors returned per row
            REQUIRE(neighbors.size() == k);

            std::unordered_set<IndexType> seenIds;
            for (std::size_t j = 0; j < neighbors.size(); ++j) {
                const auto [neighborId, dist] = neighbors[j];

                CHECK(neighborId < nRows);
                CHECK(neighborId != rowIdx); // data point must not be its own neighbor
                CHECK(dist >= 0.0); // distance must be non-negative
                CHECK(seenIds.insert(neighborId).second); // no duplicate neighbors in the same row

                // Ordered by increasing distance
                if (j > 0) {
                    CHECK(dist >= neighbors[j - 1].second);
                }
            }
        }
    }
}

TEST_CASE("findNeighbors: Approximate Recall vs. BruteForce", "[knn][recall]") {
    const auto seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    INFO("PRNG Seed: " << seed);
    std::mt19937 rng(seed);

    constexpr std::size_t nRows = 200;
    constexpr std::size_t nCols = 128;
    constexpr std::size_t k = 10;

    const auto dataBuffer = generateRandomData(nRows, nCols, rng);
    const DenseMatrixView dataView{dataBuffer.data(), nRows, nCols};

    auto metric = GENERATE(Metric::Euclidean, Metric::Cosine, Metric::Manhattan);
    auto approxMethod = GENERATE(NNMethod::HNSW, NNMethod::Annoy, NNMethod::KMKNN);

    // 1. Compute ground truth using the BruteForce backend
    Options bfOpt;
    bfOpt.nnMethod = NNMethod::BruteForce;
    bfOpt.nnMetric = metric;
    bfOpt.k = k;
    const KnnResult groundTruth = findNeighbors(dataView, bfOpt);

    // 2. Compute approximate neighbors
    Options approxOpt;
    approxOpt.nnMethod = approxMethod;
    approxOpt.nnMetric = metric;
    approxOpt.k = k;
    const KnnResult approxRes = findNeighbors(dataView, approxOpt);

    // 3. Compute Recall@k across all points
    std::size_t totalHits = 0;
    constexpr std::size_t totalPossible = nRows * k;

    for (std::size_t i = 0; i < nRows; ++i) {
        // Collect ground truth IDs for point i into a hash set
        std::unordered_set<IndexType> gtSet;
        for (const auto& [id, _] : groundTruth.neighborList[i]) {
            gtSet.insert(id);
        }

        // Count how many IDs the approximate method found
        for (const auto& [id, _] : approxRes.neighborList[i]) {
            if (gtSet.contains(id)) {
                totalHits++;
            }
        }
    }

    const double recall = static_cast<double>(totalHits) / static_cast<double>(totalPossible);

    INFO("Metric: " << toString(metric) << " | Method: " << toString(approxMethod)
                    << " | Measured Recall@" << k << ": " << recall);

    REQUIRE(recall >= 0.95);
}


TEST_CASE("findNeighbors: check affinity data", "[knn][internal]") {

    const DenseMatrixView dataView{ test::testData, test::kN, test::kDim };

    Options bfOpt;
    bfOpt.nnMethod = NNMethod::BruteForce;
    bfOpt.nnMetric = Metric::Euclidean;
    bfOpt.k = test::kK;
    const KnnResult knn = findNeighbors(dataView, bfOpt);

    const KnnResult expectedKnn = test::fixedKnn();

    CHECK(knn.k == expectedKnn.k);
    CHECK(knn.nRows() == expectedKnn.nRows());

    for (std::size_t i = 0; i < knn.nRows(); ++i) {
        for (std::size_t j = 0; j < knn.k; ++j) {
            CHECK(knn.neighborList[i][j].first == expectedKnn.neighborList[i][j].first);
        }
    }

}
