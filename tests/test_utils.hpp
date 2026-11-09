#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <vector>

#include "cellclustering/options.hpp"
#include "cellclustering/types.hpp"

#include <boost/graph/louvain_quality_functions.hpp>

// data from https://github.com/dpeerlab/PhenoGraph/blob/v1.5.7/tests/conftest.py
// Licence: MIT (2025) Jacob Levine, Columbia University
namespace cellclustering::test {
    inline constexpr std::size_t kN = 30;
    inline constexpr std::size_t kDim = 2;
    inline constexpr std::size_t kK = 5;

    inline constexpr double kQLouvain = 0.666593;

    inline const std::vector<DataType> testData = {
        8.29188122f,   6.77467787f,   8.11595558f,   6.29969468f,   8.41932782f,   7.18708683f,   7.87751289f,
        7.05695946f,   8.15498076f,   7.2317535f,    8.57681f,      7.30907322f,   7.84036909f,   7.04545501f,
        -9.22407124f,  -10.09577064f, -8.97447722f,  -10.29175587f, -9.47952697f,  -10.26263977f, -9.13671857f,
        -9.98270161f,  -9.17054197f,  -10.26074704f, -9.20421828f,  -10.40169291f, -9.28332716f,  -10.29473222f,
        -5.9341068f,   -19.0755188f,  -5.83337714f,  -19.25045898f, -6.12433156f,  -19.51021562f, -5.60446077f,
        -19.16554577f, -5.56675018f,  -19.50978007f, -5.93608833f,  -18.91981749f, -5.7953458f,   -19.04092524f,
        8.00608704f,   7.40402037f,   8.12470637f,   7.1818027f,    8.36344614f,   7.17558329f,   -9.19624784f,
        -10.08752065f, -9.41859495f,  -10.31316491f, -9.30012735f,  -10.32240619f, -6.20397103f,  -19.56631818f,
        -5.83048038f,  -19.3580962f,  -6.21581103f,  -19.22808712f};

    inline const std::vector<IndexType> knnMatrix = {
        23, 2,  22, 4,  3,  0,  3,  6,  22, 23, 23, 5,  4,  22, 0,  6,  22, 4,  21, 23, 22, 23, 21, 2,  3,
        2,  23, 4,  22, 21, 3,  22, 4,  21, 0,  24, 10, 11, 13, 26, 11, 12, 24, 13, 7,  25, 26, 13, 7,  12,
        24, 7,  11, 13, 8,  13, 26, 12, 7,  24, 26, 13, 11, 25, 8,  26, 11, 12, 25, 9,  20, 19, 15, 28, 29,
        28, 14, 20, 17, 19, 27, 29, 28, 15, 14, 20, 15, 28, 14, 18, 28, 17, 15, 20, 16, 14, 20, 15, 17, 29,
        14, 19, 15, 17, 28, 4,  22, 3,  6,  23, 4,  23, 21, 3,  2,  2,  4,  22, 5,  0,  7,  10, 11, 13, 26,
        9,  26, 13, 12, 11, 13, 25, 12, 11, 9,  16, 29, 28, 15, 14, 15, 17, 14, 18, 20, 16, 14, 27, 15, 28};


    inline const std::vector<IndexType> affinityRow = {
        0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,
        5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,
        10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14,
        15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19,
        20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24,
        25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29};

    inline const std::vector<IndexType> affinityCol = {
        23, 2,  22, 4,  3,  0,  3,  6,  22, 23, 23, 5,  4,  22, 0,  6,  22, 4,  21, 23, 22, 23, 21, 2,  3,
        2,  23, 4,  22, 21, 3,  22, 4,  21, 0,  24, 10, 11, 13, 26, 11, 12, 24, 13, 7,  25, 26, 13, 7,  12,
        24, 7,  11, 13, 8,  13, 26, 12, 7,  24, 26, 13, 11, 25, 8,  26, 11, 12, 25, 9,  20, 19, 15, 28, 29,
        28, 14, 20, 17, 19, 27, 29, 28, 15, 14, 20, 15, 28, 14, 18, 28, 17, 15, 20, 16, 14, 20, 15, 17, 29,
        14, 19, 15, 17, 28, 4,  22, 3,  6,  23, 4,  23, 21, 3,  2,  2,  4,  22, 5,  0,  7,  10, 11, 13, 26,
        9,  26, 13, 12, 11, 13, 25, 12, 11, 9,  16, 29, 28, 15, 14, 15, 17, 14, 18, 20, 16, 14, 27, 15, 28};

    inline const std::vector<DistanceType> affinityValues = {
        0.42857143, 0.42857143, 0.66666667, 0.66666667, 0.42857143, 0.42857143, 0.42857143, 0.42857143, 0.25,
        0.25,       0.66666667, 0.42857143, 0.25,       0.25,       0.42857143, 0.42857143, 0.42857143, 0.42857143,
        0.66666667, 0.25,       0.66666667, 0.25,       0.42857143, 0.25,       0.42857143, 0.42857143, 0.42857143,
        0.66666667, 0.66666667, 0.42857143, 0.42857143, 0.42857143, 0.42857143, 0.42857143, 0.42857143, 0.66666667,
        0.42857143, 0.42857143, 0.25,       0.25,       0.66666667, 0.25,       0.42857143, 0.25,       0.42857143,
        0.42857143, 0.42857143, 0.42857143, 0.25,       0.42857143, 0.42857143, 0.42857143, 0.42857143, 0.11111111,
        0.66666667, 0.25,       0.25,       0.25,       0.42857143, 0.42857143, 0.42857143, 0.42857143, 0.25,
        0.42857143, 0.25,       0.66666667, 0.25,       0.42857143, 0.66666667, 0.42857143, 0.42857143, 0.42857143,
        0.42857143, 0.25,       0.25,       0.42857143, 0.42857143, 0.66666667, 0.42857143, 0.42857143, 0.66666667,
        0.66666667, 0.25,       0.25,       0.42857143, 0.42857143, 0.42857143, 0.66666667, 0.42857143, 0.42857143,
        0.42857143, 0.42857143, 0.42857143, 0.42857143, 0.25,       0.42857143, 0.42857143, 0.42857143, 0.42857143,
        0.25,       0.42857143, 0.42857143, 0.66666667, 0.42857143, 0.42857143, 0.42857143, 0.42857143, 0.66666667,
        0.42857143, 0.25,       0.66666667, 0.25,       0.42857143, 0.42857143, 0.25,       0.66666667, 0.25,
        0.25,       0.42857143, 0.42857143, 0.66666667, 0.42857143, 0.42857143, 0.25,       0.25,       0.42857143,
        0.66666667, 0.66666667, 0.42857143, 0.42857143, 0.66666667, 0.66666667, 0.42857143, 0.25,       0.42857143,
        0.66666667, 0.66666667, 0.25,       0.25,       0.42857143, 0.42857143, 0.66666667, 0.25,       0.42857143,
        0.42857143, 0.66666667, 0.25,       0.66666667, 0.25,       0.25};

    inline const std::vector<IndexType> communities = {0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 1,
                                                       1, 1, 1, 1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1};

    inline KnnResult fixedKnn() {
        KnnResult knn;
        knn.k = test::kK;
        knn.neighborList.resize(test::kN);
        for (size_t i = 0; i < test::kN; ++i) {
            knn.neighborList[i].reserve(knn.k);

            for (size_t j = 0; j < knn.k; ++j)
                knn.neighborList[i].emplace_back(knnMatrix[i * knn.k +j], 0.0f);
        }
        return knn;
    }

    inline BGraph fixedAffinity() {
        using namespace cellclustering;

        constexpr std::size_t n = test::kN;
        const std::vector<IndexType>& rows = test::affinityRow;
        const std::vector<IndexType>& cols = test::affinityCol;
        const std::vector<DistanceType>& vals = test::affinityValues;

        if (rows.size() != cols.size() || rows.size() != vals.size())
            throw std::invalid_argument("affinity arrays must have equal length");

        IndexPairList edges;
        std::vector<DistanceType> weights;
        edges.reserve(rows.size());
        weights.reserve(rows.size());

        for (std::size_t t = 0; t < rows.size(); ++t) {
            if (rows[t] >= n || cols[t] >= n) // vecS would silently add vertices otherwise
                throw std::out_of_range("affinity index out of range");
            edges.emplace_back(rows[t], cols[t]);
            weights.push_back(vals[t]);
        }

        return BGraph(edges.begin(), edges.end(), weights.begin(), n);
    }

    /// True if A and B describe the same partition (labels may be permuted)
    template <typename T> 
    bool samePartition(const std::vector<T>& a, const std::vector<T>& b) {
        if (a.size() != b.size())
            return false;
        std::map<T, T> ab, ba;
        for (std::size_t i = 0; i < a.size(); ++i) {
            auto [it1, n1] = ab.emplace(a[i], b[i]);
            auto [it2, n2] = ba.emplace(b[i], a[i]);
            if (it1->second != b[i] || it2->second != a[i])
                return false;
        }
        return true;
    }

    inline Options makeOptions(const std::uint64_t seed) {
        Options opt;
        opt.seed = seed;
        return opt;
    }

    template <typename T>
    std::size_t nDistinct(const std::vector<T>& c) {
        return std::set<T>(c.begin(), c.end()).size();
    }

    inline double computeModularity(const BGraph& g, const std::vector<IndexType>& c) {
        const auto weights = get(boost::edge_weight, g);
        const auto components = boost::make_iterator_property_map(c.begin(), get(boost::vertex_index, g));
        return boost::newman_and_girvan{}.quality(g, components, weights);
    }

} // namespace cellclustering::test
