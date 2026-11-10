/*
 * Copyright (c) 2026 Alexander Vieth
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cellclustering/neighbors.hpp"

#pragma warning(disable : 4702) // knncolle: unreachable code
#pragma warning(disable : 4701) // hnswlib: potentially uninitialized local variable
#include <knncolle_annoy/knncolle_annoy.hpp>
#include <knncolle_hnsw/knncolle_hnsw.hpp>
#include <knncolle_kmknn/knncolle_kmknn.hpp>
#pragma warning(default : 4701)
#pragma warning(default : 4702)

#include "knn_extension.hpp"

#include <memory>
#include <stdexcept>
#include <string>

namespace cellclustering {

    using NeighborList = knncolle::NeighborList<IndexType, DistanceType>;
    using SimpleMatrix = knncolle::SimpleMatrix<IndexType, DataType>;

    namespace {

        NeighborList knnHnsw(const SimpleMatrix& mat, const Options& opt) {
            using HnswBuilder = knncolle_hnsw::HnswBuilder<IndexType, DataType, DistanceType, SimpleMatrix>;
            using HnswIndex   = knncolle::Prebuilt<IndexType, DataType, DistanceType>;

            const knncolle_hnsw::HnswOptions knnOpts = {.num_links       = opt.hnsw.numLinks,
                                                        .ef_construction = opt.hnsw.efConstruction,
                                                        .ef_search       = opt.hnsw.efSearch};

            std::unique_ptr<HnswBuilder> hnswBuilder;

            switch (opt.nnMetric) {
            case Metric::Euclidean: {
                const auto euclidConfig = knncolle_hnsw::configure_euclidean_distance<DistanceType, DataType>();
                hnswBuilder.reset(new HnswBuilder(euclidConfig, knnOpts));
                break;
            }
            case Metric::Manhattan: {
                const auto manhattanConfig = knncolle_hnsw::configure_manhattan_distance<DistanceType, DataType>();
                hnswBuilder.reset(new HnswBuilder(manhattanConfig, knnOpts));
                break;
            }
            case Metric::Cosine: {
                const auto cosineConfig = knncolle_hnsw::configure_cosine_distance<DistanceType, DataType>();
                hnswBuilder.reset(new HnswBuilder(cosineConfig, knnOpts));
                break;
            }
            }

            const std::unique_ptr<HnswIndex> hnswIndex = hnswBuilder->build_unique(mat);

            return knncolle::find_nearest_neighbors(*hnswIndex, opt.k, static_cast<int>(getNumThreads(opt.nJobs)));
        }

        NeighborList knnAnnoy(const SimpleMatrix& mat, const Options& opt) {
            using AnnoyIndex = knncolle::Prebuilt<IndexType, DataType, DistanceType>;

            std::unique_ptr<AnnoyIndex> annoyIndex;

            const knncolle_annoy::AnnoyOptions knnOpts = {.num_trees   = opt.annoy.nTrees,
                                                          .search_mult = opt.annoy.searchK};

            switch (opt.nnMetric) {
            case Metric::Euclidean: {
                knncolle_annoy::AnnoyBuilder<IndexType, DataType, DistanceType, Annoy::Euclidean> annoyBuilder(knnOpts);
                annoyIndex = annoyBuilder.build_unique(mat);
                break;
            }
            case Metric::Cosine: {
                knncolle_annoy::AnnoyBuilder<IndexType, DataType, DistanceType, Annoy::Angular> annoyBuilder(knnOpts);
                annoyIndex = annoyBuilder.build_unique(mat);
                break;
            }
            case Metric::Manhattan: {
                knncolle_annoy::AnnoyBuilder<IndexType, DataType, DistanceType, Annoy::Manhattan> annoyBuilder(knnOpts);
                annoyIndex = annoyBuilder.build_unique(mat);
                break;
            }
            }

            return knncolle::find_nearest_neighbors(*annoyIndex, opt.k, static_cast<int>(getNumThreads(opt.nJobs)));
        }

        NeighborList knnKmknn(const SimpleMatrix& mat, const Options& opt) {
            using KmknnBuilder = knncolle_kmknn::KmknnBuilder<IndexType, DataType, DistanceType, SimpleMatrix>;
            using KmknnIndex   = knncolle::Prebuilt<IndexType, DataType, DistanceType>;

            const KmknnBuilder::Options knnOpts = {
                .power = opt.kmknn.power, .initialize_algorithm = nullptr, .refine_algorithm = nullptr};

            std::unique_ptr<KmknnBuilder> kmknnBuilder;

            switch (opt.nnMetric) {
            case Metric::Euclidean: {
                const auto kmknnMetricData = std::make_shared<knncolle::EuclideanDistance<DataType, DistanceType>>();
                const auto kmknnMetricCenter =
                    std::make_shared<knncolle::EuclideanDistance<DistanceType, DistanceType>>();
                kmknnBuilder.reset(new KmknnBuilder(kmknnMetricData, kmknnMetricCenter, knnOpts));
                break;
            }
            case Metric::Manhattan: {
                const auto kmknnMetricData = std::make_shared<knncolle::ManhattanDistance<DataType, DistanceType>>();
                const auto kmknnMetricCenter =
                    std::make_shared<knncolle::ManhattanDistance<DistanceType, DistanceType>>();
                kmknnBuilder.reset(new KmknnBuilder(kmknnMetricData, kmknnMetricCenter, knnOpts));
                break;
            }
            case Metric::Cosine: {
                const auto kmknnMetricData   = std::make_shared<knncolle::CosineDistance<DataType, DistanceType>>();
                const auto kmknnMetricCenter = std::make_shared<knncolle::CosineDistance<DistanceType, DistanceType>>();
                kmknnBuilder.reset(new KmknnBuilder(kmknnMetricData, kmknnMetricCenter, knnOpts));
            }
            }

            const std::unique_ptr<KmknnIndex> kmknnIndex = kmknnBuilder->build_unique(mat);

            return knncolle::find_nearest_neighbors(*kmknnIndex, opt.k, static_cast<int>(getNumThreads(opt.nJobs)));
        }

        NeighborList knnBruteForce(const SimpleMatrix& mat, const Options& opt) {
            using BfBuilder = knncolle::BruteforceBuilder<IndexType, DataType, DistanceType, SimpleMatrix>;
            using BfIndex   = knncolle::Prebuilt<IndexType, DataType, DistanceType>;

            std::unique_ptr<BfBuilder> bfBuilder;

            switch (opt.nnMetric) {
            case Metric::Euclidean: {
                const auto bfDist = std::make_shared<knncolle::EuclideanDistance<DataType, DistanceType>>();
                bfBuilder.reset(new BfBuilder(bfDist));
                break;
            }
            case Metric::Manhattan: {
                const auto bfDist = std::make_shared<knncolle::ManhattanDistance<DataType, DistanceType>>();
                bfBuilder.reset(new BfBuilder(bfDist));
                break;
            }
            case Metric::Cosine: {
                const auto bfDist = std::make_shared<knncolle::CosineDistance<DataType, DistanceType>>();
                bfBuilder.reset(new BfBuilder(bfDist));
                break;
            }
            }

            const std::unique_ptr<BfIndex> bfIndex = bfBuilder->build_unique(mat);

            return knncolle::find_nearest_neighbors(*bfIndex, opt.k, static_cast<int>(getNumThreads(opt.nJobs)));
        }

    } // namespace

    KnnResult findNeighbors(const DenseMatrixView& data, const Options& opt) {
        validate(opt);

        if (data.cols() == 0)
            throw std::invalid_argument("data must have at least one column");
        if (static_cast<std::size_t>(opt.k) >= data.rows())
            throw std::invalid_argument("k must be smaller than the number of rows (k=" + std::to_string(opt.k) +
                                        ", rows=" + std::to_string(data.rows()) + ")");

        const SimpleMatrix mat = {data.cols(), static_cast<IndexType>(data.rows()), data.data()};

        KnnResult res;
        res.k = opt.k;

        switch (opt.nnMethod) {
        case NNMethod::HNSW:
            res.neighborList = knnHnsw(mat, opt);
            break;
        case NNMethod::Annoy:
            res.neighborList = knnAnnoy(mat, opt);
            break;
        case NNMethod::KMKNN:
            res.neighborList = knnKmknn(mat, opt);
            break;
        case NNMethod::BruteForce:
            res.neighborList = knnBruteForce(mat, opt);
            break;
        }

        return res;
    }

} // namespace cellclustering
