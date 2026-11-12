/*
 * Copyright (c) 2026 Alexander Vieth
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef CELLCLUSTERING_KNN_EXTENSION_HPP
#define CELLCLUSTERING_KNN_EXTENSION_HPP

#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>

#include <knncolle/distances.hpp>
#include <knncolle_hnsw/Hnsw.hpp> // for hnswlib::SpaceInterface, import before knncolle_hnsw/distances
#include <knncolle_hnsw/distances.hpp>

namespace knncolle {
    // This implementation chooses to return 1 if any input vector's magnitude is 0
    // instead of throwing even though the distance is not well defined
    template <typename Data_, typename Distance_>
    Distance_ cosine_distance(std::size_t num_dimensions, const Data_* x, const Data_* y) {
        Distance_ dot    = 0.0;
        Distance_ norm_a = 0.0;
        Distance_ norm_b = 0.0;

        for (std::size_t i = 0; i < num_dimensions; ++i) {
            dot += x[i] * y[i];
            norm_a += x[i] * x[i];
            norm_b += y[i] * y[i];
        }

        return static_cast<Distance_>(1.0 -
                                      dot / (std::sqrt(norm_a * norm_b) + std::numeric_limits<Distance_>::epsilon()));
    }

    /**
     * Name for loading a `CosineDistance` in the `load_distance_registry()`.
     */
    inline static constexpr const char* cosine_distance_save_name = "knncolle::Cosine";

    /**
     * @brief Compute Cosine distances between two input vectors.
     * @tparam Data_ Numeric type for the input data.
     * @tparam Distance_ Numeric type for the output distance, usually floating-point.
     */
    template <typename Data_, typename Distance_> class CosineDistance final : public DistanceMetric<Data_, Distance_> {
    public:
        /**
         * @cond
         */
        Distance_ raw(std::size_t num_dimensions, const Data_* x, const Data_* y) const override {
            return cosine_distance<Data_, Distance_>(num_dimensions, x, y);
        }

        Distance_ normalize(Distance_ raw) const override {
            return raw;
        }

        Distance_ denormalize(Distance_ norm) const override {
            return norm;
        }

        void save(const std::filesystem::path& dir) const override {
            quick_save(dir / "DISTANCE", cosine_distance_save_name, std::strlen(cosine_distance_save_name));
        }
        /**
         * @endcond
         */
    };

} // namespace knncolle

namespace knncolle_hnsw {

    /**
     * @brief Squared Euclidean distance.
     *
     * @tparam HnswData_ Type of data in the HNSW index, usually floating-point.
     */
    template <typename HnswData_ = float> class CosineDistance final : public hnswlib::SpaceInterface<HnswData_> {
    private:
        std::size_t my_data_size;
        std::size_t my_dim;

    public:
        /**
         * @param dim Number of dimensions over which to compute the distance.
         */
        CosineDistance(std::size_t dim) : my_data_size(dim * sizeof(HnswData_)), my_dim(dim) {}

        /**
         * @cond
         */
    public:
        std::size_t get_data_size() {
            return my_data_size;
        }

        hnswlib::DISTFUNC<HnswData_> get_dist_func() {
            return COSINE;
        }

        void* get_dist_func_param() {
            return &my_dim;
        }

    private:
        static HnswData_ COSINE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
            const HnswData_* pVect1 = static_cast<const HnswData_*>(pVect1v);
            const HnswData_* pVect2 = static_cast<const HnswData_*>(pVect2v);
            const std::size_t qty   = *(static_cast<const std::size_t*>(qty_ptr));
            return knncolle::cosine_distance<HnswData_, HnswData_>(qty, pVect1, pVect2);
        }
        /**
         * @endcond
         */
    };

    /**
     * @tparam Distance_ Floating-point type for the distances.
     * @tparam HnswData_ Type of data in the HNSW index, usually floating-point.
     * @return Configuration for using cosine distances in the HNSW index.
     */
    template <typename Distance_, typename HnswData_ = float>
    DistanceConfig<Distance_, HnswData_> configure_cosine_distance() {
        DistanceConfig<Distance_, HnswData_> output;
        output.create = [](std::size_t dim) -> hnswlib::SpaceInterface<HnswData_>* {
            return static_cast<hnswlib::SpaceInterface<HnswData_>*>(new CosineDistance<HnswData_>(dim));
        };
        output.normalize_method = DistanceNormalizeMethod::SQRT;
        return output;
    }

} // namespace knncolle_hnsw

#endif // CELLCLUSTERING_KNN_EXTENSION_HPP
