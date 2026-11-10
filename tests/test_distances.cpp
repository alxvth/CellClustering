#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <stdexcept>

#include <knn_extension.hpp>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("Cosine Distance", "[distance][cosine]") {
    // Cosine Distance defined as: 1.0 - (a . b) / (||a|| * ||b||)

    using namespace knncolle;

    SECTION("Identical vectors -> distance = 0.0") {
        std::array<float, 3> a = {1.0f, 2.0f, 3.0f};
        std::array<float, 3> b = {1.0f, 2.0f, 3.0f};
        const auto dist        = cosine_distance<float, double>(3, a.data(), b.data());
        REQUIRE_THAT(dist, WithinAbs(0.0, 1e-5));

        std::array<float, 3> aNeg = {-1.0f, -2.0f, -3.0f};
        std::array<float, 3> bNeg = {-1.0f, -2.0f, -3.0f};
        const auto distNeg        = cosine_distance<float, double>(3, aNeg.data(), bNeg.data());
        REQUIRE_THAT(distNeg, WithinAbs(0.0, 1e-5));
    }

    SECTION("Parallel vectors (same direction) -> distance = 0.0") {
        std::array<float, 3> a = {1.0f, 2.0f, 3.0f};
        std::array<float, 3> b = {2.0f, 4.0f, 6.0f}; // 2 * a
        const auto dist        = cosine_distance<float, double>(3, a.data(), b.data());
        REQUIRE_THAT(dist, WithinAbs(0.0, 1e-5));
    }

    SECTION("Opposite vectors -> distance = 2.0") {
        std::array<float, 3> a = {1.0f, 2.0f, 3.0f};
        std::array<float, 3> b = {-1.0f, -2.0f, -3.0f};
        const auto dist        = cosine_distance<float, double>(3, a.data(), b.data());
        REQUIRE_THAT(dist, WithinAbs(2.0, 1e-5));
    }

    SECTION("Orthogonal vectors -> distance = 1.0") {
        std::array<float, 3> a = {1.0f, 0.0f, 0.0f};
        std::array<float, 3> b = {0.0f, 1.0f, 0.0f};
        const auto dist        = cosine_distance<float, double>(3, a.data(), b.data());
        REQUIRE_THAT(dist, WithinAbs(1.0, 1e-5));
    }

    SECTION("Partially opposite") {
        std::array<float, 3> a = {1.0f, 0.0f, 0.0f};
        std::array<float, 3> b = {-1.0f, 1.0f, 0.0f};
        const auto dist        = cosine_distance<float, double>(3, a.data(), b.data());
        REQUIRE_THAT(dist, WithinAbs(1.707106781, 1e-5));
    }

    SECTION("Symmetry") {
        std::array<float, 3> a = {4.0f, 5.0f, 6.0f};
        std::array<float, 3> b = {-7.0f, 6.0f, 5.0f};
        const auto dist        = cosine_distance<float, double>(3, a.data(), b.data());
        const auto distSym     = cosine_distance<float, double>(3, b.data(), a.data());
        REQUIRE_THAT(dist, WithinAbs(0.6522971318, 1e-5));
        REQUIRE_THAT(dist, WithinAbs(distSym, 1e-5));
    }

    SECTION("Scale invariance: magnitude does not change distance") {
        std::array<float, 2> a        = {1.0f, 2.0f};
        std::array<float, 2> b        = {3.0f, 1.0f};
        std::array<float, 2> b_scaled = {30.0f, 10.0f};

        const auto distAB        = cosine_distance<float, double>(2, a.data(), b.data());
        const auto distAB_scaled = cosine_distance<float, double>(2, a.data(), b_scaled.data());

        REQUIRE_THAT(distAB, WithinRel(distAB_scaled, 1e-5));
    }

    SECTION("Zero vector handling") {
        // Mathematically not well-defined
        // This implementation chooses to return 1
        std::array<float, 3> zero = {0.0f, 0.0f, 0.0f};
        std::array<float, 3> a    = {1.0f, 2.0f, 3.0f};

        const auto dist = cosine_distance<float, double>(3, zero.data(), a.data());
        REQUIRE_THAT(dist, WithinAbs(1.0, 1e-5));
    }

    SECTION("Empty vector") {
        // Mathematically not well-defined
        // This implementation chooses to return 1
        std::array<float, 0> a = {};
        std::array<float, 0> b = {};

        const auto dist = cosine_distance<float, double>(0, a.data(), b.data());
        REQUIRE_THAT(dist, WithinAbs(1.0, 1e-5));
    }
}
