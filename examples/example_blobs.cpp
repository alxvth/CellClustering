// Usage: cellclustering_example
// Recreates:
// https://github.com/dpeerlab/PhenoGraph/blob/72a3e624345f105727046a3e835d845d02014654/examples/example-using-random-data-points.ipynb

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <cellclustering/phenograph.hpp>

namespace {

    struct RGB {
        unsigned char r;
        unsigned char g;
        unsigned char b;
    };

    void plotScatter(const std::vector<float>& X, const std::vector<RGB>& colors, const char* filename);
    std::vector<RGB> assignClusterColors(const std::vector<std::size_t>& clusterIds);
    std::vector<float> createBlobData(const std::size_t nSamples, const std::size_t nDim);

} // namespace

int main() {

    // random points: three blobs
    constexpr std::size_t nSamples = 150;
    constexpr std::size_t nDim     = 2;

    const std::vector<float> X = createBlobData(nSamples, nDim);

    plotScatter(X, {}, "example_blobs_01_data.png");

    cellclustering::Options opt;
    opt.nnMethod = cellclustering::NNMethod::Annoy;
    opt.seed     = 1;
    opt.verbose  = true;

    const auto t0         = std::chrono::steady_clock::now();
    const auto dataMatrix = cellclustering::DenseMatrixView(X, nSamples, nDim);
    const auto res        = cellclustering::phenograph(dataMatrix, opt);
    const auto dt         = std::chrono::steady_clock::now() - t0;

    plotScatter(X, assignClusterColors(res.communities), "example_blobs_02_communities.png");

    std::map<cellclustering::IndexType, cellclustering::IndexType> sizes;
    for (auto c : res.communities)
        ++sizes[c];

    std::cout << "total " << dt.count() << " s, Q = " << res.Q
              << ", graph edges = " << boost::num_edges(res.graph.value()) << "\n";
    for (auto& [label, n] : sizes)
        std::cout << "  cluster " << label << ": " << n << " cells\n";

    return 0;
}

namespace {
    void plotScatter(const std::vector<float>& X, const std::vector<RGB>& colors, const char* filename) {
        constexpr int width         = 600;
        constexpr int height        = 600;
        constexpr int imgSize       = width * height * 3;
        const std::size_t numPoints = X.size() / 2;
        const bool hasColors        = colors.size() == numPoints;

        constexpr unsigned char defaultR = 30;
        constexpr unsigned char defaultG = 100;
        constexpr unsigned char defaultB = 200;

        std::vector<unsigned char> image(imgSize, 255);

        // Determine plot bounds
        const auto [min_it, max_it] = std::ranges::minmax_element(X);

        float xmin = *min_it;
        float xmax = *max_it;

        float ymin = xmin;
        float ymax = xmax;

        for (std::size_t i = 0; i < X.size() / 2; ++i) {
            ymin = std::min(ymin, X[i * 2 + 1]);
            ymax = std::max(ymax, X[i * 2 + 1]);
        }

        constexpr float margin = 0.1f;
        const float dx         = (xmax - xmin) * margin;
        const float dy         = (ymax - ymin) * margin;

        xmin -= dx;
        xmax += dx;
        ymin -= dy;
        ymax += dy;

        // Plot points
        for (std::size_t i = 0; i < X.size() / 2; ++i) {
            const int px = static_cast<int>((X[i * 2] - xmin) / (xmax - xmin) * (width - 1));
            const int py = static_cast<int>((ymax - X[i * 2 + 1]) / (ymax - ymin) * (height - 1));

            // Draw a small black circle
            for (int y = -3; y <= 3; ++y) {
                for (int x = -3; x <= 3; ++x) {
                    if (x * x + y * y <= 9) {
                        const int xx = px + x;
                        const int yy = py + y;

                        if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                            const std::size_t idx = (yy * width + xx) * 3;

                            image[idx]     = hasColors ? colors[i].r : defaultR;
                            image[idx + 1] = hasColors ? colors[i].g : defaultG;
                            image[idx + 2] = hasColors ? colors[i].b : defaultB;
                        }
                    }
                }
            }
        }

        stbi_write_png(filename, width, height, 3, image.data(), width * 3);
    }

    std::vector<float> createBlobData(const std::size_t nSamples, const std::size_t nDim) {

        constexpr std::array<std::array<float, 2>, 3> centers = {{{-2.0f, -2.0f}, {-3.0f, 3.0f}, {2.0f, 2.0f}}};

        constexpr std::array<float, 3> clusterStd = {0.8f, 1.0f, 1.2f};

        const std::size_t samplesPerCluster = nSamples / centers.size();

        const auto seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
        std::cout << "PRNG Seed: " << seed << "\n";
        std::mt19937 gen(seed);

        std::vector<float> X;
        X.reserve(samplesPerCluster);

        for (std::size_t c = 0; c < centers.size(); ++c) {
            std::normal_distribution<float> dist(0.0f, clusterStd[c]);

            for (std::size_t i = 0; i < samplesPerCluster; ++i) {
                for (std::size_t d = 0; d < nDim; ++d) {
                    X.push_back(centers[c][d] + dist(gen));
                }
            }
        }

        // Shuffle samples
        {
            std::vector<std::size_t> indices(nSamples);
            std::iota(indices.begin(), indices.end(), 0);
            std::ranges::shuffle(indices, gen);

            std::vector<float> XShuffled;
            XShuffled.reserve(X.size());

            for (const std::size_t i : indices) {
                std::copy_n(X.begin() + i * nDim, nDim, std::back_inserter(XShuffled));
            }

            X = std::move(XShuffled);
        }

        return X;
    }

    std::vector<RGB> assignClusterColors(const std::vector<std::size_t>& clusterIds) {
        static const std::vector<RGB> palette = {
            {230, 25, 75},   // Red
            {60, 180, 75},   // Green
            {0, 130, 200},   // Blue
            {245, 130, 48},  // Orange
            {145, 30, 180},  // Purple
            {70, 240, 240},  // Cyan
            {240, 50, 230},  // Magenta
            {210, 245, 60},  // Lime
            {250, 190, 190}, // Pink
            {0, 128, 128},   // Teal
            {230, 190, 255}, // Lavender
            {170, 110, 40}   // Brown
        };

        std::unordered_map<std::size_t, RGB> clusterColors;
        std::vector<RGB> pointColors;
        pointColors.reserve(clusterIds.size());

        for (const auto clusterId : clusterIds) {
            auto [it, inserted] = clusterColors.try_emplace(clusterId, palette[clusterColors.size() % palette.size()]);

            pointColors.push_back(it->second);
        }

        return pointColors;
    }
} // namespace