# CellClustering

A small C++20 library implementing several (as of now just one) cell clustering algorithms:

- PhenoGraph ([Levine et al., Cell 2015](https://doi.org/10.1016/j.cell.2015.05.047))
  - kNN graph `->` Jaccard (or Gaussian) weights `->` symmetrization `->` community detection
  - other implementation: [dpeerlab/PhenoGraph](https://github.com/dpeerlab/PhenoGraph)

## Build

```sh
cmake -B build -S . "-DCMAKE_TOOLCHAIN_FILE=[path to vcpkg]/scripts/buildsystems/vcpkg.cmake"
cmake --build build
```

or

```sh
export VCPKG_ROOT=/path/to/vcpkg
cmake --preset default
cmake --build --preset default
ctest --preset default
```

## Usage

```cpp
#include <cellclustering/cellclustering.hpp>

cellclustering::Options opt;
opt.k = 30;
opt.primary_metric = cellclustering::Metric::Cosine;
opt.nn_method = cellclustering::NNMethod::Annoy;
opt.seed = 1;

// X is a std::vector<float> of size nSamples * nDim
const auto dataMatrix = cellclustering::DenseMatrixView(X, nSamples, nDim);
cellclustering::Result r = cellclustering::cluster(dataMatrix, opt);
// r.communities : one label per row, outliers marked with cellclustering::NOT_ASSIGNED
// r.graph       : graph used for community detection (optionally returned)
// r.Q           : modularity
```


## Options
Similar to the options from the Python implementation in [dpeerlab/PhenoGraph](https://github.com/dpeerlab/PhenoGraph):

| `Options` | Default | Notes |
|---|---|---|
| `nnMethod` | `HNSW` | `HNSW`, `Annoy`, `KMKNN` (approximate or exact depending on backend), `BruteForce` (exact) |
| `nnMetric` | `Euclidean` | `Euclidean`, `Manhattan`, `Cosine` |
| `k` | 30 | number of nearest neighbors (must be >= 1 and smaller than the number of points) |
| `clusteringAlgo` | `Louvain` | [`Louvain`](https://en.wikipedia.org/wiki/Louvain_method), TODO: implement [`Leiden`](https://en.wikipedia.org/wiki/Leiden_algorithm) |
| `directed`, `prune` | `false`, `false` | `prune` forces `directed = false` |
| `minClusterSize` | 10 | clusters with size **<=** it are labelled `NOT_ASSIGNED` (`std::numeric_limits<std::size_t>::max()`) |
| `jaccard` | `true` | `false` -> Gaussian kernel, sigma 1 |
| `returnGraph` | `true` | whether the graph is returned with the clustering |
| `nJobs` | -1 | `< 0` -> all hardware threads, `0` -> single thread, `> 0` -> `min(nJobs, hardware threads)` |
| `louvainParams.qTol`, `louvainParams.timeLimit` | 1e-3, 2000 s | restart policy as `runlouvain`; a restart is accepted if it beats the best Q by more than `qTol` |
| `louvainParams.maxRuns`, `louvainParams.patience` | 100, 20 | stop after `maxRuns` runs, or `patience` runs without acceptance |
| `hnsw` (`numLinks`, `efConstruction`, `efSearch`) | 16, 200, 10 | hnswlib settings; `numLinks` is the `M` parameter |
| `annoy` (`nTrees`, `searchK`) | 50, -1 | `searchK = -1` -> Annoy default |
| `kmknn` (`power`) | 0.5 | number of cluster centers is `n^power`, power in (0, 1] |
| `seed` | none | seeds ANN builds and Louvain restarts |
| `verbose` | `false` | progress messages on stderr |

