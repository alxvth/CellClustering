
# -----------------------------------------------------------------------------
# Exmaple: Basic
# -----------------------------------------------------------------------------
find_package(Stb REQUIRED)

add_executable(CellClusteringExampleBlobs example_blobs.cpp)

target_include_directories(CellClusteringExampleBlobs 
    PRIVATE 
        ${Stb_INCLUDE_DIR}
)

target_link_libraries(CellClusteringExampleBlobs 
    PRIVATE 
        CellClustering::CellClustering
)

target_compile_features(CellClusteringExampleBlobs 
    PUBLIC 
        cxx_std_20
)

if(MSVC)
    target_compile_options(CellClusteringExampleBlobs PRIVATE /MP /W4 /Zc:__cplusplus /Zc:preprocessor)
else()
    target_compile_options(CellClusteringExampleBlobs PRIVATE -Wall -Wextra -Wpedantic)
endif()
