find_package(Catch2 3 REQUIRED)

add_executable(CellClusteringTests
    test_distances.cpp
    test_graph.cpp
    test_louvain.cpp
    test_neighbors.cpp
    test_phenograph.cpp
    test_utils.hpp
)

target_include_directories(CellClusteringTests 
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        "${CMAKE_CURRENT_SOURCE_DIR}/../src/"
)
  
target_link_libraries(CellClusteringTests 
    PRIVATE 
        CellClustering::CellClustering 
        Catch2::Catch2WithMain
        knncolle::knncolle
        knncolle::knncolle_annoy
        knncolle::knncolle_hnsw
        knncolle::knncolle_kmknn
)

if(CELLCLUSTERING_BUILD_SHARED_LIBS AND WIN32)
    add_custom_command(TARGET CellClusteringTests POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              $<TARGET_RUNTIME_DLLS:CellClusteringTests>
              $<TARGET_FILE_DIR:CellClusteringTests>
      COMMAND_EXPAND_LISTS
    )
endif()

include(CTest)
include(Catch)
catch_discover_tests(CellClusteringTests)
