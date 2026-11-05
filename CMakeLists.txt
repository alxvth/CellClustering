cmake_minimum_required(VERSION 3.31)

# -----------------------------------------------------------------------------
# User Options
# -----------------------------------------------------------------------------

option(CELLCLUSTERING_BUILD_TESTS "Build the Catch2 test suite" OFF)
option(CELLCLUSTERING_BUILD_EXAMPLES "Build the example program" OFF)
option(CELLCLUSTERING_INSTALL "Generate install rules" OFF)

if(CELLCLUSTERING_BUILD_EXAMPLES)
    list(APPEND VCPKG_MANIFEST_FEATURES "examples")
    message(STATUS "Adding vcpkg feature: examples")
endif()

if(CELLCLUSTERING_BUILD_TESTS)
    list(APPEND VCPKG_MANIFEST_FEATURES "tests")
    message(STATUS "Adding vcpkg feature: tests")
endif()

# -----------------------------------------------------------------------------
# Project
# -----------------------------------------------------------------------------

project(CellClustering 
    VERSION 0.1.0 
    LANGUAGES CXX
    DESCRIPTION "Cell clustering algorithms"
)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

find_package(OpenMP REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS graph)
find_package(knncolle_knncolle CONFIG REQUIRED)
find_package(knncolle_knncolle_annoy CONFIG REQUIRED)
find_package(knncolle_knncolle_hnsw CONFIG REQUIRED)
find_package(knncolle_knncolle_kmknn CONFIG REQUIRED)
find_package(Threads REQUIRED)

# -----------------------------------------------------------------------------
# Library target
# -----------------------------------------------------------------------------

add_library(CellClustering)
add_library(CellClustering::CellClustering ALIAS CellClustering)

set(CELLCLUSTERING_SOURCES
    src/graph.cpp
    src/louvain.cpp
    src/neighbors.cpp
    src/options.cpp
    src/phenograph.cpp
)

set(CELLCLUSTERING_HEADERS_INTERNAL
    src/knn_extension.hpp
)

set(CELLCLUSTERING_HEADERS_API
    include/cellclustering/cellclustering.hpp
    include/cellclustering/graph.hpp
    include/cellclustering/louvain.hpp
    include/cellclustering/neighbors.hpp
    include/cellclustering/options.hpp
    include/cellclustering/phenograph.hpp
    include/cellclustering/types.hpp
)
target_sources(CellClustering
    PRIVATE
        ${CELLCLUSTERING_SOURCES}
        ${CELLCLUSTERING_HEADERS_INTERNAL}

    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS
            include
        FILES
            ${CELLCLUSTERING_HEADERS_API}
)

# -----------------------------------------------------------------------------
# Target linking, properties, etc.
# -----------------------------------------------------------------------------

target_include_directories(CellClustering 
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
    
target_compile_features(CellClustering 
    PUBLIC 
        cxx_std_20
)

target_link_libraries(CellClustering 
    PUBLIC 
        Boost::graph
    PRIVATE 
        knncolle::knncolle_annoy
        knncolle::knncolle_hnsw
        knncolle::knncolle_kmknn
        OpenMP::OpenMP_CXX
        Threads::Threads
)

if(MSVC)
    target_compile_options(CellClustering PRIVATE /MP /W4 /Zc:__cplusplus /Zc:preprocessor)
else()
    target_compile_options(CellClustering PRIVATE -Wall -Wextra -Wpedantic)
endif()

# -----------------------------------------------------------------------------
# Test and example targets
# -----------------------------------------------------------------------------

if(CELLCLUSTERING_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(CELLCLUSTERING_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# -----------------------------------------------------------------------------
# Install
# -----------------------------------------------------------------------------

if(CELLCLUSTERING_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)
        
    # Installation
    install(TARGETS CellClustering
        EXPORT CellClusteringTargets

        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/$<CONFIG>
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/$<CONFIG>
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/$<CONFIG>

        FILE_SET HEADERS
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    # Export targets
    install(EXPORT CellClusteringTargets
        FILE CellClusteringTargets.cmake
        NAMESPACE CellClustering::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CellClustering
    )

    # Package configuration
    configure_package_config_file(
        cmake/CellClusteringConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/CellClusteringConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CellClustering
    )

    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/CellClusteringConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/CellClusteringConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/CellClusteringConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CellClustering
    )
endif()
