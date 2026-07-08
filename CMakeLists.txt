cmake_minimum_required(VERSION 3.15)

# Project name
project(PACSProject LANGUAGES CXX)

# ============================================================================
# Find deal.II
# ============================================================================
find_package(deal.II 9.5.0
  COMPONENTS PETSc
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../dealii-install $ENV{DEAL_II_DIR}
  REQUIRED
)

# ============================================================================
# Find Eigen3
# ============================================================================
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/eigen3")
if(NOT EXISTS ${EIGEN3_INCLUDE_DIR}/Eigen)
  message(FATAL_ERROR "Eigen3 not found at ${EIGEN3_INCLUDE_DIR}")
endif()
message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")

# ============================================================================
# Find OpenMP
# ============================================================================
if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  execute_process(
    COMMAND brew --prefix libomp
    OUTPUT_VARIABLE LIBOMP_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
  set(OpenMP_CXX_LIB_NAMES "omp")
  set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
endif()
find_package(OpenMP REQUIRED COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# ============================================================================
# Initialize deal.II
# ============================================================================
deal_ii_initialize_cached_variables()

# ============================================================================
# Include directories
# ============================================================================
include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIR})

# ============================================================================
# Unified AMG Data Generator
# ============================================================================
set(TARGET_UNIFIED "generate_amg_data")
add_executable(${TARGET_UNIFIED} src/generate_amg_data.cpp)
deal_ii_setup_target(${TARGET_UNIFIED})
target_link_libraries(${TARGET_UNIFIED} ${OpenMP_CXX_LIBRARIES})
target_include_directories(${TARGET_UNIFIED} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ============================================================================
# Build summary
# ============================================================================
message(STATUS "========================================")
message(STATUS "Build configured successfully!")
message(STATUS "========================================")
message(STATUS "")
message(STATUS "Target executables:")
message(STATUS "  generate_amg_data - Unified AMG data generator")
message(STATUS "")
message(STATUS "Supported problem types:")
message(STATUS "  D  - Diffusion equations (FEM)")
message(STATUS "  E  - Elastic equations (FEM)")
message(STATUS "  S  - Stokes equations (FEM)")
message(STATUS "  GL - Graph Laplacian (random graphs)")
message(STATUS "  SC - Spectral Clustering (k-NN graphs)")
message(STATUS "")
message(STATUS "Libraries:")
message(STATUS "  Eigen3: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "  OpenMP: ${OpenMP_CXX_FLAGS}")
message(STATUS "")
message(STATUS "Usage example:")
message(STATUS "  ./generate_amg_data -p D -f all -c large --use-npy")
message(STATUS "")
message(STATUS "Run './generate_amg_data --help' for full options")
message(STATUS "========================================")
