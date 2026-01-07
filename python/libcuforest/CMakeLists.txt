# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)

include(rapids-cuda)
rapids_cuda_init_architectures(libcuforest-python)

project(
  libcuforest-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA)

# Check if cuForest is already available. If so, it is the user's responsibility to ensure that the CMake package is
# also available at build time of the Python cuForest package.
find_package(cuforest "${RAPIDS_VERSION}")

if(cuforest_FOUND)
  return()
endif()

unset(cuforest_FOUND)

# --- cuForest --- #
set(BUILD_CUFOREST_TESTS OFF)

set(CUFOREST_CPP_TARGET "cuforest++")
set(CUFOREST_CPP_SRC "../../cpp")

# --- raft --- #
set(CUFOREST_RAFT_CLONE_ON_PIN OFF)
set(CUFOREST_EXCLUDE_RAFT_FROM_ALL ON)

# --- treelite --- #
set(CUFOREST_PYTHON_TREELITE_TARGET treelite::treelite_static)
set(CUFOREST_USE_TREELITE_STATIC ON)
set(CUFOREST_EXCLUDE_TREELITE_FROM_ALL ON)

# --- CUDA --- #
set(CUDA_STATIC_RUNTIME ON)
set(CUDA_STATIC_MATH_LIBRARIES OFF)

add_subdirectory(../../cpp cuforest-cpp)

# assumes libcuforest++ is installed 2 levels deep, e.g.
# site-packages/cuforest/lib64/libcuforest++.so
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  # starting with CTK 13 wheels, all libraries except NCCL are grouped in this 'nvidia/cu13/lib' directory
  set(rpaths "$ORIGIN/../../nvidia/cu13/lib")
else()
  set(rpaths
      "$ORIGIN/../../nvidia/cublas/lib" "$ORIGIN/../../nvidia/cufft/lib" "$ORIGIN/../../nvidia/curand/lib"
      "$ORIGIN/../../nvidia/cusolver/lib" "$ORIGIN/../../nvidia/cusparse/lib" "$ORIGIN/../../nvidia/nvjitlink/lib")
endif()
list(APPEND rpaths "$ORIGIN/../../nvidia/nccl/lib")

set_property(
  TARGET ${CUFOREST_CPP_TARGET}
  PROPERTY INSTALL_RPATH ${rpaths}
  APPEND)
