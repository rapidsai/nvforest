# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)

include(rapids-cuda)
rapids_cuda_init_architectures(libnvforest-python)

project(
  libnvforest-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA)

# Check if nvForest is already available. If so, it is the user's responsibility to ensure that the CMake package is
# also available at build time of the Python nvForest package.
find_package(nvforest "${RAPIDS_VERSION}")

if(nvforest_FOUND)
  return()
endif()

unset(nvforest_FOUND)

# --- nvForest --- #
set(BUILD_NVFOREST_TESTS OFF)

set(NVFOREST_CPP_TARGET "nvforest++")
set(NVFOREST_CPP_SRC "../../cpp")

# --- raft --- #
set(NVFOREST_RAFT_CLONE_ON_PIN OFF)
set(NVFOREST_EXCLUDE_RAFT_FROM_ALL ON)

# --- treelite --- #
set(NVFOREST_PYTHON_TREELITE_TARGET treelite::treelite_static)
set(NVFOREST_USE_TREELITE_STATIC ON)
set(NVFOREST_EXCLUDE_TREELITE_FROM_ALL ON)

# --- CUDA --- #
set(CUDA_STATIC_RUNTIME ON)

add_subdirectory(../../cpp nvforest-cpp)

set_property(
  TARGET ${NVFOREST_CPP_TARGET}
  PROPERTY INSTALL_RPATH ${rpaths}
  APPEND)
