# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)

set(language_list "CXX")

include(rapids-cuda)
rapids_cuda_init_architectures(nvforest-python)
list(APPEND language_list "CUDA")

project(
  nvforest-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES ${language_list})

# ######################################################################################################################
# * User Options  --------------------------------------------------------------
option(USE_LIBNVFOREST_WHEEL "Use libnvforest wheel to provide some dependencies" OFF)

set(NVFOREST_CPP_TARGET "nvforest++")
set(NVFOREST_CPP_SRC "../../cpp")

# ######################################################################################################################
# * Process User Options  ------------------------------------------------------

include(rapids-cpm)
include(rapids-export)
rapids_cpm_init()

# --- treelite --- # Need to call get_treelite explicitly because we need the correct ${TREELITE_LIBS} definition for
# RF.
#
# And because nvforest Cython code needs the headers to satisfy calls like 'cdef extern from "treelite/c_api.h"'

# wheel builds use a static treelite, because the 'libtreelite.so' in 'treelite' wheels isn't intended for dynamic
# linking by third-party projects (e.g. hides its symbols)
if(USE_LIBNVFOREST_WHEEL)
  set(NVFOREST_PYTHON_TREELITE_TARGET treelite::treelite_static)
  set(NVFOREST_USE_TREELITE_STATIC ON)
else()
  set(NVFOREST_PYTHON_TREELITE_TARGET treelite::treelite)
  set(NVFOREST_USE_TREELITE_STATIC OFF)
endif()

set(NVFOREST_EXCLUDE_TREELITE_FROM_ALL ON)

include(${NVFOREST_CPP_SRC}/cmake/thirdparty/get_treelite.cmake)

# --- libnvforest --- #
find_package(nvforest "${RAPIDS_VERSION}" REQUIRED)

# ######################################################################################################################
# * Build Cython artifacts -----------------------------------------------------

message(VERBOSE "NVFOREST_PY: Building nvForest")

include(rapids-cython-core)
rapids_cython_init()

set(linked_libraries "nvforest::${NVFOREST_CPP_TARGET}" "${NVFOREST_PYTHON_TREELITE_TARGET}")

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "nvforest/detail/forest_inference.pyx"
  LINKED_LIBRARIES "${linked_libraries}"
  INSTALL_DIR nvforest/detail MODULE_PREFIX nvforest_)

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "nvforest/detail/treelite.pyx"
  LINKED_LIBRARIES "${linked_libraries}"
  INSTALL_DIR nvforest/detail MODULE_PREFIX nvforest_)
