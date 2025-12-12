# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)

set(language_list "CXX")

include(rapids-cuda)
rapids_cuda_init_architectures(cuforest-python)
list(APPEND language_list "CUDA")

project(
  cuforest-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES ${language_list}
)

# ##################################################################################################
# * User Options  --------------------------------------------------------------
option(USE_LIBCUFOREST_WHEEL "Use libcuforest wheel to provide some dependencies" OFF)

set(CUFOREST_CPP_TARGET "cuforest++")
set(CUFOREST_CPP_SRC "../../cpp")

# ##################################################################################################
# * Process User Options  ------------------------------------------------------

include(rapids-cpm)
include(rapids-export)
rapids_cpm_init()

# --- treelite --- # Need to call get_treelite explicitly because we need the correct
# ${TREELITE_LIBS} definition for RF.
#
# And because cuforest Cython code needs the headers to satisfy calls like 'cdef extern from
# "treelite/c_api.h"'

# wheel builds use a static treelite, because the 'libtreelite.so' in 'treelite' wheels isn't
# intended for dynamic linking by third-party projects (e.g. hides its symbols)
if(USE_LIBCUFOREST_WHEEL)
  set(CUFOREST_PYTHON_TREELITE_TARGET treelite::treelite_static)
  set(CUFOREST_USE_TREELITE_STATIC ON)
else()
  set(CUFOREST_PYTHON_TREELITE_TARGET treelite::treelite)
  set(CUFOREST_USE_TREELITE_STATIC OFF)
endif()

set(CUFOREST_EXCLUDE_TREELITE_FROM_ALL ON)

include(${CUFOREST_CPP_SRC}/cmake/thirdparty/get_treelite.cmake)

# --- libcuforest --- #
find_package(cuforest "${RAPIDS_VERSION}" REQUIRED)

# ##################################################################################################
# * Build Cython artifacts -----------------------------------------------------

message(VERBOSE "CUFOREST_PY: Building cuForest")

include(rapids-cython-core)
rapids_cython_init()

set(linked_libraries "cuforest::${CUFOREST_CPP_TARGET}" "${CUFOREST_PYTHON_TREELITE_TARGET}")

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "cuforest/detail/forest_inference.pyx"
  LINKED_LIBRARIES "${linked_libraries}"
  INSTALL_DIR cuforest/detail
  MODULE_PREFIX cuforest_
)

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "cuforest/detail/treelite.pyx"
  LINKED_LIBRARIES "${linked_libraries}"
  INSTALL_DIR cuforest/detail
  MODULE_PREFIX cuforest_
)
