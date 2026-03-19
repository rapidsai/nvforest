# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION. SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
# =============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../cmake/rapids_config.cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(NVForest)

project(
  NVForest
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA)

# Write the version header
rapids_cmake_write_version_file(include/nvforest/version_config.hpp)

# ######################################################################################################################
# * build type ---------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# this is needed for clang-tidy runs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ######################################################################################################################

# ######################################################################################################################
# * User Options  ------------------------------------------------------------

option(NVFOREST_ENABLE_GPU "Enable building GPU-accelerated algorithms" ON)
option(BUILD_SHARED_LIBS "Build nvForest shared libraries" ON)
option(BUILD_NVFOREST_TESTS "Build nvForest algorithm tests" ON)
option(CUDA_ENABLE_KERNEL_INFO "Enable kernel resource usage info" OFF)
option(CUDA_ENABLE_LINE_INFO "Enable lineinfo in nvcc" OFF)
option(DETECT_CONDA_ENV "Enable detection of conda environment for dependencies" ON)
option(DISABLE_DEPRECATION_WARNINGS "Disable deprecation warnings " ON)
option(DISABLE_OPENMP "Disable OpenMP" OFF)
option(NVTX "Enable nvtx markers" OFF)
option(USE_CCACHE "Cache build artifacts with ccache" OFF)
option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)
option(NVFOREST_USE_RAFT_STATIC "Build and statically link the RAFT library" OFF)
option(NVFOREST_USE_TREELITE_STATIC "Build and statically link the treelite library" OFF)
option(NVFOREST_EXPORT_TREELITE_LINKAGE "Whether to publicly or privately link treelite to libnvforest++" OFF)
option(CUDA_WARNINGS_AS_ERRORS "Enable -Werror=all-warnings for all CUDA compilation" ON)

# The options below allow incorporating libnvforest into another build process without installing all its components.
# This is useful if total file size is at a premium and we do not expect other consumers to use any APIs of the
# dependency except those that are directly linked to by the dependent library.
option(NVFOREST_EXCLUDE_RAFT_FROM_ALL "Exclude RAFT targets from nvForest's 'all' target" OFF)
option(NVFOREST_EXCLUDE_TREELITE_FROM_ALL "Exclude Treelite targets from nvForest's 'all' target" OFF)
option(NVFOREST_RAFT_CLONE_ON_PIN "Explicitly clone RAFT branch when pinned to non-feature branch" ON)

message(VERBOSE "NVFOREST_CPP: Building nvForest algorithm tests: ${BUILD_NVFOREST_TESTS}")
message(VERBOSE "NVFOREST_CPP: Enabling detection of conda environment for dependencies: ${DETECT_CONDA_ENV}")
message(VERBOSE "NVFOREST_CPP: Disabling OpenMP: ${DISABLE_OPENMP}")
message(VERBOSE "NVFOREST_CPP: Enabling kernel resource usage info: ${KERNEL_INFO}")
message(VERBOSE "NVFOREST_CPP: Enabling kernelinfo in nvcc: ${CUDA_ENABLE_KERNEL_INFO}")
message(VERBOSE "NVFOREST_CPP: Enabling lineinfo in nvcc: ${CUDA_ENABLE_LINE_INFO}")
message(VERBOSE "NVFOREST_CPP: Enabling nvtx markers: ${NVTX}")
message(VERBOSE "NVFOREST_CPP: Cache build artifacts with ccache: ${USE_CCACHE}")
message(VERBOSE "NVFOREST_CPP: Statically link the CUDA runtime: ${CUDA_STATIC_RUNTIME}")
message(VERBOSE "NVFOREST_CPP: Build and statically link RAFT library: ${NVFOREST_USE_RAFT_STATIC}")
message(VERBOSE "NVFOREST_CPP: Build and statically link Treelite library: ${NVFOREST_USE_TREELITE_STATIC}")

# Set RMM logging level
set(RMM_LOGGING_LEVEL
    "INFO"
    CACHE STRING "Choose the logging level.")
set_property(
  CACHE RMM_LOGGING_LEVEL
  PROPERTY STRINGS
           "TRACE"
           "DEBUG"
           "INFO"
           "WARN"
           "ERROR"
           "CRITICAL"
           "OFF")
message(VERBOSE "NVFOREST_CPP: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

# Set logging level
set(LIBNVFOREST_LOGGING_LEVEL
    "DEBUG"
    CACHE STRING "Choose the logging level.")
set_property(
  CACHE LIBNVFOREST_LOGGING_LEVEL
  PROPERTY STRINGS
           "TRACE"
           "DEBUG"
           "INFO"
           "WARN"
           "ERROR"
           "CRITICAL"
           "OFF")
message(VERBOSE "NVFOREST: LIBNVFOREST_LOGGING_LEVEL = '${LIBNVFOREST_LOGGING_LEVEL}'.")

if(BUILD_NVFOREST_TESTS OR BUILD_PRIMS_TESTS)
  # Needed because GoogleBenchmark changes the state of FindThreads.cmake, causing subsequent runs to have different
  # values for the `Threads::Threads` target. Setting this flag ensures `Threads::Threads` is the same value in first
  # run and subsequent runs.
  set(THREADS_PREFER_PTHREAD_FLAG ON)
endif()

# ######################################################################################################################
# * Target names -------------------------------------------------------------

set(NVFOREST_CPP_TARGET "nvforest++")

# ######################################################################################################################
# * Conda environment detection ----------------------------------------------

if(DETECT_CONDA_ENV)
  rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND DEFINED ENV{CONDA_PREFIX})
    message(STATUS "NVFOREST_CPP: No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}")
    set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
  endif()
endif()

# ######################################################################################################################
# * compiler options ---------------------------------------------------------

if(NOT DISABLE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_FOUND)
    message(STATUS "NVFOREST_CPP: OpenMP found in ${OPENMP_INCLUDE_DIRS}")
    list(APPEND NVFOREST_CXX_FLAGS ${OpenMP_CXX_FLAGS})
  endif()
endif()

# CUDA runtime
rapids_cuda_init_runtime(USE_STATIC ${CUDA_STATIC_RUNTIME})

# * find CUDAToolkit package
# * determine GPU architectures
# * enable the CMake CUDA language
# * set other CUDA compilation flags
rapids_find_package(
  CUDAToolkit REQUIRED
  BUILD_EXPORT_SET nvforest-exports
  INSTALL_EXPORT_SET nvforest-exports)
include(cmake/modules/ConfigureCUDA.cmake)

if(USE_CCACHE)
  set(CMAKE_C_COMPILER_LAUNCHER ccache)
  set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
  set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
endif()

# ######################################################################################################################
# * Requirements -------------------------------------------------------------

# add third party dependencies using CPM
rapids_cpm_init()
rapids_cmake_install_lib_dir(lib_dir)

include(${rapids-cmake-dir}/cpm/rapids_logger.cmake)
rapids_cpm_rapids_logger(BUILD_EXPORT_SET nvforest-exports INSTALL_EXPORT_SET nvforest-exports)
create_logger_macros(NVFOREST "nvforest::default_logger()" include/nvforest/common)

if(BUILD_NVFOREST_TESTS)
  find_package(Threads)
endif()

# CCCL before RMM, and RMM before RAFT
include(cmake/thirdparty/get_cccl.cmake)
include(cmake/thirdparty/get_rmm.cmake)
include(cmake/thirdparty/get_raft.cmake)
include(cmake/thirdparty/get_treelite.cmake)

if(BUILD_NVFOREST_TESTS)
  include(${rapids-cmake-dir}/cpm/gtest.cmake)
  rapids_cpm_gtest(BUILD_STATIC)
endif()

# ######################################################################################################################
# * build libnvforest++ shared library -------------------------------------------

file(
  WRITE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld"
  [=[
SECTIONS
{
.nvFatBinSegment : { *(.nvFatBinSegment) }
.nv_fatbin : { *(.nv_fatbin) }
}
]=])

# Copy the interface include directories from INCLUDED_TARGET to TARGET. This is necessary when INCLUDED_TARGET was
# compiled statically but includes public APIs that may still require consumers to have the same interface headers
# available.
function(copy_interface_excludes)
  set(_options "")
  set(_one_value TARGET INCLUDED_TARGET)
  set(_multi_value "")
  cmake_parse_arguments(_NVFOREST_INCLUDES "${_options}" "${_one_value}" "${_multi_value}" ${ARGN})
  get_target_property(_includes ${_NVFOREST_INCLUDES_INCLUDED_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
  target_include_directories(${_NVFOREST_INCLUDES_TARGET} PUBLIC ${_includes})
endfunction()

add_library(${NVFOREST_CPP_TARGET})
if(NVFOREST_ENABLE_GPU)
  target_compile_definitions(${NVFOREST_CPP_TARGET} PUBLIC NVFOREST_ENABLE_GPU)
  target_sources(
    ${NVFOREST_CPP_TARGET}
    PRIVATE src/infer0.cu
            src/infer1.cu
            src/infer2.cu
            src/infer3.cu
            src/infer4.cu
            src/infer5.cu
            src/infer6.cu
            src/infer7.cu
            src/infer8.cu
            src/infer9.cu
            src/infer10.cu
            src/infer11.cu)
endif()
target_sources(
  ${NVFOREST_CPP_TARGET}
  PRIVATE src/infer0.cpp
          src/infer1.cpp
          src/infer2.cpp
          src/infer3.cpp
          src/infer4.cpp
          src/infer5.cpp
          src/infer6.cpp
          src/infer7.cpp
          src/infer8.cpp
          src/infer9.cpp
          src/infer10.cpp
          src/infer11.cpp)

add_library(nvforest::${NVFOREST_CPP_TARGET} ALIAS ${NVFOREST_CPP_TARGET})

set_target_properties(
  ${NVFOREST_CPP_TARGET}
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             # set target compile options
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(${NVFOREST_CPP_TARGET} PRIVATE NVFOREST_CPP_API)

target_compile_options(${NVFOREST_CPP_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${NVFOREST_CXX_FLAGS}>"
                                                      "$<$<COMPILE_LANGUAGE:CUDA>:${NVFOREST_CUDA_FLAGS}>")
target_compile_definitions(${NVFOREST_CPP_TARGET}
                           PUBLIC "NVFOREST_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_${LIBNVFOREST_LOGGING_LEVEL}")

target_include_directories(
  ${NVFOREST_CPP_TARGET}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  INTERFACE $<INSTALL_INTERFACE:include>)

set(_nvforest_cpp_public_libs)
set(_nvforest_cpp_private_libs)

if(NVFOREST_USE_RAFT_STATIC AND (TARGET raft::raft))
  copy_interface_excludes(INCLUDED_TARGET raft::raft TARGET ${NVFOREST_CPP_TARGET})
endif()

if(NVFOREST_USE_TREELITE_STATIC AND (TARGET treelite::treelite_static))
  set(TREELITE_LIBS treelite::treelite_static)
  copy_interface_excludes(INCLUDED_TARGET treelite::treelite_static TARGET ${NVFOREST_CPP_TARGET})
elseif(NVFOREST_EXPORT_TREELITE_LINKAGE)
  list(APPEND _nvforest_cpp_public_libs ${TREELITE_LIBS})
endif()

# These are always private:
list(APPEND _nvforest_cpp_private_libs ${TREELITE_LIBS} ${OpenMP_CXX_LIB_NAMES})

set(_nvforest_cpp_libs_var_name "_nvforest_cpp_public_libs")
if(CUDA_STATIC_RUNTIME)
  set(_nvforest_cpp_libs_var_name "_nvforest_cpp_private_libs")
  # Add CTK include paths because we're going to make our CTK library links private below
  target_include_directories(${NVFOREST_CPP_TARGET} SYSTEM PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
endif()

target_link_libraries(
  ${NVFOREST_CPP_TARGET}
  PUBLIC rapids_logger::rapids_logger rmm::rmm raft::raft ${_nvforest_cpp_public_libs}
  PRIVATE ${_nvforest_cpp_private_libs})

# Ensure CUDA symbols aren't relocated to the middle of the debug build binaries
target_link_options(${NVFOREST_CPP_TARGET} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld")

# ######################################################################################################################
# * build test executables ---------------------------------------------------

if(BUILD_NVFOREST_TESTS)
  include(CTest)
  # TODO(hcho3): Add tests
  add_subdirectory(tests)
endif()

# ######################################################################################################################
# # - install targets -------------------------------------------------------------------------------
include(CPack)

set(NVFOREST_TARGETS ${NVFOREST_CPP_TARGET})

install(
  TARGETS ${NVFOREST_TARGETS}
  DESTINATION ${lib_dir}
  EXPORT nvforest-exports)

install(DIRECTORY include/nvforest/ DESTINATION include/nvforest)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/nvforest/version_config.hpp DESTINATION include/nvforest)

# ######################################################################################################################
# # - install export -------------------------------------------------------------------------------
set(doc_string
    [=[
Provide targets for nvForest.

nvForest Inference Library provides an optimized library
for running inference with decision tree models on NVIDIA GPUs.

]=])

set(code_string)

if(TARGET treelite::treelite)
  string(
    APPEND
    code_string
    [=[
if (TARGET treelite::treelite AND (NOT TARGET treelite))
    add_library(treelite ALIAS treelite::treelite)
endif()
]=])
else()
  string(
    APPEND
    code_string
    [=[
if (TARGET treelite::treelite_static AND (NOT TARGET treelite_static))
    add_library(treelite_static ALIAS treelite::treelite_static)
endif()
]=])

endif()

rapids_export(
  INSTALL nvforest
  EXPORT_SET nvforest-exports
  GLOBAL_TARGETS ${NVFOREST_C_TARGET} ${NVFOREST_CPP_TARGET}
  NAMESPACE nvforest::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)

# ######################################################################################################################
# * build export -------------------------------------------------------------------------------

rapids_export(
  BUILD nvforest
  EXPORT_SET nvforest-exports
  GLOBAL_TARGETS ${NVFOREST_C_TARGET} ${NVFOREST_CPP_TARGET}
  NAMESPACE nvforest::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)

# ######################################################################################################################
# * doxygen targets ----------------------------------------------------------

include(cmake/doxygen.cmake)
add_doxygen_target(IN_DOXYFILE doxygen/Doxyfile.in OUT_DOXYFILE ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/Doxyfile CWD
                   ${CMAKE_CURRENT_SOURCE_DIR}/doxygen)
