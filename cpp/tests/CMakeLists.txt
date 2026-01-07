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
enable_testing()

# We use rapids-cmake testing infrastructure to allow us to run multiple GPU tests concurrently without causing OOM
# issues. Use the `GPUS` and `PERCENT` options to control how 'much' of the GPUs you need for a test:
#
# GPUS 1 PERCENT 25 -> I need 25% of a single GPU GPUS 1 PERCENT 100 -> all of 1 GPU GPUS 2 PERCENT 200 -> all of 2 GPUs
# (will only run this test on 2 GPU machines)
include(rapids-test)
rapids_test_init()

function(ConfigureTest)

  set(options)
  set(one_value NAME GPUS PERCENT)
  set(multi_value TARGETS CONFIGURATIONS)
  cmake_parse_arguments(_CUFOREST_TEST "${options}" "${one_value}" "${multi_value}" ${ARGN})
  if(NOT DEFINED _CUFOREST_TEST_GPUS AND NOT DEFINED _CUFOREST_TEST_PERCENT)
    set(_CUFOREST_TEST_GPUS 1)
    set(_CUFOREST_TEST_PERCENT 15)
  endif()
  if(NOT DEFINED _CUFOREST_TEST_GPUS)
    set(_CUFOREST_TEST_GPUS 1)
  endif()
  if(NOT DEFINED _CUFOREST_TEST_PERCENT)
    set(_CUFOREST_TEST_PERCENT 100)
  endif()

  add_executable(${_CUFOREST_TEST_NAME} ${_CUFOREST_TEST_UNPARSED_ARGUMENTS})
  target_link_libraries(
    ${_CUFOREST_TEST_NAME}
    PRIVATE ${CUFOREST_CPP_TARGET}
            rmm::rmm
            raft::raft
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
            ${OpenMP_CXX_LIB_NAMES}
            Threads::Threads
            ${TREELITE_LIBS}
            $<TARGET_NAME_IF_EXISTS:conda_env>)

  target_compile_options(${_CUFOREST_TEST_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUFOREST_CXX_FLAGS}>"
                                                        "$<$<COMPILE_LANGUAGE:CUDA>:${CUFOREST_CUDA_FLAGS}>")

  target_include_directories(${_CUFOREST_TEST_NAME} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
                                                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src>)
  set_target_properties(
    ${_CUFOREST_TEST_NAME}
    PROPERTIES INSTALL_RPATH "\$ORIGIN/../../../lib"
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON)

  set(_CUFOREST_TEST_COMPONENT_NAME testing)

  rapids_test_add(
    NAME ${_CUFOREST_TEST_NAME}
    COMMAND ${_CUFOREST_TEST_NAME}
    GPUS ${_CUFOREST_TEST_GPUS}
    PERCENT ${_CUFOREST_TEST_PERCENT}
    INSTALL_COMPONENT_SET ${_CUFOREST_TEST_COMPONENT_NAME})

endfunction()

# ######################################################################################################################
# * Build ml_test executable -------------------------------------------------
ConfigureTest(NAME HOST_BUFFER_TEST raft_proto/buffer.cpp)
ConfigureTest(NAME DEVICE_BUFFER_TEST raft_proto/buffer.cu)
ConfigureTest(NAME FOREST_TRAVERSAL_TEST forest/traversal_forest.cpp)
ConfigureTest(NAME TREELITE_TRAVERSAL_TEST forest/treelite_traversal.cpp)
ConfigureTest(NAME TREELITE_IMPORTER_TEST treelite_importer.cpp)

rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/gtests/libcuforest)
