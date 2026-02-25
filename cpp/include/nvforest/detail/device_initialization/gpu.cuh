/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvforest/constants.hpp>
#include <nvforest/detail/forest.hpp>
#include <nvforest/detail/gpu_introspection.hpp>
#include <nvforest/detail/infer_kernel/gpu.cuh>
#include <nvforest/detail/raft_proto/device_id.hpp>
#include <nvforest/detail/raft_proto/device_setter.hpp>
#include <nvforest/detail/raft_proto/device_type.hpp>
#include <nvforest/detail/raft_proto/gpu_support.hpp>
#include <nvforest/detail/specializations/device_initialization_macros.hpp>

#include <cuda_runtime_api.h>

#include <type_traits>

namespace nvforest::detail::device_initialization {

/* The implementation of the template used to initialize GPU device options
 *
 * On GPU-enabled builds, the GPU specialization of this template ensures that
 * the inference kernels have access to the maximum available dynamic shared
 * memory.
 */
template <typename forest_t, raft_proto::device_type D>
std::enable_if_t<std::conjunction_v<std::bool_constant<raft_proto::GPU_ENABLED>,
                                    std::bool_constant<D == raft_proto::device_type::gpu>>,
                 void>
initialize_device(raft_proto::device_id<D> device)
{
  auto device_context           = raft_proto::device_setter(device);
  auto max_shared_mem_per_block = get_max_shared_mem_per_block(device);
  // Run solely for side-effect of caching SM count
  get_sm_count(device);
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<false, 1, forest_t, std::nullptr_t, std::nullptr_t>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(infer_kernel<false, 2, forest_t>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(infer_kernel<false, 4, forest_t>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(infer_kernel<false, 8, forest_t>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(infer_kernel<false, 16, forest_t>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(infer_kernel<false, 32, forest_t>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<false, 1, forest_t, typename forest_t::io_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<false, 2, forest_t, typename forest_t::io_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<false, 4, forest_t, typename forest_t::io_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<false, 8, forest_t, typename forest_t::io_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<false, 16, forest_t, typename forest_t::io_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<false, 32, forest_t, typename forest_t::io_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(infer_kernel<true, 1, forest_t>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(infer_kernel<true, 2, forest_t>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(infer_kernel<true, 4, forest_t>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(infer_kernel<true, 8, forest_t>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(infer_kernel<true, 16, forest_t>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(infer_kernel<true, 32, forest_t>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<true, 1, forest_t, typename forest_t::io_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<true, 2, forest_t, typename forest_t::io_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<true, 4, forest_t, typename forest_t::io_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<true, 8, forest_t, typename forest_t::io_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<true, 16, forest_t, typename forest_t::io_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<true, 32, forest_t, typename forest_t::io_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(
    infer_kernel<true, 1, forest_t, std::nullptr_t, typename forest_t::node_type::index_type*>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(
    infer_kernel<true, 2, forest_t, std::nullptr_t, typename forest_t::node_type::index_type*>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(
    infer_kernel<true, 4, forest_t, std::nullptr_t, typename forest_t::node_type::index_type*>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(
    infer_kernel<true, 8, forest_t, std::nullptr_t, typename forest_t::node_type::index_type*>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(
    infer_kernel<true, 16, forest_t, std::nullptr_t, typename forest_t::node_type::index_type*>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(
    infer_kernel<true, 32, forest_t, std::nullptr_t, typename forest_t::node_type::index_type*>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(
    infer_kernel<true, 1, forest_t, typename forest_t::io_type*, std::nullptr_t>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(
    infer_kernel<true, 2, forest_t, typename forest_t::io_type*, std::nullptr_t>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(
    infer_kernel<true, 4, forest_t, typename forest_t::io_type*, std::nullptr_t>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(
    infer_kernel<true, 8, forest_t, typename forest_t::io_type*, std::nullptr_t>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(
    infer_kernel<true, 16, forest_t, typename forest_t::io_type*, std::nullptr_t>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    max_shared_mem_per_block));
  raft_proto::cuda_check(cudaFuncSetAttribute(
    infer_kernel<true, 32, forest_t, typename forest_t::io_type*, std::nullptr_t>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<true,
                                      1,
                                      forest_t,
                                      typename forest_t::io_type*,
                                      typename forest_t::node_type::index_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<true,
                                      2,
                                      forest_t,
                                      typename forest_t::io_type*,
                                      typename forest_t::node_type::index_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<true,
                                      4,
                                      forest_t,
                                      typename forest_t::io_type*,
                                      typename forest_t::node_type::index_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<true,
                                      8,
                                      forest_t,
                                      typename forest_t::io_type*,
                                      typename forest_t::node_type::index_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<true,
                                      16,
                                      forest_t,
                                      typename forest_t::io_type*,
                                      typename forest_t::node_type::index_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
  raft_proto::cuda_check(
    cudaFuncSetAttribute(infer_kernel<true,
                                      32,
                                      forest_t,
                                      typename forest_t::io_type*,
                                      typename forest_t::node_type::index_type*>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_per_block));
}

NVFOREST_INITIALIZE_DEVICE(extern template, 0)
NVFOREST_INITIALIZE_DEVICE(extern template, 1)
NVFOREST_INITIALIZE_DEVICE(extern template, 2)
NVFOREST_INITIALIZE_DEVICE(extern template, 3)
NVFOREST_INITIALIZE_DEVICE(extern template, 4)
NVFOREST_INITIALIZE_DEVICE(extern template, 5)
NVFOREST_INITIALIZE_DEVICE(extern template, 6)
NVFOREST_INITIALIZE_DEVICE(extern template, 7)
NVFOREST_INITIALIZE_DEVICE(extern template, 8)
NVFOREST_INITIALIZE_DEVICE(extern template, 9)
NVFOREST_INITIALIZE_DEVICE(extern template, 10)
NVFOREST_INITIALIZE_DEVICE(extern template, 11)

}  // namespace nvforest::detail::device_initialization
