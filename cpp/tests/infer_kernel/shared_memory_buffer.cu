/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nvforest/detail/infer_kernel/shared_memory_buffer.cuh>
#include <nvforest/detail/raft_proto/buffer.hpp>
#include <nvforest/detail/raft_proto/cuda_check.hpp>
#include <nvforest/detail/raft_proto/cuda_stream.hpp>
#include <nvforest/detail/raft_proto/device_type.hpp>
#include <nvforest/detail/utils.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <vector>

namespace nvforest {
namespace {

struct allocation_result {
  bool copied_to_shared;
  int copy_value;
  index_type remaining_after_copy;
  bool subsequent_fill_succeeded;
  index_type remaining_after_fill;
  int fill_value;
};

NVFOREST_KERNEL void check_2d_allocation(int* source,
                                         allocation_result* result,
                                         index_type row_count,
                                         index_type col_count,
                                         index_type row_pad,
                                         index_type shared_mem_size)
{
  extern __shared__ std::byte shared_mem[];
  auto buffer               = shared_memory_buffer{shared_mem, shared_mem_size};
  auto* copy_result         = buffer.copy(source, row_count, col_count, row_pad);
  auto remaining_after_copy = buffer.remaining();
  auto* fill_result         = buffer.fill<int>(1, 7);
  buffer.sync();

  if (threadIdx.x == 0) {
    result->copied_to_shared          = copy_result != source;
    result->copy_value                = copy_result[0];
    result->remaining_after_copy      = remaining_after_copy;
    result->subsequent_fill_succeeded = fill_result != nullptr;
    result->remaining_after_fill      = buffer.remaining();
    result->fill_value                = fill_result == nullptr ? 0 : fill_result[0];
  }
}

auto run_2d_allocation(index_type row_count,
                       index_type col_count,
                       index_type row_pad,
                       index_type shared_mem_size,
                       std::size_t source_size)
{
  auto source_data = std::vector<int>(source_size, 42);
  auto source =
    raft_proto::buffer<int>{source_data.begin(), source_data.end(), raft_proto::device_type::gpu};
  auto result = raft_proto::buffer<allocation_result>{1, raft_proto::device_type::gpu};

  check_2d_allocation<<<1, 32, shared_mem_size>>>(
    source.data(), result.data(), row_count, col_count, row_pad, shared_mem_size);
  raft_proto::cuda_check(cudaGetLastError());

  auto result_data = allocation_result{};
  auto host_result =
    raft_proto::buffer<allocation_result>{&result_data, 1, raft_proto::device_type::cpu};
  raft_proto::copy<true>(host_result, result);
  raft_proto::cuda_check(cudaStreamSynchronize(raft_proto::cuda_stream{}));
  return result_data;
}

TEST(SharedMemoryBuffer, CopiesAllocationThatFitsExactly)
{
  auto const result = run_2d_allocation(2, 2, 0, 4 * sizeof(int), 4);

  EXPECT_TRUE(result.copied_to_shared);
  EXPECT_EQ(result.copy_value, 42);
  EXPECT_EQ(result.remaining_after_copy, 0);
  EXPECT_FALSE(result.subsequent_fill_succeeded);
  EXPECT_EQ(result.remaining_after_fill, 0);
}

TEST(SharedMemoryBuffer, PreservesStateIfAllocationIsOneByteTooLarge)
{
  auto constexpr shared_mem_size = 4 * sizeof(int) - 1;
  auto const result              = run_2d_allocation(2, 2, 0, shared_mem_size, 4);

  EXPECT_FALSE(result.copied_to_shared);
  EXPECT_EQ(result.remaining_after_copy, shared_mem_size);
  EXPECT_TRUE(result.subsequent_fill_succeeded);
  EXPECT_EQ(result.remaining_after_fill, shared_mem_size - sizeof(int));
  EXPECT_EQ(result.fill_value, 7);
}

TEST(SharedMemoryBuffer, RejectsWideInputWithoutOverflowingSizeCalculation)
{
  auto constexpr shared_mem_size = 16 * sizeof(int);
  auto const result              = run_2d_allocation(1, 100'000, 0, shared_mem_size, 100'000);

  EXPECT_FALSE(result.copied_to_shared);
  EXPECT_EQ(result.remaining_after_copy, shared_mem_size);
  EXPECT_TRUE(result.subsequent_fill_succeeded);
  EXPECT_EQ(result.remaining_after_fill, shared_mem_size - sizeof(int));
}

TEST(SharedMemoryBuffer, RejectsDimensionsWhoseProductWouldOverflow)
{
  auto constexpr max_index       = std::numeric_limits<index_type>::max();
  auto constexpr shared_mem_size = 16 * sizeof(int);
  auto const result = run_2d_allocation(max_index, max_index, max_index, shared_mem_size, 1);

  EXPECT_FALSE(result.copied_to_shared);
  EXPECT_EQ(result.remaining_after_copy, shared_mem_size);
  EXPECT_TRUE(result.subsequent_fill_succeeded);
  EXPECT_EQ(result.remaining_after_fill, shared_mem_size - sizeof(int));
}

NVFOREST_KERNEL void check_failed_copy_sync(int* source, unsigned int* completed_blocks)
{
  auto buffer = shared_memory_buffer{};
  buffer.copy(source, index_type{1});
  buffer.sync();
  if (threadIdx.x == 0) { atomicAdd(completed_blocks, 1); }
}

TEST(SharedMemoryBuffer, RepeatedlySkipsSyncIfNoCopyOccurs)
{
  auto source_data = std::vector<int>{42};
  auto source =
    raft_proto::buffer<int>{source_data.begin(), source_data.end(), raft_proto::device_type::gpu};
  auto completed_data = std::vector<unsigned int>{0};
  auto completed      = raft_proto::buffer<unsigned int>{
    completed_data.begin(), completed_data.end(), raft_proto::device_type::gpu};

  auto constexpr launch_count = 1'000;
  auto constexpr block_count  = 32;
  for (auto i = 0; i < launch_count; ++i) {
    check_failed_copy_sync<<<block_count, 128>>>(source.data(), completed.data());
  }
  raft_proto::cuda_check(cudaGetLastError());

  auto host_completed = raft_proto::buffer<unsigned int>{
    completed_data.data(), completed_data.size(), raft_proto::device_type::cpu};
  raft_proto::copy<true>(host_completed, completed);
  raft_proto::cuda_check(cudaStreamSynchronize(raft_proto::cuda_stream{}));
  EXPECT_EQ(completed_data[0], launch_count * block_count);
}

}  // namespace
}  // namespace nvforest
