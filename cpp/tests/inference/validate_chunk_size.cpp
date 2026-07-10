/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nvforest/detail/validate_chunk_size.hpp>
#include <nvforest/exceptions.hpp>

#include <gtest/gtest.h>

#include <optional>

namespace nvforest::detail {
namespace {

TEST(ValidateChunkSize, AcceptsValidGpuValues)
{
  for (auto chunk_size : {1, 2, 4, 8, 16, 32}) {
    EXPECT_NO_THROW(validate_chunk_size(chunk_size, raft_proto::device_type::gpu));
  }
  EXPECT_NO_THROW(validate_chunk_size(std::nullopt, raft_proto::device_type::gpu));
}

TEST(ValidateChunkSize, RejectsInvalidGpuValues)
{
  for (auto chunk_size : {0, 3, 6, 17, 33}) {
    EXPECT_THROW(validate_chunk_size(chunk_size, raft_proto::device_type::gpu), runtime_error);
  }
}

TEST(ValidateChunkSize, AcceptsAnyPositiveCpuValue)
{
  for (auto chunk_size : {1, 2, 3, 6, 17, 33, 512}) {
    EXPECT_NO_THROW(validate_chunk_size(chunk_size, raft_proto::device_type::cpu));
  }
  EXPECT_NO_THROW(validate_chunk_size(std::nullopt, raft_proto::device_type::cpu));
}

TEST(ValidateChunkSize, RejectsZeroForCpu)
{
  EXPECT_THROW(validate_chunk_size(0, raft_proto::device_type::cpu), runtime_error);
}

}  // namespace
}  // namespace nvforest::detail
