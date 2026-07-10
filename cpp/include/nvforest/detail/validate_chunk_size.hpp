/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvforest/detail/index_type.hpp>
#include <nvforest/detail/raft_proto/device_type.hpp>
#include <nvforest/exceptions.hpp>

#include <optional>

namespace nvforest::detail {

inline void validate_chunk_size(std::optional<index_type> specified_chunk_size,
                                raft_proto::device_type device_type)
{
  if (!specified_chunk_size.has_value()) { return; }

  auto const chunk_size = specified_chunk_size.value();
  if (chunk_size == 0) { throw runtime_error("Chunk size must be greater than zero"); }

  if (device_type == raft_proto::device_type::gpu &&
      (chunk_size > 32 || (chunk_size & (chunk_size - 1)) != 0)) {
    throw runtime_error("GPU chunk size must be a power of two between 1 and 32");
  }
}

}  // namespace nvforest::detail
