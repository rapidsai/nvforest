/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvforest/detail/raft_proto/cuda_stream.hpp>

#include <raft/core/device_resources.hpp>

namespace raft_proto {

// TODO(nvforest#121): This is a compatibility bridge while nvForest accepts
// raft::device_resources publicly but still uses raft_proto::cuda_stream
// internally. Remove this once internal resource/stream handling is migrated to
// RAFT/RMM abstractions consistently.
inline cuda_stream get_next_usable_stream(raft::device_resources const& resource)
{
#ifdef NVFOREST_ENABLE_GPU
  return resource.get_next_usable_stream().value();
#else
  (void)resource;
  return cuda_stream{};
#endif
}

}  // namespace raft_proto
