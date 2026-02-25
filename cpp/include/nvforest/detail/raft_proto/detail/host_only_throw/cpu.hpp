/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <nvforest/detail/raft_proto/detail/host_only_throw/base.hpp>
#include <nvforest/detail/raft_proto/gpu_support.hpp>

namespace raft_proto::detail {
template <typename T>
struct host_only_throw<T, true> {
  template <typename... Args>
  host_only_throw(Args&&... args) noexcept(false)
  {
    throw T{std::forward<Args>(args)...};
  }
};
}  // namespace raft_proto::detail
