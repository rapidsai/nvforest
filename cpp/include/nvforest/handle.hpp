/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <nvforest/detail/raft_proto/handle.hpp>

#ifdef NVFOREST_ENABLE_GPU
#include <raft/core/handle.hpp>

#include <memory>
#endif

namespace nvforest {

#ifdef NVFOREST_ENABLE_GPU
/**
 * A thin wrapper around raft_proto::handle_t that owns the underlying raft::handle_t.
 *
 * Default construction automatically creates both a raft::handle_t and the
 * raft_proto::handle_t that references it, so callers do not need to manage
 * RAFT handles directly.
 */
struct handle_t {
  /** Default constructor: creates and owns a raft::handle_t and wraps it */
  handle_t()
    : owned_raft_handle_{std::make_unique<raft::handle_t>()},
      raft_proto_handle_{*owned_raft_handle_}
  {
  }

  /** Wrap an externally-owned raft::handle_t without taking ownership */
  handle_t(raft::handle_t const& raft_handle) : raft_proto_handle_{raft_handle} {}

  auto get_next_usable_stream() const { return raft_proto_handle_.get_next_usable_stream(); }
  auto get_stream_pool_size() const { return raft_proto_handle_.get_stream_pool_size(); }
  auto get_usable_stream_count() const { return raft_proto_handle_.get_usable_stream_count(); }
  void synchronize() const { raft_proto_handle_.synchronize(); }

 private:
  // Null when wrapping an external raft::handle_t
  std::unique_ptr<raft::handle_t> owned_raft_handle_;
  raft_proto::handle_t raft_proto_handle_;
};
#else
/**
 * CPU-only handle: thin wrapper around the no-op raft_proto::handle_t.
 */
struct handle_t {
  auto get_next_usable_stream() const { return raft_proto_handle_.get_next_usable_stream(); }
  auto get_stream_pool_size() const { return raft_proto_handle_.get_stream_pool_size(); }
  auto get_usable_stream_count() const { return raft_proto_handle_.get_usable_stream_count(); }
  void synchronize() const { raft_proto_handle_.synchronize(); }

 private:
  raft_proto::handle_t raft_proto_handle_;
};
#endif

}  // namespace nvforest
