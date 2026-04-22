/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/device_resources.hpp>

namespace nvforest {

class device_resources {
 public:
  device_resources() : res_{} {}

  auto get_next_usable_stream() const { return res_.get_next_usable_stream(); }
  auto get_stream_pool_size() const { return res_.get_stream_pool_size(); }
  void synchronize() const
  {
    res_.sync_stream_pool();
    res_.sync_stream();
  }

 private:
  raft::device_resources res_;
};

}  // namespace nvforest
