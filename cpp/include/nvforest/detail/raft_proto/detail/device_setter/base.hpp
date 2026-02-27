/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <nvforest/detail/raft_proto/device_id.hpp>
#include <nvforest/detail/raft_proto/device_type.hpp>

namespace raft_proto::detail {

/** Struct for setting current device within a code block */
template <device_type D>
struct device_setter {
  device_setter(device_id<D> device) {}
};

}  // namespace raft_proto::detail
