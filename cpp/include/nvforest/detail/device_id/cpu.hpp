/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <nvforest/detail/device_id/base.hpp>
#include <nvforest/device_type.hpp>

namespace raft_proto::detail {
template <>
struct device_id<device_type::cpu> {
  using value_type = int;
  device_id() : id_{value_type{}} {};
  device_id(value_type dev_id) : id_{dev_id} {};

  auto value() const noexcept { return id_; }

 private:
  value_type id_;
};
}  // namespace raft_proto::detail
