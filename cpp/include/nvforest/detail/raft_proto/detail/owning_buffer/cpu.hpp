/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <nvforest/detail/raft_proto/detail/owning_buffer/base.hpp>
#include <nvforest/detail/raft_proto/device_id.hpp>
#include <nvforest/detail/raft_proto/device_type.hpp>

#include <memory>
#include <type_traits>

namespace raft_proto::detail {
template <typename T>
struct owning_buffer<device_type::cpu, T> {
  // TODO(wphicks): Assess need for buffers of const T
  using value_type = std::remove_const_t<T>;

  owning_buffer() : data_{std::unique_ptr<T[]>{nullptr}} {}

  owning_buffer(std::size_t size) : data_{std::make_unique<T[]>(size)} {}

  auto* get() const { return data_.get(); }

 private:
  // TODO(wphicks): Back this with RMM-allocated host memory
  std::unique_ptr<T[]> data_;
};
}  // namespace raft_proto::detail
