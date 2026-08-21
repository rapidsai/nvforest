/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <nvforest/detail/cuda_check.hpp>
#include <nvforest/detail/device_id/base.hpp>
#include <nvforest/device_type.hpp>

namespace nvforest::detail {
template <>
struct device_id<device_type::gpu> {
  device_id() noexcept(false)
    : id_{[]() {
        auto raw_id = int{};
        cuda_check(cudaGetDevice(&raw_id));
        return raw_id;
      }()} {};
  device_id(int dev_id) noexcept : id_{dev_id} {};

  auto value() const noexcept { return id_; }

 private:
  int id_;
};
}  // namespace nvforest::detail
