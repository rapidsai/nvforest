/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <nvforest/cuda_stream.hpp>
#include <nvforest/device_type.hpp>

#include <type_traits>

namespace nvforest::detail {

template <device_type D, typename T>
struct owning_buffer {
  owning_buffer() = default;
  owning_buffer(std::size_t size, cuda_stream stream) {}
  auto* get() const { return static_cast<T*>(nullptr); }
};

}  // namespace nvforest::detail
