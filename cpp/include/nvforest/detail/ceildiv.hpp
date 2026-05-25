/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <nvforest/detail/gpu_support.hpp>

#include <type_traits>

namespace nvforest::detail {
template <typename T, typename U>
HOST DEVICE auto constexpr ceildiv(T dividend, U divisor)
{
  static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "Arguments must be integers");
  return dividend / divisor + (dividend % divisor != 0);
}
}  // namespace nvforest::detail
