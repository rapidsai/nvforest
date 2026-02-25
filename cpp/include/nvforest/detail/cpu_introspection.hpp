/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstddef>
#include <new>

namespace nvforest::detail {
#ifdef __cpplib_hardware_interference_size
using std::hardware_constructive_interference_size;
#else
auto constexpr static const hardware_constructive_interference_size = std::size_t{64};
#endif
}  // namespace nvforest::detail
