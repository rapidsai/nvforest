/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvforest/detail/device_initialization/cpu.hpp>

#include <variant>
#ifdef NVFOREST_ENABLE_GPU
#include <nvforest/detail/device_initialization/gpu.hpp>
#endif

namespace nvforest::detail {
/* Set any required device options for optimizing nvForest compute */
template <typename forest_t, device_type D>
void initialize_device()
{
  if constexpr (D == device_type::gpu) {
    device_initialization::initialize_device<forest_t, device_type::gpu>();
  }
}

}  // namespace nvforest::detail
