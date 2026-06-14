/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvforest/detail/device_id.hpp>
#include <nvforest/detail/gpu_support.hpp>
#include <nvforest/device_type.hpp>

#include <type_traits>

namespace nvforest::detail::device_initialization {

/* Specialization for any initialization required for CPUs
 *
 * This specialization will also be used for non-GPU-enabled builds
 * (as a GPU no-op).
 */
template <typename forest_t, device_type D>
std::enable_if_t<
  std::disjunction_v<std::bool_constant<!GPU_ENABLED>, std::bool_constant<D == device_type::cpu>>,
  void>
initialize_device(device_id<D> device)
{
}

}  // namespace nvforest::detail::device_initialization
