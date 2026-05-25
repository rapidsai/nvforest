/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvforest/detail/device_id/base.hpp>
#include <nvforest/detail/device_id/cpu.hpp>
#ifdef NVFOREST_ENABLE_GPU
#include <nvforest/detail/device_id/gpu.hpp>
#endif
#include <nvforest/device_type.hpp>

#include <variant>

namespace nvforest::detail {
using device_id_variant = std::variant<device_id<device_type::cpu>, device_id<device_type::gpu>>;
}  // namespace nvforest::detail
