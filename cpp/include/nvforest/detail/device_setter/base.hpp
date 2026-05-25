/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <nvforest/detail/device_id.hpp>
#include <nvforest/device_type.hpp>

namespace nvforest::detail {

/** Struct for setting current device within a code block */
template <device_type D>
struct device_setter {
  device_setter(device_id<D> device) {}
};

}  // namespace nvforest::detail
