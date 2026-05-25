/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <nvforest/device_type.hpp>

namespace nvforest::detail {

template <device_type D, typename error_t>
void cuda_check(error_t const& err)
{
}

}  // namespace nvforest::detail
