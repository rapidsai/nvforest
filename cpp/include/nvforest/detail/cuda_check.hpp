/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <nvforest/detail/cuda_check/base.hpp>
#ifdef NVFOREST_ENABLE_GPU
#include <nvforest/detail/cuda_check/gpu.hpp>
#endif
#include <nvforest/detail/gpu_support.hpp>
#include <nvforest/device_type.hpp>

namespace raft_proto {
template <typename error_t>
void cuda_check(error_t const& err) noexcept(!GPU_ENABLED)
{
  detail::cuda_check<device_type::gpu>(err);
}
}  // namespace raft_proto
