/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <nvforest/detail/owning_buffer/cpu.hpp>
#include <nvforest/device_type.hpp>
#ifdef NVFOREST_ENABLE_GPU
#include <nvforest/detail/owning_buffer/gpu.hpp>
#endif
namespace raft_proto {
template <device_type D, typename T>
using owning_buffer = detail::owning_buffer<D, T>;
}  // namespace raft_proto
