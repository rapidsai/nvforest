/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <nvforest/detail/device_initialization/gpu.cuh>
#include <nvforest/detail/infer/gpu.cuh>
#include <nvforest/detail/specializations/device_initialization_macros.hpp>
#include <nvforest/detail/specializations/infer_macros.hpp>
namespace nvforest::detail {
namespace inference {
NVFOREST_INFER_ALL(template, raft_proto::device_type::gpu, 5)
}  // namespace inference
namespace device_initialization {
NVFOREST_INITIALIZE_DEVICE(template, 5)
}  // namespace device_initialization
}  // namespace nvforest::detail
