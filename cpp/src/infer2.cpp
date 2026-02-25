/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <nvforest/detail/infer/cpu.hpp>
#include <nvforest/detail/specializations/infer_macros.hpp>
namespace nvforest::detail::inference {
NVFOREST_INFER_ALL(template, raft_proto::device_type::cpu, 2)
}  // namespace nvforest::detail::inference
