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
