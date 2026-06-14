/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <nvforest/detail/device_setter/base.hpp>
#ifdef NVFOREST_ENABLE_GPU
#include <nvforest/detail/device_setter/gpu.hpp>
#endif
#include <nvforest/device_type.hpp>
