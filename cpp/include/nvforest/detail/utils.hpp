/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef __CUDACC__
#define NVFOREST_KERNEL __global__ static
#else
#define NVFOREST_KERNEL static
#endif
