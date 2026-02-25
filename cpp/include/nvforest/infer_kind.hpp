/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
namespace nvforest {

/** Enum representing distinct prediction tasks */
enum class infer_kind : unsigned char { default_kind = 0, per_tree = 1, leaf_id = 2 };

}  // namespace nvforest
