/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <nvforest/detail/raft_proto/device_id.hpp>
#include <nvforest/detail/raft_proto/device_type.hpp>
#include <nvforest/detail/specializations/forest_macros.hpp>
/* Declare device initialization function for the types specified by the given
 * variant index */
#define NVFOREST_INITIALIZE_DEVICE(template_type, variant_index)                     \
  template_type void                                                                 \
    initialize_device<NVFOREST_FOREST(variant_index), raft_proto::device_type::gpu>( \
      raft_proto::device_id<raft_proto::device_type::gpu>);
