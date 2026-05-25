/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <nvforest/cuda_stream.hpp>
#include <nvforest/detail/gpu_support.hpp>
#include <nvforest/device_type.hpp>

#include <stdint.h>

#include <algorithm>
#include <cstring>

namespace nvforest {

template <device_type dst_type, device_type src_type, typename T>
std::enable_if_t<std::conjunction_v<std::bool_constant<dst_type == device_type::cpu>,
                                    std::bool_constant<src_type == device_type::cpu>>,
                 void>
copy(T* dst, T const* src, uint32_t size, cuda_stream stream)
{
  std::copy(src, src + size, dst);
}

template <device_type dst_type, device_type src_type, typename T>
std::enable_if_t<
  std::conjunction_v<std::disjunction<std::bool_constant<dst_type != device_type::cpu>,
                                      std::bool_constant<src_type != device_type::cpu>>,
                     std::bool_constant<!detail::GPU_ENABLED>>,
  void>
copy(T* dst, T const* src, uint32_t size, cuda_stream stream)
{
  throw detail::gpu_unsupported("Copying from or to device in non-GPU build");
}

}  // namespace nvforest
