/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <nvforest/detail/cuda_check.hpp>
#include <nvforest/detail/device_id.hpp>
#include <nvforest/detail/device_setter/base.hpp>
#include <nvforest/device_type.hpp>

#include <cuda_runtime_api.h>

#include <cstdio>

#define NVFOREST_CUDA_TRY_NO_THROW(call)                           \
  do {                                                             \
    cudaError_t const status = call;                               \
    if (cudaSuccess != status) {                                   \
      printf("CUDA call='%s' at file=%s line=%d failed with %s\n", \
             #call,                                                \
             __FILE__,                                             \
             __LINE__,                                             \
             cudaGetErrorString(status));                          \
    }                                                              \
  } while (0)

namespace nvforest::detail {

/** Struct for setting current device within a code block */
template <>
struct device_setter<device_type::gpu> {
  device_setter(device_id<device_type::gpu> device) noexcept(false)
    : prev_device_{[]() {
        auto result = int{};
        cuda_check(cudaGetDevice(&result));
        return result;
      }()}
  {
    cuda_check(cudaSetDevice(device.value()));
  }

  ~device_setter() { NVFOREST_CUDA_TRY_NO_THROW(cudaSetDevice(prev_device_.value())); }

 private:
  device_id<device_type::gpu> prev_device_;
};

}  // namespace nvforest::detail
