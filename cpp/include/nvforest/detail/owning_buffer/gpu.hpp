/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <nvforest/cuda_stream.hpp>
#include <nvforest/detail/device_id.hpp>
#include <nvforest/detail/owning_buffer/base.hpp>
#include <nvforest/device_type.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <type_traits>

namespace nvforest::detail {

struct owning_device_buffer_type_erased_impl;

struct owning_device_buffer_type_erased {
  owning_device_buffer_type_erased();
  owning_device_buffer_type_erased(device_id<device_type::gpu> device_id,
                                   std::size_t size,
                                   cuda::stream_ref stream);
  owning_device_buffer_type_erased(owning_device_buffer_type_erased&& other) noexcept;
  owning_device_buffer_type_erased& operator=(owning_device_buffer_type_erased&& other) noexcept;
  ~owning_device_buffer_type_erased();
  std::byte* get();

 private:
  std::unique_ptr<owning_device_buffer_type_erased_impl> impl_;
};

template <typename T>
struct owning_buffer<device_type::gpu, T> {
  // TODO(wphicks): Assess need for buffers of const T
  using value_type = std::remove_const_t<T>;
  owning_buffer()  = default;
  owning_buffer(device_id<device_type::gpu> device_id,
                std::size_t size,
                cuda_stream stream) noexcept(false)
    : data_{device_id, size * sizeof(value_type), cuda::stream_ref{stream}}
  {
  }

  auto* get() const { return reinterpret_cast<T*>(data_.get()); }

 private:
  mutable owning_device_buffer_type_erased data_;
};
}  // namespace nvforest::detail
