/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <nvforest/detail/device_id.hpp>
#include <nvforest/detail/device_setter/gpu.hpp>
#include <nvforest/detail/owning_buffer/gpu.hpp>

#include <cuda/buffer>
#include <cuda/memory_resource>
#include <cuda/stream>

#include <cstddef>
#include <memory>

namespace nvforest::detail {

struct owning_device_buffer_type_erased_impl {
  owning_device_buffer_type_erased_impl(int device_id, std::size_t size, cudaStream_t stream)
    : buffer_{[&stream, device_id, size]() {
        auto device = cuda::device_ref{device_id};
        auto mr     = cuda::device_default_memory_pool(device);
        return cuda::make_buffer<std::byte>(cuda::stream_ref{stream}, mr, size, cuda::no_init);
      }()}
  {
  }

  std::byte* get() { return buffer_.data(); }

  cuda::device_buffer<std::byte> buffer_;
};

owning_device_buffer_type_erased::owning_device_buffer_type_erased() : impl_{nullptr} {}

owning_device_buffer_type_erased::owning_device_buffer_type_erased(
  device_id<device_type::gpu> device_id, std::size_t size, cudaStream_t stream)
{
  auto device_context = device_setter{device_id};
  impl_ = std::make_unique<owning_device_buffer_type_erased_impl>(device_id.value(), size, stream);
}

owning_device_buffer_type_erased::owning_device_buffer_type_erased(
  owning_device_buffer_type_erased&& other) noexcept = default;
owning_device_buffer_type_erased& owning_device_buffer_type_erased::operator=(
  owning_device_buffer_type_erased&& other) noexcept                  = default;
owning_device_buffer_type_erased::~owning_device_buffer_type_erased() = default;

std::byte* owning_device_buffer_type_erased::get() { return impl_->get(); }

}  // namespace nvforest::detail
