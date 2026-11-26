/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <nvforest/detail/cuda_check.hpp>
#include <nvforest/detail/owning_buffer/gpu.hpp>

#include <cuda/buffer>
#include <cuda/memory_resource>
#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>

namespace nvforest::detail {

struct owning_device_buffer_type_erased_impl {
  owning_device_buffer_type_erased_impl(std::size_t size, cuda::stream_ref stream)
    : buffer_{[&stream, size]() {
        int device_id;
        auto const handle = stream.get();
        if (handle == nullptr || handle == cudaStreamLegacy || handle == cudaStreamPerThread) {
          // Make a context current if needed so the allocator can resolve the default stream.
          cuda_check(cudaFree(nullptr));
          cuda_check(cudaGetDevice(&device_id));
        } else {
          device_id = stream.device().get();
        }
        auto device = cuda::device_ref{device_id};
        auto mr     = cuda::device_default_memory_pool(device);
        return cuda::make_buffer<std::byte>(stream, mr, size, cuda::no_init);
      }()}
  {
  }

  std::byte* get() { return buffer_.data(); }

  cuda::device_buffer<std::byte> buffer_;
};

owning_device_buffer_type_erased::owning_device_buffer_type_erased() : impl_{nullptr} {}

owning_device_buffer_type_erased::owning_device_buffer_type_erased(std::size_t size,
                                                                   cuda::stream_ref stream)
{
  impl_ = std::make_unique<owning_device_buffer_type_erased_impl>(size, stream);
}

owning_device_buffer_type_erased::owning_device_buffer_type_erased(
  owning_device_buffer_type_erased&& other) noexcept = default;
owning_device_buffer_type_erased& owning_device_buffer_type_erased::operator=(
  owning_device_buffer_type_erased&& other) noexcept                  = default;
owning_device_buffer_type_erased::~owning_device_buffer_type_erased() = default;

std::byte* owning_device_buffer_type_erased::get() { return impl_ ? impl_->get() : nullptr; }

}  // namespace nvforest::detail
