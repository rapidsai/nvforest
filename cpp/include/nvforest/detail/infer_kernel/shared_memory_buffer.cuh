/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <nvforest/detail/index_type.hpp>

#include <stddef.h>

#include <cstddef>
#include <type_traits>

namespace nvforest {

/* A struct used to simplify complex access to a buffer of shared memory
 *
 * @param buffer A pointer to the shared memory allocation
 * @param size The size in bytes of the shared memory allocation
 */
struct shared_memory_buffer {
  __device__ shared_memory_buffer(std::byte* buffer = nullptr, index_type size = index_type{})
    : data{buffer},
      total_size{size},
      remaining_data{buffer},
      remaining_size{size},
      requires_sync{false}
  {
  }

  /* If possible, copy the given number of rows with the given number of columns from source
   * to the end of this buffer, padding each row by the given number of
   * elements (usually to reduce memory bank conflicts). If there is not enough
   * room, no copy is performed. Return a pointer to the desired data, whether
   * that is in the original location or copied to shared memory. */
  template <typename T>
  __device__ auto* copy(T* source,
                        index_type row_count,
                        index_type col_count,
                        index_type row_pad = index_type{})
  {
    auto const row_width          = std::size_t{col_count} + std::size_t{row_pad};
    auto const available_elements = std::size_t{remaining_size} / sizeof(T);
    if (row_count != 0 && row_width > available_elements / row_count) { return source; }

    auto* dest              = reinterpret_cast<std::remove_const_t<T>*>(remaining_data);
    auto const source_count = std::size_t{row_count} * std::size_t{col_count};
    auto const dest_count   = std::size_t{row_count} * row_width;
    for (auto i = std::size_t{threadIdx.x}; i < source_count; i += blockDim.x) {
      dest[i + row_pad * (i / col_count)] = source[i];
    }

    auto const offset = dest_count * sizeof(T);
    remaining_data += offset;
    remaining_size -= index_type(offset);
    requires_sync = requires_sync || source_count != 0;

    return static_cast<T*>(dest);
  }

  /* If possible, copy the given number of elements from source to the end of this buffer
   * If there is not enough room, no copy is performed. Return a pointer to the
   * desired data, whether that is in the original location or copied to shared
   * memory. */
  template <typename T>
  __device__ auto* copy(T* source, index_type element_count)
  {
    if (std::size_t{element_count} > std::size_t{remaining_size} / sizeof(T)) { return source; }

    auto* dest = reinterpret_cast<std::remove_const_t<T>*>(remaining_data);
    for (auto i = std::size_t{threadIdx.x}; i < element_count; i += blockDim.x) {
      dest[i] = source[i];
    }

    auto const offset = std::size_t{element_count} * sizeof(T);
    remaining_data += offset;
    remaining_size -= index_type(offset);
    requires_sync = requires_sync || element_count != 0;

    return static_cast<T*>(dest);
  }

  /* If possible, fill the next element_count elements with given value. If
   * there is not enough room, the fill is not performed. Return a pointer to
   * the start of the desired data if the fill was possible, or else the
   * provided fallback buffer (nullptr by default). */
  template <typename T>
  __device__ auto* fill(index_type element_count, T value = T{}, T* fallback_buffer = nullptr)
  {
    if (std::size_t{element_count} > std::size_t{remaining_size} / sizeof(T)) {
      return fallback_buffer;
    }

    auto* dest = reinterpret_cast<std::remove_const_t<T>*>(remaining_data);
    for (auto i = std::size_t{threadIdx.x}; i < element_count; i += blockDim.x) {
      dest[i] = value;
    }

    auto const offset = std::size_t{element_count} * sizeof(T);
    remaining_data += offset;
    remaining_size -= index_type(offset);
    requires_sync = requires_sync || element_count != 0;

    return static_cast<T*>(dest);
  }

  /* Clear all stored data and return a pointer to the beginning of available
   * shared memory */
  __device__ auto* clear()
  {
    remaining_size = total_size;
    remaining_data = data;
    requires_sync  = false;
    return remaining_data;
  }

  /* Pad stored data to ensure correct alignment for given type */
  template <typename T>
  __device__ void align()
  {
    auto pad_required = (total_size - remaining_size) % index_type(sizeof(T));
    remaining_data += pad_required;
    remaining_size -= pad_required;
  }

  /* If necessary, sync threads. Note that this can cause a deadlock if not all
   * threads call this method. */
  __device__ void sync()
  {
    if (requires_sync) { __syncthreads(); }
    requires_sync = false;
  }

  /* Return the remaining size in bytes left in this buffer */
  __device__ auto remaining() { return remaining_size; }

 private:
  std::byte* data;
  index_type total_size;
  std::byte* remaining_data;
  index_type remaining_size;
  bool requires_sync;
};

}  // namespace nvforest
