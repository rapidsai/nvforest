/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <nvforest/detail/raft_proto/detail/cuda_check/base.hpp>
#include <nvforest/detail/raft_proto/device_type.hpp>
#include <nvforest/detail/raft_proto/exceptions.hpp>

#include <cuda_runtime_api.h>
namespace raft_proto::detail {

template <>
inline void cuda_check<device_type::gpu, cudaError_t>(cudaError_t const& err) noexcept(false)
{
  if (err != cudaSuccess) {
    cudaGetLastError();
    throw bad_cuda_call(cudaGetErrorString(err));
  }
}

}  // namespace raft_proto::detail
