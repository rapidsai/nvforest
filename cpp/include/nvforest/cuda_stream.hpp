/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#ifdef NVFOREST_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif

namespace nvforest {
#ifdef NVFOREST_ENABLE_GPU
using cuda_stream = cudaStream_t;
#else
using cuda_stream = int;
#endif
inline void synchronize(cuda_stream stream)
{
#ifdef NVFOREST_ENABLE_GPU
  cudaStreamSynchronize(stream);
#endif
}
}  // namespace nvforest

namespace nvforest::detail {
using nvforest::cuda_stream;
using nvforest::synchronize;
}  // namespace nvforest::detail
