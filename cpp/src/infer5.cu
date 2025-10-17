/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cuforest/detail/device_initialization/gpu.cuh>
#include <cuforest/detail/infer/gpu.cuh>
#include <cuforest/detail/specializations/device_initialization_macros.hpp>
#include <cuforest/detail/specializations/infer_macros.hpp>
namespace cuforest::detail {
namespace inference {
CUFOREST_INFER_ALL(template, raft_proto::device_type::gpu, 5)
}  // namespace inference
namespace device_initialization {
CUFOREST_INITIALIZE_DEVICE(template, 5)
}  // namespace device_initialization
}  // namespace cuforest::detail
