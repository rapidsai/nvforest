/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <nvforest/detail/forest.hpp>
#include <nvforest/detail/index_type.hpp>
#include <nvforest/detail/postprocessor.hpp>
#include <nvforest/detail/raft_proto/cuda_stream.hpp>
#include <nvforest/detail/raft_proto/device_id.hpp>
#include <nvforest/detail/raft_proto/device_type.hpp>
#include <nvforest/infer_kind.hpp>

#include <cstddef>
#include <optional>

namespace nvforest::detail::inference {

/* The CUDA-free header declaration of the GPU infer template */
template <raft_proto::device_type D,
          bool has_categorical_nodes,
          typename forest_t,
          typename vector_output_t    = std::nullptr_t,
          typename categorical_data_t = std::nullptr_t>
std::enable_if_t<D == raft_proto::device_type::gpu, void> infer(
  forest_t const& forest,
  postprocessor<typename forest_t::io_type> const& postproc,
  typename forest_t::io_type* output,
  typename forest_t::io_type* input,
  index_type row_count,
  index_type col_count,
  index_type class_count,
  vector_output_t vector_output                  = nullptr,
  categorical_data_t categorical_data            = nullptr,
  infer_kind infer_type                          = infer_kind::default_kind,
  std::optional<index_type> specified_chunk_size = std::nullopt,
  raft_proto::device_id<D> device                = raft_proto::device_id<D>{},
  raft_proto::cuda_stream stream                 = raft_proto::cuda_stream{});

}  // namespace nvforest::detail::inference
