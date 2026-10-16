/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nvforest/cuda_stream.hpp>
#include <nvforest/detail/device_setter.hpp>
#include <nvforest/device_type.hpp>
#include <nvforest/infer_kind.hpp>
#include <nvforest/tree_layout.hpp>
#include <nvforest/treelite_importer.hpp>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_pool.hpp>

#include <cuda/stream>
#include <thrust/device_vector.h>

#include <gtest/gtest.h>
#include <treelite/model_builder.h>
#include <treelite/tree.h>

#include <array>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace nvforest {

std::unique_ptr<treelite::Model> get_tree_model()
{
  auto model_builder = treelite::model_builder::GetModelBuilder(
    treelite::TypeInfo::kFloat32,
    treelite::TypeInfo::kFloat32,
    treelite::model_builder::Metadata{
      1,
      treelite::TaskType::kRegressor,
      false,
      1,
      {1},
      {1, 1},
    },
    treelite::model_builder::TreeAnnotation{2, {0, 0}, {0, 0}},
    treelite::model_builder::PostProcessorFunc{"identity"},
    std::vector<double>{0.0});
  model_builder->StartTree();
  model_builder->StartNode(0);
  model_builder->NumericalTest(0, 0.0, true, treelite::Operator::kLT, 1, 2);
  model_builder->EndNode();
  model_builder->StartNode(1);
  model_builder->LeafScalar(-1.0);
  model_builder->EndNode();
  model_builder->StartNode(2);
  model_builder->LeafScalar(1.0);
  model_builder->EndNode();
  model_builder->EndTree();

  model_builder->StartTree();
  model_builder->StartNode(0);
  model_builder->NumericalTest(0, 0.0, true, treelite::Operator::kLT, 1, 2);
  model_builder->EndNode();
  model_builder->StartNode(1);
  model_builder->LeafScalar(-1.0);
  model_builder->EndNode();
  model_builder->StartNode(2);
  model_builder->LeafScalar(1.0);
  model_builder->EndNode();
  model_builder->EndTree();

  return model_builder->CommitModel();
}

std::vector<float> repeat(std::vector<float> const& x, std::size_t n_repeat)
{
  std::vector<float> ret(x.size() * n_repeat);
  for (std::size_t i = 0; i < n_repeat; ++i) {
    for (std::size_t j = 0; j < x.size(); j++) {
      ret[i * x.size() + j] = x[j];
    }
  }
  return ret;
}

// Exercise every input/output memory combination on GPU 0 and, when available, GPU 1.
class GPUModelInferenceTest
  : public testing::TestWithParam<std::tuple<int, device_type, device_type>> {};

TEST_P(GPUModelInferenceTest, Predict)
{
  auto const [device_id, in_mem_type, out_mem_type] = GetParam();
  int device_count;
  detail::cuda_check(cudaGetDeviceCount(&device_count));
  if (device_id >= device_count) { GTEST_SKIP() << "Requires GPU " << device_id; }
  auto device_context = detail::device_setter{detail::device_id<device_type::gpu>{device_id}};

  constexpr std::size_t n_streams = 4;
  std::vector<cuda::stream> stream_pool;
  for (std::size_t i = 0; i < n_streams; ++i) {
    stream_pool.emplace_back(cuda::device_ref{device_id});
  }
  auto tree_model     = get_tree_model();
  auto nvforest_model = import_from_treelite_model(*tree_model,
                                                   tree_layout::breadth_first,
                                                   index_type{},
                                                   false,
                                                   device_type::gpu,
                                                   device_id,
                                                   stream_pool[0].get());
  // Model initialization must finish before inference can use the other streams.
  stream_pool[0].sync();

  // Larger than four minimum-size partitions, with an uneven final partition.
  constexpr std::size_t n_repeat = 16385;
  auto host_input                = repeat(std::vector<float>{-1.0f, 1.0f}, n_repeat);
  auto input_view                = buffer<float>{host_input.data(), host_input.size()};
  auto input = buffer<float>{input_view, in_mem_type, device_id, stream_pool[0].get()};
  stream_pool[0].sync();
  auto const cases = std::array{
    std::pair{infer_kind::default_kind, std::vector<float>{-2.0f, 2.0f}},
    std::pair{infer_kind::per_tree, std::vector<float>{-1.0f, -1.0f, 1.0f, 1.0f}},
    std::pair{infer_kind::leaf_id, std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f}},
  };
  std::size_t stream_calls = 0;
  auto stream_pool_next    = [&]() {
    return cuda_stream{stream_pool[stream_calls++ % n_streams].get()};
  };
  for (auto const& [infer_type, row_pair] : cases) {
    SCOPED_TRACE(static_cast<int>(infer_type));
    auto expected = repeat(row_pair, n_repeat);
    for (bool use_pool : {false, true}) {
      SCOPED_TRACE(use_pool);
      for (bool use_buffers : {false, true}) {
        SCOPED_TRACE(use_buffers);
        auto output = buffer<float>{expected.size(), out_mem_type, device_id, stream_pool[0].get()};
        stream_pool[0].sync();
        auto const previous_stream_calls = stream_calls;
        if (use_pool) {
          if (use_buffers) {
            nvforest_model.predict(stream_pool_next, n_streams, output, input, infer_type);
          } else {
            nvforest_model.predict(stream_pool_next,
                                   n_streams,
                                   output.data(),
                                   input.data(),
                                   host_input.size(),
                                   out_mem_type,
                                   in_mem_type,
                                   infer_type);
          }
          auto const cross_memory =
            in_mem_type != device_type::gpu || out_mem_type != device_type::gpu;
          EXPECT_EQ(stream_calls - previous_stream_calls, cross_memory ? n_streams : 1);
        } else if (use_buffers) {
          nvforest_model.predict(stream_pool[0].get(), output, input, infer_type);
        } else {
          nvforest_model.predict(stream_pool[0].get(),
                                 output.data(),
                                 input.data(),
                                 host_input.size(),
                                 out_mem_type,
                                 in_mem_type,
                                 infer_type);
        }
        for (auto& stream : stream_pool) {
          stream.sync();
        }
        auto actual = buffer<float>{output, device_type::cpu, 0, stream_pool[0].get()};
        stream_pool[0].sync();
        EXPECT_EQ(std::vector<float>(actual.data(), actual.data() + actual.size()), expected);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(MemoryLocations,
                         GPUModelInferenceTest,
                         testing::Combine(testing::Values(0, 1),
                                          testing::Values(device_type::cpu, device_type::gpu),
                                          testing::Values(device_type::cpu, device_type::gpu)));

TEST(Inference, EmptyStreamPool)
{
  auto tree_model  = get_tree_model();
  auto model       = import_from_treelite_model(*tree_model, tree_layout::breadth_first);
  auto input       = buffer<float>{1};
  auto output      = buffer<float>{1};
  auto next_stream = []() -> cuda_stream {
    ADD_FAILURE() << "An empty stream pool must be rejected before requesting a stream";
    return cuda_stream{};
  };
  EXPECT_THROW(model.predict(next_stream, 0, output, input), std::invalid_argument);
  EXPECT_THROW(
    model.predict(
      next_stream, 0, output.data(), input.data(), 1, device_type::cpu, device_type::cpu),
    std::invalid_argument);
}

// TODO(chyunsu3): Remove once we remove the RAFT dependency in 27.02.
TEST(Inference, GPUModelWithRAFTHandle)
{
  auto tree_model     = get_tree_model();
  auto nvforest_model = import_from_treelite_model(
    *tree_model, tree_layout::breadth_first, index_type{}, false, device_type::gpu);

  std::size_t n_repeat = 256;
  auto input = thrust::device_vector<float>(repeat(std::vector<float>{-1.0f, 1.0f}, n_repeat));
  auto const cases = std::array{
    std::pair{infer_kind::default_kind,
              thrust::device_vector<float>(repeat(std::vector<float>{-2.0f, 2.0f}, n_repeat))},
    std::pair{
      infer_kind::per_tree,
      thrust::device_vector<float>(repeat(std::vector<float>{-1.0f, -1.0f, 1.0f, 1.0f}, n_repeat))},
    std::pair{
      infer_kind::leaf_id,
      thrust::device_vector<float>(repeat(std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f}, n_repeat))},
  };

  std::size_t n_streams = 4;
  raft::handle_t handle{cuda::stream_ref{cudaStreamPerThread},
                        std::make_shared<rmm::cuda_stream_pool>(n_streams)};
  for (auto const& [infer_type, expected] : cases) {
    SCOPED_TRACE(static_cast<int>(infer_type));
    auto output = thrust::device_vector<float>(expected.size());
    nvforest_model.predict([&] { return cuda_stream{handle.get_next_usable_stream().get()}; },
                           handle.get_stream_pool_size(),
                           thrust::raw_pointer_cast(output.data()),
                           thrust::raw_pointer_cast(input.data()),
                           input.size(),
                           device_type::gpu,
                           device_type::gpu,
                           infer_type);
    handle.sync_stream_pool();
    EXPECT_EQ(output, expected);
  }
}

}  // namespace nvforest
