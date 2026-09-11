/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nvforest/detail/cuda_check.hpp>
#include <nvforest/detail/device_id.hpp>
#include <nvforest/detail/device_setter.hpp>
#include <nvforest/treelite_importer.hpp>

#include <cuda/devices>
#include <thrust/device_vector.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <treelite/model_builder.h>
#include <treelite/tree.h>

#include <iostream>
#include <stdexcept>

namespace nvforest {

TEST(InvalidStream, wrong_device)
{
  auto const num_devices = cuda::devices.size();
  if (num_devices < 2) { GTEST_SKIP() << "Test requires at least 2 GPU devices"; }

  // Make a stump tree
  auto model_builder =
    treelite::model_builder::GetModelBuilder(treelite::TypeInfo::kFloat32,
                                             treelite::TypeInfo::kFloat32,
                                             treelite::model_builder::Metadata{
                                               1,
                                               treelite::TaskType::kRegressor,
                                               false,
                                               1,
                                               {1},
                                               {1, 1},
                                             },
                                             treelite::model_builder::TreeAnnotation{1, {0}, {0}},
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

  cuda_stream stream;
  {
    auto device_context = detail::device_setter{detail::device_id<device_type::gpu>{1}};
    detail::cuda_check(cudaStreamCreate(&stream));
  }

  // Loading a model on device 0 and inferencing on device 1 is an error.
  auto tl_model       = model_builder->CommitModel();
  auto nvforest_model = import_from_treelite_model(
    *tl_model, tree_layout::breadth_first, index_type{}, false, device_type::gpu, 0, stream);
  auto input  = thrust::device_vector<float>{1.0f};
  auto output = thrust::device_vector<float>(1);
  EXPECT_THAT(
    [&]() {
      nvforest_model.predict(stream,
                             thrust::raw_pointer_cast(output.data()),
                             thrust::raw_pointer_cast(input.data()),
                             1,
                             device_type::gpu,
                             device_type::gpu,
                             infer_kind::default_kind);
    },
    testing::ThrowsMessage<std::runtime_error>(
      testing::HasSubstr("Stream on the wrong device. Expected: 0, Actual: 1")));
}

}  // namespace nvforest
