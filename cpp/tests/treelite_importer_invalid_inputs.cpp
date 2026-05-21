/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nvforest/decision_forest.hpp>
#include <nvforest/exceptions.hpp>
#include <nvforest/tree_layout.hpp>
#include <nvforest/treelite_importer.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <treelite/enum/task_type.h>
#include <treelite/enum/typeinfo.h>
#include <treelite/model_builder.h>

#include <cstdint>
#include <limits>
#include <optional>

namespace nvforest {

namespace {

auto make_categorical_model(std::uint32_t category)
{
  auto metadata = treelite::model_builder::Metadata{
    1,
    treelite::TaskType::kRegressor,
    false,
    1,
    {1},
    {1, 1},
  };
  auto tree_annotation = treelite::model_builder::TreeAnnotation{1, {0}, {0}};
  auto model_builder =
    treelite::model_builder::GetModelBuilder(treelite::TypeInfo::kFloat32,
                                             treelite::TypeInfo::kFloat32,
                                             metadata,
                                             tree_annotation,
                                             treelite::model_builder::PostProcessorFunc{"identity"},
                                             {0.0});

  model_builder->StartTree();
  model_builder->StartNode(0);
  model_builder->CategoricalTest(0, false, {category}, false, 1, 2);
  model_builder->EndNode();

  model_builder->StartNode(1);
  model_builder->LeafScalar(1.0f);
  model_builder->EndNode();

  model_builder->StartNode(2);
  model_builder->LeafScalar(-1.0f);
  model_builder->EndNode();

  model_builder->EndTree();
  return model_builder->CommitModel();
}

}  // namespace

TEST(TreeliteImporter, LargeCategoryValue)
{
  auto tl_model           = make_categorical_model(std::numeric_limits<std::uint32_t>::max());
  auto expected_error_msg = std::string{"Tree 0, Node 0: Category index must be at most "} +
                            std::to_string(std::numeric_limits<std::uint32_t>::max() - 1);

  ASSERT_THAT([&] { import_from_treelite_model(*tl_model, tree_layout::breadth_first); },
              testing::ThrowsMessage<model_import_error>(testing::HasSubstr(expected_error_msg)));
  ASSERT_THAT(
    [&] {
      import_from_treelite_model(
        *tl_model, tree_layout::breadth_first, 0, /*use_double_precision=*/true);
    },
    testing::ThrowsMessage<model_import_error>(testing::HasSubstr(expected_error_msg)));
}

TEST(TreeliteImporter, LargeCategoryValueBelowLimit)
{
  auto tl_model = make_categorical_model(std::numeric_limits<std::uint32_t>::max() - 1);
  ASSERT_NO_THROW(import_from_treelite_model(*tl_model, tree_layout::breadth_first));
}

TEST(TreeliteImporter, LargeFeatureId)
{
  auto metadata = treelite::model_builder::Metadata{
    9000,
    treelite::TaskType::kRegressor,
    false,
    1,
    {1},
    {1, 1},
  };
  auto tree_annotation = treelite::model_builder::TreeAnnotation{1, {0}, {0}};
  auto model_builder =
    treelite::model_builder::GetModelBuilder(treelite::TypeInfo::kFloat32,
                                             treelite::TypeInfo::kFloat32,
                                             metadata,
                                             tree_annotation,
                                             treelite::model_builder::PostProcessorFunc{"identity"},
                                             {0.0});

  model_builder->StartTree();
  model_builder->StartNode(0);
  model_builder->NumericalTest(8999, 0.0, false, treelite::Operator::kGT, 1, 2);
  model_builder->EndNode();

  model_builder->StartNode(1);
  model_builder->LeafScalar(1.0f);
  model_builder->EndNode();

  model_builder->StartNode(2);
  model_builder->LeafScalar(-1.0f);
  model_builder->EndNode();

  model_builder->EndTree();

  auto tl_model = model_builder->CommitModel();
  ASSERT_NO_THROW(import_from_treelite_model(*tl_model, tree_layout::breadth_first));

  auto variant_index = get_forest_variant_index(false, 2, 1);
  auto importer      = treelite_importer<tree_layout::breadth_first>{};

  auto expected_error_msg =
    std::string{"Tree 0, Node 0: The 'feature' value in the node must be at most "} +
    std::to_string(0x1FFF);
  ASSERT_THAT(
    [&] {
      importer.import_to_specific_variant<index_type{}>(variant_index,
                                                        *tl_model,
                                                        importer.get_num_class(*tl_model),
                                                        importer.get_num_feature(*tl_model),
                                                        importer.get_max_num_categories(*tl_model),
                                                        importer.get_offsets(*tl_model));
    },
    testing::ThrowsMessage<model_import_error>(testing::HasSubstr(expected_error_msg)));
}

TEST(TreeliteImporter, SafeCastFloatingPoint)
{
  ASSERT_NO_THROW(detail::safe_cast_floating_point<float>(double{3.1}));
  ASSERT_NO_THROW(detail::safe_cast_floating_point<double>(std::numeric_limits<float>::max()));

  ASSERT_NO_THROW(detail::safe_cast_floating_point<float>(std::numeric_limits<float>::infinity()));
  ASSERT_NO_THROW(
    detail::safe_cast_floating_point<double>(std::numeric_limits<double>::infinity()));
  ASSERT_NO_THROW(detail::safe_cast_floating_point<double>(std::numeric_limits<float>::infinity()));
  ASSERT_NO_THROW(detail::safe_cast_floating_point<float>(std::numeric_limits<float>::quiet_NaN()));
  ASSERT_NO_THROW(
    detail::safe_cast_floating_point<double>(std::numeric_limits<double>::quiet_NaN()));
  ASSERT_NO_THROW(
    detail::safe_cast_floating_point<double>(std::numeric_limits<float>::quiet_NaN()));

  auto inf_msg = std::string{"Cannot cast an INF or NaN value"};
  ASSERT_THAT(
    [] { detail::safe_cast_floating_point<float>(std::numeric_limits<double>::infinity()); },
    testing::ThrowsMessage<detail::floating_point_truncation_error>(testing::HasSubstr(inf_msg)));
  ASSERT_THAT(
    [] { detail::safe_cast_floating_point<float>(std::numeric_limits<double>::quiet_NaN()); },
    testing::ThrowsMessage<detail::floating_point_truncation_error>(testing::HasSubstr(inf_msg)));
  ASSERT_THAT([] { detail::safe_cast_floating_point<float>(double{1e100}); },
              testing::ThrowsMessage<detail::floating_point_truncation_error>(
                testing::HasSubstr("Input must be at most")));
  ASSERT_THAT([] { detail::safe_cast_floating_point<float>(double{-1e100}); },
              testing::ThrowsMessage<detail::floating_point_truncation_error>(
                testing::HasSubstr("Input must be at least")));
}

TEST(TreeliteImporter, InvalidPostprocConstant)
{
  auto metadata = treelite::model_builder::Metadata{
    1,
    treelite::TaskType::kRegressor,
    false,
    1,
    {1},
    {1, 1},
  };
  auto tree_annotation = treelite::model_builder::TreeAnnotation{1, {0}, {0}};
  auto model_builder   = treelite::model_builder::GetModelBuilder(
    treelite::TypeInfo::kFloat32,
    treelite::TypeInfo::kFloat32,
    metadata,
    tree_annotation,
    treelite::model_builder::PostProcessorFunc{
      "sigmoid", {{"sigmoid_alpha", std::numeric_limits<double>::quiet_NaN()}}},
    {0.0});

  model_builder->StartTree();
  model_builder->StartNode(0);
  model_builder->LeafScalar(0.0f);
  model_builder->EndNode();
  model_builder->EndTree();

  auto tl_model = model_builder->CommitModel();

  ASSERT_THAT([&] { import_from_treelite_model(*tl_model, tree_layout::breadth_first); },
              testing::ThrowsMessage<model_import_error>(
                testing::HasSubstr("Found an invalid value for postprocessing constant")));
}

}  // namespace nvforest
