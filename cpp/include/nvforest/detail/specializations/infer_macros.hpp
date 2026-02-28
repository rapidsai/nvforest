/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <nvforest/constants.hpp>
#include <nvforest/detail/forest.hpp>
#include <nvforest/detail/index_type.hpp>
#include <nvforest/detail/postprocessor.hpp>
#include <nvforest/detail/raft_proto/cuda_stream.hpp>
#include <nvforest/detail/raft_proto/device_id.hpp>
#include <nvforest/detail/raft_proto/device_type.hpp>
#include <nvforest/detail/specialization_types.hpp>
#include <nvforest/detail/specializations/forest_macros.hpp>
#include <nvforest/infer_kind.hpp>

#include <cstddef>
#include <variant>

/* Macro which expands to the valid arguments to an inference call for a forest
 * model without vector leaves or non-local categorical data.*/
#define NVFOREST_SCALAR_LOCAL_ARGS(dev, variant_index)                 \
  (NVFOREST_FOREST(variant_index) const&,                              \
   postprocessor<NVFOREST_SPEC(variant_index)::threshold_type> const&, \
   NVFOREST_SPEC(variant_index)::threshold_type*,                      \
   NVFOREST_SPEC(variant_index)::threshold_type*,                      \
   index_type,                                                         \
   index_type,                                                         \
   index_type,                                                         \
   std::nullptr_t,                                                     \
   std::nullptr_t,                                                     \
   infer_kind,                                                         \
   std::optional<index_type>,                                          \
   raft_proto::device_id<dev>,                                         \
   raft_proto::cuda_stream stream)

/* Macro which expands to the valid arguments to an inference call for a forest
 * model with vector leaves but without non-local categorical data.*/
#define NVFOREST_VECTOR_LOCAL_ARGS(dev, variant_index)                 \
  (NVFOREST_FOREST(variant_index) const&,                              \
   postprocessor<NVFOREST_SPEC(variant_index)::threshold_type> const&, \
   NVFOREST_SPEC(variant_index)::threshold_type*,                      \
   NVFOREST_SPEC(variant_index)::threshold_type*,                      \
   index_type,                                                         \
   index_type,                                                         \
   index_type,                                                         \
   NVFOREST_SPEC(variant_index)::threshold_type*,                      \
   std::nullptr_t,                                                     \
   infer_kind,                                                         \
   std::optional<index_type>,                                          \
   raft_proto::device_id<dev>,                                         \
   raft_proto::cuda_stream stream)

/* Macro which expands to the valid arguments to an inference call for a forest
 * model without vector leaves but with non-local categorical data.*/
#define NVFOREST_SCALAR_NONLOCAL_ARGS(dev, variant_index)              \
  (NVFOREST_FOREST(variant_index) const&,                              \
   postprocessor<NVFOREST_SPEC(variant_index)::threshold_type> const&, \
   NVFOREST_SPEC(variant_index)::threshold_type*,                      \
   NVFOREST_SPEC(variant_index)::threshold_type*,                      \
   index_type,                                                         \
   index_type,                                                         \
   index_type,                                                         \
   std::nullptr_t,                                                     \
   NVFOREST_SPEC(variant_index)::index_type*,                          \
   infer_kind,                                                         \
   std::optional<index_type>,                                          \
   raft_proto::device_id<dev>,                                         \
   raft_proto::cuda_stream stream)

/* Macro which expands to the valid arguments to an inference call for a forest
 * model with vector leaves and with non-local categorical data.*/
#define NVFOREST_VECTOR_NONLOCAL_ARGS(dev, variant_index)              \
  (NVFOREST_FOREST(variant_index) const&,                              \
   postprocessor<NVFOREST_SPEC(variant_index)::threshold_type> const&, \
   NVFOREST_SPEC(variant_index)::threshold_type*,                      \
   NVFOREST_SPEC(variant_index)::threshold_type*,                      \
   index_type,                                                         \
   index_type,                                                         \
   index_type,                                                         \
   NVFOREST_SPEC(variant_index)::threshold_type*,                      \
   NVFOREST_SPEC(variant_index)::index_type*,                          \
   infer_kind,                                                         \
   std::optional<index_type>,                                          \
   raft_proto::device_id<dev>,                                         \
   raft_proto::cuda_stream stream)

/* Macro which expands to the declaration of an inference template for a forest
 * of the type indicated by the variant index */
#define NVFOREST_INFER_TEMPLATE(template_type, dev, variant_index, categorical) \
  template_type void infer<dev, categorical, NVFOREST_FOREST(variant_index)>

/* Macro which expands to the declaration of an inference template for a forest
 * of the type indicated by the variant index on the given device type without
 * vector leaves or categorical nodes*/
#define NVFOREST_INFER_DEV_SCALAR_LEAF_NO_CAT(template_type, dev, variant_index) \
  NVFOREST_INFER_TEMPLATE(template_type, dev, variant_index, false)              \
  NVFOREST_SCALAR_LOCAL_ARGS(dev, variant_index);

/* Macro which expands to the declaration of an inference template for a forest
 * of the type indicated by the variant index on the given device type without
 * vector leaves and with only local categorical nodes*/
#define NVFOREST_INFER_DEV_SCALAR_LEAF_LOCAL_CAT(template_type, dev, variant_index) \
  NVFOREST_INFER_TEMPLATE(template_type, dev, variant_index, true)                  \
  NVFOREST_SCALAR_LOCAL_ARGS(dev, variant_index);

/* Macro which expands to the declaration of an inference template for a forest
 * of the type indicated by the variant index on the given device type without
 * vector leaves and with non-local categorical nodes*/
#define NVFOREST_INFER_DEV_SCALAR_LEAF_NONLOCAL_CAT(template_type, dev, variant_index) \
  NVFOREST_INFER_TEMPLATE(template_type, dev, variant_index, true)                     \
  NVFOREST_SCALAR_NONLOCAL_ARGS(dev, variant_index);

/* Macro which expands to the declaration of an inference template for a forest
 * of the type indicated by the variant index on the given device type with
 * vector leaves and without categorical nodes*/
#define NVFOREST_INFER_DEV_VECTOR_LEAF_NO_CAT(template_type, dev, variant_index) \
  NVFOREST_INFER_TEMPLATE(template_type, dev, variant_index, false)              \
  NVFOREST_VECTOR_LOCAL_ARGS(dev, variant_index);

/* Macro which expands to the declaration of an inference template for a forest
 * of the type indicated by the variant index on the given device type with
 * vector leaves and with only local categorical nodes*/
#define NVFOREST_INFER_DEV_VECTOR_LEAF_LOCAL_CAT(template_type, dev, variant_index) \
  NVFOREST_INFER_TEMPLATE(template_type, dev, variant_index, true)                  \
  NVFOREST_VECTOR_LOCAL_ARGS(dev, variant_index);

/* Macro which expands to the declaration of an inference template for a forest
 * of the type indicated by the variant index on the given device type with
 * vector leaves and with non-local categorical nodes*/
#define NVFOREST_INFER_DEV_VECTOR_LEAF_NONLOCAL_CAT(template_type, dev, variant_index) \
  NVFOREST_INFER_TEMPLATE(template_type, dev, variant_index, true)                     \
  NVFOREST_VECTOR_NONLOCAL_ARGS(dev, variant_index);

/* Macro which expands to the declaration of all valid inference templates for
 * the given device on the forest type specified by the given variant index */
#define NVFOREST_INFER_ALL(template_type, dev, variant_index)                    \
  NVFOREST_INFER_DEV_SCALAR_LEAF_NO_CAT(template_type, dev, variant_index)       \
  NVFOREST_INFER_DEV_SCALAR_LEAF_LOCAL_CAT(template_type, dev, variant_index)    \
  NVFOREST_INFER_DEV_SCALAR_LEAF_NONLOCAL_CAT(template_type, dev, variant_index) \
  NVFOREST_INFER_DEV_VECTOR_LEAF_NO_CAT(template_type, dev, variant_index)       \
  NVFOREST_INFER_DEV_VECTOR_LEAF_LOCAL_CAT(template_type, dev, variant_index)    \
  NVFOREST_INFER_DEV_VECTOR_LEAF_NONLOCAL_CAT(template_type, dev, variant_index)
