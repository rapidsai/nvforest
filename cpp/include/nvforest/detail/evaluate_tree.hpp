/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <stdint.h>

#include <type_traits>
#ifndef __CUDACC__
#include <math.h>
#endif
#include <nvforest/detail/bitset.hpp>
#include <nvforest/detail/gpu_support.hpp>
namespace nvforest::detail {

/**
 * Convert a floating-point feature value to a categorical bitset index.
 *
 * Truncates toward zero, matching LightGBM's int-cast categorical lookup:
 * values in (-1, 0) map to 0, and any other negative value is out of range,
 * as is anything at or beyond num_bits. A direct float-to-unsigned
 * conversion is undefined behavior for negative values and differs between
 * x86 (wraps out of range) and CUDA (clamps to 0), so the conversion must
 * be explicit. Out-of-range values, including infinities, return an index
 * >= num_bits, which bitset::test reports as false. NaN must be handled by
 * the caller.
 */
template <typename io_t, typename index_t>
HOST DEVICE auto categorical_bitset_index(io_t input_val, index_t num_bits) -> index_t
{
#ifdef __CUDA_ARCH__
  // PTX defines cvt float-to-int as clamping to the int range (NaN -> 0);
  // negative results wrap to >= num_bits, which bitset::test rejects.
  static_cast<void>(num_bits);
  auto truncated = 0;
  if constexpr (std::is_same_v<io_t, float>) {
    asm("cvt.rzi.s32.f32 %0, %1;" : "=r"(truncated) : "f"(input_val));
  } else {
    asm("cvt.rzi.s32.f64 %0, %1;" : "=r"(truncated) : "d"(input_val));
  }
  return static_cast<index_t>(truncated);
#else
  return (input_val > io_t{-1} && input_val < static_cast<io_t>(num_bits))
           ? static_cast<index_t>(input_val)
           : num_bits;
#endif
}

/*
 * Evaluate a single tree on a single row.
 * If node_id_mapping is not-nullptr, this kernel outputs leaf node's ID
 * instead of the leaf value.
 *
 * @tparam has_vector_leaves Whether or not this tree has vector leaves
 * @tparam has_categorical_nodes Whether or not this tree has any nodes with
 * categorical splits
 * @tparam node_t The type of nodes in this tree
 * @tparam io_t The type used for input to and output from this tree (typically
 * either floats or doubles)
 * @tparam node_id_mapping_t If non-nullptr_t, this indicates the type we expect for
 * node_id_mapping.
 * @param node Pointer to the root node of this tree
 * @param row Pointer to the input data for this row
 * @param first_root_node Pointer to the root node of the first tree.
 * @param node_id_mapping Array representing the mapping from internal node IDs to
 * final leaf ID outputs
 */
template <bool has_vector_leaves,
          bool has_categorical_nodes,
          typename node_t,
          typename io_t,
          typename node_id_mapping_t = std::nullptr_t>
HOST DEVICE auto evaluate_tree_impl(node_t const* __restrict__ node,
                                    io_t const* __restrict__ row,
                                    node_t const* __restrict__ first_root_node = nullptr,
                                    node_id_mapping_t node_id_mapping          = nullptr)
{
  using categorical_set_type = bitset<uint32_t, typename node_t::index_type const>;
  auto cur_node              = *node;
  do {
    auto input_val = row[cur_node.feature_index()];
    auto condition = true;
    if constexpr (has_categorical_nodes) {
      if (cur_node.is_categorical()) {
        auto valid_categories = categorical_set_type{
          &cur_node.index(), uint32_t(sizeof(typename node_t::index_type) * 8)};
        condition =
          !isnan(input_val) && valid_categories.test(categorical_bitset_index(
                                 input_val, uint32_t(sizeof(typename node_t::index_type) * 8)));
      } else {
        condition = (input_val < cur_node.threshold());
      }
    } else {
      condition = (input_val < cur_node.threshold());
    }
    if (!condition && cur_node.default_distant()) { condition = isnan(input_val); }
    node += cur_node.child_offset(condition);
    cur_node = *node;
  } while (!cur_node.is_leaf());
  if constexpr (std::is_same_v<node_id_mapping_t, std::nullptr_t>) {
    return cur_node.template output<has_vector_leaves>();
  } else {
    return node_id_mapping[node - first_root_node];
  }
}

/*
 * Evaluate a single tree which requires external categorical storage on a
 * single node.
 * If node_id_mapping is not-nullptr, this kernel outputs leaf node's ID
 * instead of the leaf value.
 *
 * For non-categorical models and models with a relatively small number of
 * categories for any feature, all information necessary for model evaluation
 * can be stored on a single node. If the number of categories for any
 * feature exceeds the available space on a node, however, the
 * categorical split data must be stored external to the node. We pass a
 * pointer to this external data and reconstruct bitsets from it indicating
 * the positive and negative categories for each categorical node.
 *
 * @tparam has_vector_leaves Whether or not this tree has vector leaves
 * @tparam node_t The type of nodes in this tree
 * @tparam io_t The type used for input to and output from this tree (typically
 * either floats or doubles)
 * @tparam categorical_storage_t The underlying type used for storing
 * categorical data (typically char)
 * @tparam node_id_mapping_t If non-nullptr_t, this indicates the type we expect for
 * node_id_mapping.
 * @param node Pointer to the root node of this tree
 * @param row Pointer to the input data for this row
 * @param categorical_storage Pointer to where categorical split data is
 * stored.
 */
template <bool has_vector_leaves,
          typename node_t,
          typename io_t,
          typename categorical_storage_t,
          typename node_id_mapping_t = std::nullptr_t>
HOST DEVICE auto evaluate_tree_impl(node_t const* __restrict__ node,
                                    io_t const* __restrict__ row,
                                    categorical_storage_t const* __restrict__ categorical_storage,
                                    node_t const* __restrict__ first_root_node = nullptr,
                                    node_id_mapping_t node_id_mapping          = nullptr)
{
  using categorical_set_type = bitset<uint32_t, categorical_storage_t const>;
  auto cur_node              = *node;
  do {
    auto input_val = row[cur_node.feature_index()];
    auto condition = cur_node.default_distant();
    if (!isnan(input_val)) {
      if (cur_node.is_categorical()) {
        auto valid_categories =
          categorical_set_type{categorical_storage + cur_node.index() + 1,
                               uint32_t(categorical_storage[cur_node.index()])};
        condition = valid_categories.test(
          categorical_bitset_index(input_val, uint32_t(categorical_storage[cur_node.index()])));
      } else {
        condition = (input_val < cur_node.threshold());
      }
    }
    node += cur_node.child_offset(condition);
    cur_node = *node;
  } while (!cur_node.is_leaf());
  if constexpr (std::is_same_v<node_id_mapping_t, std::nullptr_t>) {
    return cur_node.template output<has_vector_leaves>();
  } else {
    return node_id_mapping[node - first_root_node];
  }
}

/**
 * Dispatch to an appropriate version of evaluate_tree kernel.
 *
 * @tparam has_vector_leaves Whether or not this tree has vector leaves
 * @tparam has_categorical_nodes Whether or not this tree has any nodes with
 * categorical splits
 * @tparam has_nonlocal_categories Whether or not this tree has any nodes that store
 * categorical split data externally
 * @tparam predict_leaf Whether to predict leaf IDs
 * @tparam forest_t The type of forest
 * @tparam io_t The type used for input to and output from this tree (typically
 * either floats or doubles)
 * @tparam categorical_data_t The type for non-local categorical data storage.
 * @param forest The forest used to perform inference
 * @param tree_index The index of the tree we are evaluating
 * @param row The data row we are evaluating
 * @param categorical_data The pointer to where non-local data on categorical splits are stored.
 */
template <bool has_vector_leaves,
          bool has_categorical_nodes,
          bool has_nonlocal_categories,
          bool predict_leaf,
          typename forest_t,
          typename io_t,
          typename categorical_data_t>
HOST DEVICE auto evaluate_tree(forest_t const& forest,
                               index_type tree_index,
                               io_t const* __restrict__ row,
                               categorical_data_t categorical_data)
{
  using node_t = typename forest_t::node_type;
  if constexpr (predict_leaf) {
    auto leaf_node_id = index_type{};
    if constexpr (has_nonlocal_categories) {
      leaf_node_id = evaluate_tree_impl<has_vector_leaves>(forest.get_tree_root(tree_index),
                                                           row,
                                                           categorical_data,
                                                           forest.get_tree_root(0),
                                                           forest.get_node_id_mapping());
    } else {
      leaf_node_id = evaluate_tree_impl<has_vector_leaves, has_categorical_nodes>(
        forest.get_tree_root(tree_index),
        row,
        forest.get_tree_root(0),
        forest.get_node_id_mapping());
    }
    return leaf_node_id;
  } else {
    auto tree_output = std::conditional_t<has_vector_leaves,
                                          typename node_t::index_type,
                                          typename node_t::threshold_type>{};
    if constexpr (has_nonlocal_categories) {
      tree_output = evaluate_tree_impl<has_vector_leaves>(
        forest.get_tree_root(tree_index), row, categorical_data);
    } else {
      tree_output = evaluate_tree_impl<has_vector_leaves, has_categorical_nodes>(
        forest.get_tree_root(tree_index), row);
    }
    return tree_output;
  }
}

}  // namespace nvforest::detail
