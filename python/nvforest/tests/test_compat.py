# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Compatibility test to ensure that legacy APIs are working
"""

import cupy as cp
import numpy as np
import pytest
import treelite
from cuda.core import Device

import nvforest


def test_deprecated_handle():
    """Test use of RAFT handle with non-default stream"""
    builder = treelite.model_builder.ModelBuilder(
        threshold_type="float64",
        leaf_output_type="float64",
        metadata=treelite.model_builder.Metadata(
            num_feature=1,
            task_type="kRegressor",
            average_tree_output=False,
            num_target=1,
            num_class=[1],
            leaf_vector_shape=(1, 1),
        ),
        tree_annotation=treelite.model_builder.TreeAnnotation(
            num_tree=1, target_id=[0], class_id=[0]
        ),
        postprocessor=treelite.model_builder.PostProcessorFunc(
            name="identity"
        ),
        base_scores=[0.0],
    )
    builder.start_tree()
    builder.start_node(0)
    builder.numerical_test(
        feature_id=0,
        threshold=0.0,
        default_left=False,
        opname="<",
        left_child_key=1,
        right_child_key=2,
    )
    builder.end_node()
    builder.start_node(1)
    builder.leaf(-1.0)
    builder.end_node()
    builder.start_node(2)
    builder.leaf(1.0)
    builder.end_node()
    builder.end_tree()

    treelite_model = builder.commit()

    device = Device()
    device.set_current()
    stream = device.create_stream()
    handle = nvforest.Handle(stream=stream)

    rng = np.random.default_rng(42)
    X = rng.standard_normal(size=(1000, 1), dtype="float64")

    with pytest.warns(
        FutureWarning,
        match=".*`handle` parameter is deprecated and will be removed in 27.02.*",
    ):
        fm = nvforest.load_from_treelite_model(
            treelite_model, device="gpu", handle=handle
        )
        pred_with_handle = cp.asnumpy(fm.predict(X))

    stream2 = cp.cuda.Stream(non_blocking=True)
    fm2 = nvforest.load_from_treelite_model(
        treelite_model, device="gpu", stream=stream2
    )
    pred_with_stream = cp.asnumpy(fm2.predict(X))
    np.testing.assert_almost_equal(pred_with_handle, pred_with_stream)
