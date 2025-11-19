#
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import pathlib
from typing import Union

import numpy as np

from cuforest.detail.treelite import safe_treelite_call

from libc.stdint cimport uint32_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t as raft_handle_t

from cuforest.detail.raft_proto.handle cimport handle_t as raft_proto_handle_t
from cuforest.detail.raft_proto.optional cimport nullopt, optional
from cuforest.detail.raft_proto.cuda_stream cimport (
    cuda_stream as raft_proto_stream_t,
)
from cuforest.detail.raft_proto.device_type cimport (
    device_type as raft_proto_device_t,
)
from cuforest.detail.treelite cimport (
    TreeliteDeserializeModelFromBytes,
    TreeliteFreeModel,
    TreeliteModelHandle,
)
from cuforest.postprocessing cimport element_op, row_op
from cuforest.tree_layout cimport tree_layout as cuforest_tree_layout

cdef extern from "cuforest/forest_model.hpp" namespace "cuforest" nogil:
    cdef cppclass forest_model:
        void predict[io_t](
            const raft_proto_handle_t&,
            io_t*,
            io_t*,
            size_t,
            raft_proto_device_t,
            raft_proto_device_t,
            infer_kind,
            optional[uint32_t]
        ) except +

        bool is_double_precision() except +
        size_t num_features() except +
        size_t num_outputs() except +
        size_t num_trees() except +
        bool has_vector_leaves() except +
        row_op row_postprocessing() except +
        element_op elem_postprocessing() except +

cdef extern from "cuforest/treelite_importer.hpp" namespace "cuforest" nogil:
    forest_model import_from_treelite_handle(
        TreeliteModelHandle,
        cuforest_tree_layout,
        uint32_t,
        optional[bool],
        raft_proto_device_t,
        int,
        raft_proto_stream_t
    ) except +


cdef class ForestInference_impl():
    cdef forest_model model
    cdef raft_proto_handle_t raft_proto_handle
    cdef object raft_handle

    def __cinit__(
        self,
        raft_handle,
        tl_model_bytes,
        *,
        layout="depth_first",
        align_bytes=0,
        use_double_precision=None,
        device="cpu",
        device_id=None,
    ):
        # Store reference to RAFT handle to control lifetime, since raft_proto
        # handle keeps a pointer to it
        self.raft_handle = raft_handle
        self.raft_proto_handle = raft_proto_handle_t(
            <raft_handle_t*><size_t>self.raft_handle.getHandle()
        )

        cdef optional[bool] use_double_precision_c
        cdef bool use_double_precision_bool
        if use_double_precision is None:
            use_double_precision_c = nullopt
        else:
            use_double_precision_bool = use_double_precision
            use_double_precision_c = use_double_precision_bool

        cdef TreeliteModelHandle tl_handle = NULL
        safe_treelite_call(
            TreeliteDeserializeModelFromBytes(
                tl_model_bytes, len(tl_model_bytes), &tl_handle),
            "Failed to load Treelite model from bytes:"
        )

        cdef raft_proto_device_t dev_type
        dev_type = raft_proto_device_t.gpu if device == "gpu" else raft_proto_device_t.cpu
        cdef cuforest_tree_layout tree_layout
        if layout.lower() == "depth_first":
            tree_layout = cuforest_tree_layout.depth_first
        elif layout.lower() == "breadth_first":
            tree_layout = cuforest_tree_layout.breadth_first
        elif layout.lower() == "layered":
            tree_layout = cuforest_tree_layout.layered_children_together
        else:
            raise RuntimeError(f"Unrecognized tree layout {layout}")

        # Use assertion here, since device_id being None would indicate
        # a bug, not a user error. The outer ForestInference object
        # should set an integer device_id before passing it to
        # ForestInference_impl.
        assert device_id is not None, (
            "device_id should be set before building ForestInference_impl"
        )

        self.model = import_from_treelite_handle(
            tl_handle,
            tree_layout,
            align_bytes,
            use_double_precision_c,
            dev_type,
            device_id,
            self.raft_proto_handle.get_next_usable_stream()
        )

        safe_treelite_call(
            TreeliteFreeModel(tl_handle),
            "Failed to free Treelite model:"
        )

    def get_dtype(self):
        return [np.float32, np.float64][self.model.is_double_precision()]

    def num_features(self):
        return self.model.num_features()

    def num_outputs(self):
        return self.model.num_outputs()

    def num_trees(self):
        return self.model.num_trees()

    def row_postprocessing(self):
        enum_val = self.model.row_postprocessing()
        if enum_val == row_op.row_disable:
            return "disable"
        elif enum_val == row_op.softmax:
            return "softmax"
        elif enum_val == row_op.max_index:
            return "max_index"

    def elem_postprocessing(self):
        enum_val = self.model.elem_postprocessing()
        if enum_val == element_op.elem_disable:
            return "disable"
        elif enum_val == element_op.signed_square:
            return "signed_square"
        elif enum_val == element_op.hinge:
            return "hinge"
        elif enum_val == element_op.sigmoid:
            return "sigmoid"
        elif enum_val == element_op.exponential:
            return "exponential"
        elif enum_val == element_op.logarithm_one_plus_exp:
            return "logarithm_one_plus_exp"


def load_model(
    model_file: Union[str, pathlib.Path],
    device: str
):
    pass
