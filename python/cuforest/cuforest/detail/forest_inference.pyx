#
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

from typing import Optional, Union

import numpy as np
import treelite

from cuforest._handle import Handle
from cuforest._typing import DataType
from cuforest.detail.treelite import safe_treelite_call

from libc.stdint cimport uint32_t, uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t as raft_handle_t

from cuforest.detail.infer_kind cimport infer_kind
from cuforest.detail.postprocessing cimport element_op, row_op
from cuforest.detail.raft_proto.cuda_stream cimport (
    cuda_stream as raft_proto_stream_t,
)
from cuforest.detail.raft_proto.device_type cimport (
    device_type as raft_proto_device_t,
)
from cuforest.detail.raft_proto.handle cimport handle_t as raft_proto_handle_t
from cuforest.detail.raft_proto.optional cimport nullopt, optional
from cuforest.detail.tree_layout cimport tree_layout as cuforest_tree_layout
from cuforest.detail.treelite cimport (
    TreeliteDeserializeModelFromBytes,
    TreeliteFreeModel,
    TreeliteModelHandle,
)


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
    cdef object device

    def __cinit__(
        self,
        raft_handle: object,
        tl_model_bytes: Union[bytes, bytearray],
        *,
        layout: str = "depth_first",
        align_bytes: int = 0,
        use_double_precision: Optional[bool] = None,
        device: str = "cpu",
        device_id: Optional[int] = None,
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
        self.device = device

        print(f"device_id = {device_id}")
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
        return ""

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
        return ""

    def predict(
        self,
        X: DataType,
        *,
        predict_type: str = "default",
        chunk_size: Optional[int] = None,
    ) -> DataType:
        cdef uintptr_t in_ptr
        cdef raft_proto_device_t in_dev
        cdef uintptr_t out_ptr
        cdef raft_proto_device_t out_dev
        cdef infer_kind infer_type_enum
        cdef optional[uint32_t] chunk_specification

        n_rows = X.shape[0]
        model_dtype = self.get_dtype()

        if predict_type == "default":
            infer_type_enum = infer_kind.default_kind
            output_shape = (n_rows, self.model.num_outputs())
        elif predict_type == "per_tree":
            infer_type_enum = infer_kind.per_tree
            if self.model.has_vector_leaves():
                output_shape = (n_rows, self.model.num_trees(), self.model.num_outputs())
            else:
                output_shape = (n_rows, self.model.num_trees())
        elif predict_type == "leaf_id":
            infer_type_enum = infer_kind.leaf_id
            output_shape = (n_rows, self.model.num_trees())
        else:
            raise ValueError(f"Unrecognized predict_type: {predict_type}")

        if self.device == "cpu":
            X = np.asarray(X, dtype=model_dtype, order="C")
            preds = np.empty(
                shape=output_shape,
                dtype=model_dtype,
                order="C",
            )
            in_ptr = X.__array_interface__["data"][0]
            in_dev = raft_proto_device_t.cpu
            out_ptr = preds.__array_interface__["data"][0]
            out_dev = raft_proto_device_t.cpu
        else:
            assert self.device == "gpu"
            import cupy as cp
            X = cp.asarray(X, dtype=model_dtype, order="C", blocking=True)
            preds = cp.empty(
                shape=output_shape,
                dtype=model_dtype,
                order="C",
            )
            in_ptr = X.__cuda_array_interface__["data"][0]
            in_dev = raft_proto_device_t.gpu
            out_ptr = preds.__cuda_array_interface__["data"][0]
            out_dev = raft_proto_device_t.gpu

        if chunk_size is None:
            chunk_specification = nullopt
        else:
            chunk_specification = <uint32_t> chunk_size

        if model_dtype == np.float32:
            self.model.predict[float](
                self.raft_proto_handle,
                <float *> out_ptr,
                <float *> in_ptr,
                n_rows,
                out_dev,
                in_dev,
                infer_type_enum,
                chunk_specification
            )
        else:
            self.model.predict[double](
                self.raft_proto_handle,
                <double *> out_ptr,
                <double *> in_ptr,
                n_rows,
                out_dev,
                in_dev,
                infer_type_enum,
                chunk_specification
            )

        if self.device == "gpu":
            self.raft_proto_handle.synchronize()
        return preds


class ForestInferenceImpl:
    def __init__(
        self,
        *,
        treelite_model: treelite.Model,
        device: str,
        device_id: int,
        handle: Optional[Handle] = None,
        layout: str = "depth_first",
        default_chunk_size: Optional[int] = None,
        align_bytes: Optional[int] = None,
        precision: Optional[str] = None,
    ):
        # Assumption: The caller needs to pass in correct (device, device_id) pair
        # This function will not contain any logic for auto-detecting device.
        self.handle = Handle() if handle is None else handle
        self._layout = layout
        self.precision = precision
        self.default_chunk_size = default_chunk_size
        self.device = device
        self.device_id = device_id

        if align_bytes is not None:
            self.align_bytes = align_bytes
        else:
            self.align_bytes = 64 if self.device == "cpu" else 0

        # TODO(hcho3): Change this to use the match statement
        #              once Cython supports structural pattern matching.
        #              See https://github.com/cython/cython/issues/4029
        if self.precision in ("native", None):
            self._use_double_precision = None
        elif self.precision in ("double", "float64"):
            self._use_double_precision = True
        elif self.precision in ("single", "float32"):
            self._use_double_precision = False
        else:
            raise ValueError(f"Unknown precision value: {self.precision}")

        # Store treelite model bytes for creating new instances with different settings
        self._treelite_model_bytes = treelite_model.serialize_bytes()

        self.impl = ForestInference_impl(
            self.handle,
            self._treelite_model_bytes,
            layout=self._layout,
            align_bytes=self.align_bytes,
            use_double_precision=self._use_double_precision,
            device=self.device,
            device_id=self.device_id
        )

    @property
    def layout(self) -> str:
        return self._layout

    @property
    def treelite_model_bytes(self) -> bytes:
        """Return the serialized treelite model bytes."""
        return self._treelite_model_bytes

    @property
    def num_outputs(self) -> int:
        return self.impl.num_outputs()

    @property
    def num_trees(self) -> int:
        return self.impl.num_trees()

    @property
    def num_features(self) -> int:
        return self.impl.num_features()

    def get_dtype(self):
        """Return the dtype (float32 or float64) used by the model."""
        return self.impl.get_dtype()

    @property
    def row_postprocessing(self) -> str:
        return self.impl.row_postprocessing()

    @property
    def elem_postprocessing(self) -> str:
        return self.impl.elem_postprocessing()

    def predict(
        self,
        X: DataType,
        *,
        chunk_size: Optional[int] = None,
    ) -> DataType:
        # Returns probabilities if the model is a classifier
        return self.impl.predict(
            X, chunk_size=(chunk_size or self.default_chunk_size)
        )

    def predict_per_tree(
        self,
        X: DataType,
        *,
        chunk_size: Optional[int] = None,
    ) -> DataType:
        chunk_size = (chunk_size or self.default_chunk_size)
        return self.impl.predict(
            X, predict_type="per_tree", chunk_size=chunk_size
        )

    def apply(
        self,
        X: DataType,
        *,
        chunk_size: Optional[int] = None,
    ) -> DataType:
        chunk_size = (chunk_size or self.default_chunk_size)
        return self.impl.predict(
            X, predict_type="leaf_id", chunk_size=chunk_size
        )
