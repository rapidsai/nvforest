#
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import pathlib
from typing import Union, Optional

from cuda.bindings import runtime
import treelite
import numpy as np
from pylibraft.common.handle import Handle as RaftHandle

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


class ForestInference:
    def _reload_model(self):
        """Reload model on any device (CPU/GPU) where model has already been
        loaded"""
        if hasattr(self, "_gpu_forest"):
            self._load(device="gpu", device_id=self.device_id)
        if hasattr(self, "_cpu_forest"):
            self._load(device="cpu")

    def _get_default_align_bytes(self):
        if self.device == "cpu":
            return 64
        else:
            return 0

    @property
    def align_bytes(self):
        try:
            return self._align_bytes_
        except AttributeError:
            return self._get_default_align_bytes()

    @align_bytes.setter
    def align_bytes(self, value):
        try:
            old_value = self._align_bytes_
        except AttributeError:
            old_value = None
        if value is None:
            if old_value is not None:
                del self._align_bytes_
                self._reload_model()
        else:
            self._align_bytes_ = value
            if old_value is None or value != old_value:
                self._reload_model()

    @property
    def precision(self):
        try:
            use_double_precision = self._use_double_precision_
        except AttributeError:
            self._use_double_precision_ = False
            use_double_precision = self._use_double_precision_
        if use_double_precision is None:
            return "native"
        elif use_double_precision:
            return "double"
        else:
            return "single"

    @precision.setter
    def precision(self, value):
        try:
            old_value = self._use_double_precision_
        except AttributeError:
            self._use_double_precision_ = False
            old_value = self._use_double_precision_
        if value in ("native", None):
            self._use_double_precision_ = None
        elif value in ("double", 'float64'):
            self._use_double_precision_ = True
        else:
            self._use_double_precision_ = False
        if old_value != self._use_double_precision_:
            self._reload_model()

    @property
    def is_classifier(self):
        try:
            return self._is_classifier_
        except AttributeError:
            self._is_classifier_ = False
            return self._is_classifier_

    @is_classifier.setter
    def is_classifier(self, value):
        if not hasattr(self, "_is_classifier_"):
            self._is_classifier_ = value
        elif value is not None:
            self._is_classifier_ = value

    @property
    def device_id(self):
        try:
            return self._device_id_
        except AttributeError:
            self._device_id_ = None
            return self._device_id_

    @device_id.setter
    def device_id(self, value):
        try:
            old_value = self.device_id
        except AttributeError:
            old_value = None
        self._device_id_ = value
        if (
                self.treelite_model is not None
                and self.device_id != old_value
                and hasattr(self, "_gpu_forest")
        ):
            self._load(device="gpu", device_id=self.device_id)

    @property
    def treelite_model(self):
        try:
            return self._treelite_model_
        except AttributeError:
            return None

    @treelite_model.setter
    def treelite_model(self, value):
        if value is not None:
            self._treelite_model_ = value
            self._reload_model()

    @property
    def layout(self):
        try:
            return self._layout_
        except AttributeError:
            self._layout_ = "depth_first"
        return self._layout_

    @layout.setter
    def layout(self, value):
        try:
            old_value = self._layout_
        except AttributeError:
            old_value = None
        if value is not None:
            self._layout_ = value
        if old_value != value:
            self._reload_model()

    def __init__(
        self,
        *,
        raft_handle: Optional[RaftHandle] = None,
        treelite_model: Optional[treelite.Model] = None,
        is_classifier: bool = False,
        layout: str = "depth_first",
        default_chunk_size: Optional[int] = None,
        align_bytes: Optional[int] = None,
        precision: str = "single",
        device: str = "auto",
        device_id: Optional[int] = None,
    ):
        self.raft_handle = RaftHandle() if raft_handle is None else raft_handle
        self.is_classifier = is_classifier
        self.default_chunk_size = default_chunk_size
        self.align_bytes = align_bytes
        self.layout = layout
        self.precision = precision
        self.device = device
        self.device_id = device_id
        self.treelite_model = treelite_model
        self._load(device=device, device_id=device_id)

    @staticmethod
    def _detect_current_device(
        require: bool
    ) -> Optional[int]:
        """
        Query the currently active GPU.

        Parameters
        ----------
        require:
            Whether to raise an exception when no GPU is available.

        Returns
        -------
        int or None
            ID of the currently active GPU device, or None if no GPU is available.
        """
        status, current_device_id = runtime.cudaGetDevice()
        if status != runtime.cudaError_t.cudaSuccess:
            if not require:
                return None
            _, name = runtime.cudaGetErrorName(status)
            _, msg = runtime.cudaGetErrorString(status)
            raise RuntimeError(f"Failed to run cudaGetDevice(). {name}: {msg}")
        return current_device_id

    def _load(self, device, device_id):
        if device == "auto":
            # Auto mode: Use GPU if available; use CPU otherwise.
            device_id = self._detect_current_device(require=False)
            device = "cpu" if device_id is None else "gpu"
        elif device == "gpu" and device_id is None:
            # If no device ID is explicitly given, use the currently
            # active device
            device_id = self._detect_current_device(require=True)

        self.device_id = device_id if device == "gpu" else -1

        if self.treelite_model is not None:
            if isinstance(self.treelite_model, treelite.Model):
                treelite_model_bytes = self.treelite_model.serialize_bytes()
            elif isinstance(self.treelite_model, bytes):
                treelite_model_bytes = self.treelite_model
            else:
                raise ValueError("treelite_model should be either treelite.Model or bytes")
            impl = ForestInference_impl(
                self.raft_handle,
                treelite_model_bytes,
                layout=self.layout,
                align_bytes=self.align_bytes,
                use_double_precision=self._use_double_precision_,
                device=device,
                device_id=self.device_id
            )

            if device == "gpu":
                self._gpu_forest = impl
            else:
                self._cpu_forest = impl


def load_model(
    model_file: Union[str, pathlib.Path],
    device: str,
):
    pass
