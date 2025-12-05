#
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import pathlib
from enum import Enum
from typing import Union, Optional, Any

from cuda.bindings import runtime
import treelite
import numpy as np
from pylibraft.common.handle import Handle as RaftHandle

from cuforest.detail.treelite import safe_treelite_call

from libc.stdint cimport uint32_t, uintptr_t
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
from cuforest.infer_kind cimport infer_kind
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


DataType = Union[np.ndarray, "cupy.ndarray"]


# TaskType enum class from Treelite
class TaskTypeEnum(Enum):
    kBinaryClf = 0
    kRegressor = 1
    kMultiClf = 2
    kLearningToRank = 3
    kIsolationForest = 4


def _infer_is_classifier(treelite_model: treelite.Model) -> bool:
    header = treelite_model.get_header_accessor()
    return header.get_field("task_type") in (
        TaskTypeEnum.kBinaryClf.value,
        TaskTypeEnum.kMultiClf.value,
    )


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


class ForestInference:
    def _reload_model(self):
        """Reload model on any device (CPU/GPU) where model has already been
        loaded"""
        if hasattr(self, "_gpu_forest"):
            self._load(device="gpu", device_id=self.device_id)
        if hasattr(self, "_cpu_forest"):
            self._load(device="cpu", device_id=None)

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
        elif value in ("double", "float64"):
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
            and self.device == "gpu"
            and self.device_id != old_value
            and hasattr(self, "_gpu_forest")
        ):
            self._load(device="gpu", device_id=self.device_id)

    @property
    def treelite_model(self) -> treelite.Model:
        try:
            return self._treelite_model_
        except AttributeError:
            return None

    @treelite_model.setter
    def treelite_model(self, value: treelite.Model):
        if value is not None:
            self._treelite_model_ = value
            self._is_classifier_ = _infer_is_classifier(self._treelite_model_)
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
        layout: str = "depth_first",
        default_chunk_size: Optional[int] = None,
        align_bytes: Optional[int] = None,
        precision: Optional[str] = "single",
        device: str = "auto",
        device_id: Optional[int] = None,
    ):
        self.raft_handle = RaftHandle() if raft_handle is None else raft_handle
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
            name, msg = name.decode("utf-8"), msg.decode("utf-8")
            raise RuntimeError(
                f"Failed to detect a GPU device. Diagnostic:\n"
                f"    {name}: {msg}"
            )
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

    @property
    def gpu_forest(self):
        """The underlying cuForest forest model loaded in GPU-accessible memory"""
        try:
            return self._gpu_forest
        except AttributeError:
            self._load(device="gpu", device_id=self.device_id)
            return self._gpu_forest

    @property
    def cpu_forest(self):
        """The underlying cuForest forest model loaded in CPU-accessible memory"""
        try:
            return self._cpu_forest
        except AttributeError:
            self._load(device="cpu", device_id=None)
            return self._cpu_forest

    @property
    def forest(self):
        """The underlying cuForest forest model loaded in memory compatible with the
        current device setting"""
        if self.device == "gpu":
            return self.gpu_forest
        return self.cpu_forest

    def num_outputs(self):
        return self.forest.num_outputs()

    def num_trees(self):
        return self.forest.num_trees()

    def predict_proba(
        self,
        X: DataType,
        *,
        chunk_size: Optional[int] = None,
    ) -> DataType:
        """
        Predict the class probabilities for each row in X.

        Parameters
        ----------
        X :
            The input data of shape Rows * Features. This can be a numpy
            array or cupy array. cuForest is optimized for C-major arrays (e.g.
            numpy/cupy arrays). Inputs whose datatype does not match the
            precision of the loaded model (float/double) will be converted
            to the correct datatype before inference. If this input is in a
            memory location that is inaccessible to the current device type
            (as set with the 'device' parameter in the constructor),
            it will be copied to the correct location. This copy will be
            distributed across as many CUDA streams as are available
            in the stream pool of the model's RAFT handle.
        chunk_size :
            The number of rows to simultaneously process in one iteration
            of the inference algorithm. Batches are further broken down into
            "chunks" of this size when assigning available threads to tasks.
            The choice of chunk size can have a substantial impact on
            performance, but the optimal choice depends on model and
            hardware and is difficult to predict a priori. In general,
            larger batch sizes benefit from larger chunk sizes, and smaller
            batch sizes benefit from small chunk sizes. On GPU, valid
            values are powers of 2 from 1 to 32. On CPU, valid values are
            any power of 2, but little benefit is expected above a chunk size
            of 512.
        """
        if not self.is_classifier:
            raise RuntimeError(
                "predict_proba is not available for regression models. Load"
                " with is_classifier=True if this is a classifier."
            )
        return self.forest.predict(
            X, chunk_size=(chunk_size or self.default_chunk_size)
        )

    def predict(
        self,
        X: DataType,
        *,
        chunk_size: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> DataType:
        """
        For classification models, predict the class for each row. For
        regression models, predict the output for each row.

        Parameters
        ----------
        X
            The input data of shape Rows X Features. This can be a numpy
            array or cupy array. cuForest is optimized for C-major arrays (e.g.
            numpy/cupy arrays). Inputs whose datatype does not match the
            precision of the loaded model (float/double) will be converted
            to the correct datatype before inference. If this input is in a
            memory location that is inaccessible to the current device type
            (as set with the 'device' parameter in the constructor),
            it will be copied to the correct location. This copy will be
            distributed across as many CUDA streams as are available
            in the stream pool of the model's RAFT handle.
        chunk_size :
            The number of rows to simultaneously process in one iteration
            of the inference algorithm. Batches are further broken down into
            "chunks" of this size when assigning available threads to tasks.
            The choice of chunk size can have a substantial impact on
            performance, but the optimal choice depends on model and
            hardware and is difficult to predict a priori. In general,
            larger batch sizes benefit from larger chunk sizes, and smaller
            batch sizes benefit from small chunk sizes. On GPU, valid
            values are powers of 2 from 1 to 32. On CPU, valid values are
            any power of 2, but little benefit is expected above a chunk size
            of 512.
        threshold :
            For binary classifiers, output probabilities above this threshold
            will be considered positive detections. If None, a threshold
            of 0.5 will be used for binary classifiers. For multiclass
            classifiers, the highest probability class is chosen regardless
            of threshold.
        """
        chunk_size = (chunk_size or self.default_chunk_size)
        if self.forest.row_postprocessing() == "max_index":
            raw_out = self.forest.predict(X, chunk_size=chunk_size)
            return raw_out[:, 0]
        elif self.is_classifier:
            proba = self.forest.predict(X, chunk_size=chunk_size)
            if len(proba.shape) < 2 or proba.shape[1] == 1:
                if threshold is None:
                    threshold = 0.5
                result = (proba > threshold).astype("int")
            else:
                result = proba.argmax(axis=1)
            return result
        else:
            return self.forest.predict(
                X, predict_type="default", chunk_size=chunk_size
            )

    def predict_per_tree(
        self,
        X: DataType,
        *,
        chunk_size: Optional[int] = None,
    ) -> DataType:
        """
        Output prediction of each tree.
        This function computes one or more margin scores per tree.

        Parameters
        ----------
        X:
            The input data of shape Rows X Features. This can be a numpy
            array or cupy array. cuForest is optimized for C-major arrays (e.g.
            numpy/cupy arrays). Inputs whose datatype does not match the
            precision of the loaded model (float/double) will be converted
            to the correct datatype before inference. If this input is in a
            memory location that is inaccessible to the current device type
            (as set with the 'device' parameter in the constructor),
            it will be copied to the correct location. This copy will be
            distributed across as many CUDA streams as are available
            in the stream pool of the model's RAFT handle.
        chunk_size :
            The number of rows to simultaneously process in one iteration
            of the inference algorithm. Batches are further broken down into
            "chunks" of this size when assigning available threads to tasks.
            The choice of chunk size can have a substantial impact on
            performance, but the optimal choice depends on model and
            hardware and is difficult to predict a priori. In general,
            larger batch sizes benefit from larger chunk sizes, and smaller
            batch sizes benefit from small chunk sizes. On GPU, valid
            values are powers of 2 from 1 to 32. On CPU, valid values are
            any power of 2, but little benefit is expected above a chunk size
            of 512.
        """
        chunk_size = (chunk_size or self.default_chunk_size)
        return self.forest.predict(
            X, predict_type="per_tree", chunk_size=chunk_size
        )

    def apply(
        self,
        X: DataType,
        *,
        chunk_size: Optional[int] = None,
    ) -> DataType:
        """
        Output the ID of the leaf node for each tree.

        Parameters
        ----------
        X
            The input data of shape Rows X Features. This can be a numpy
            array or cupy array. cuForest is optimized for C-major arrays (e.g.
            numpy/cupy arrays). Inputs whose datatype does not match the
            precision of the loaded model (float/double) will be converted
            to the correct datatype before inference. If this input is in a
            memory location that is inaccessible to the current device type
            (as set with the 'device' parameter in the constructor),
            it will be copied to the correct location. This copy will be
            distributed across as many CUDA streams as are available
            in the stream pool of the model's RAFT handle.
        chunk_size :
            The number of rows to simultaneously process in one iteration
            of the inference algorithm. Batches are further broken down into
            "chunks" of this size when assigning available threads to tasks.
            The choice of chunk size can have a substantial impact on
            performance, but the optimal choice depends on model and
            hardware and is difficult to predict a priori. In general,
            larger batch sizes benefit from larger chunk sizes, and smaller
            batch sizes benefit from small chunk sizes. On GPU, valid
            values are powers of 2 from 1 to 32. On CPU, valid values are
            any power of 2, but little benefit is expected above a chunk size
            of 512.
        """
        return self.forest.predict(
            X, predict_type="leaf_id", chunk_size=chunk_size
        )


def load_model(
    model_file: Union[str, pathlib.Path],
    *,
    model_type: Optional[str] = None,
    device: str = "auto",
    layout: str = "depth_first",
    default_chunk_size: Optional[int] = None,
    align_bytes: Optional[int] = None,
    precision: Optional[str] = None,
    device_id: Optional[int] = None,
    raft_handle: Optional[RaftHandle] = None,
) -> ForestInference:
    """Load a model into cuForest from a serialized model file.

    Parameters
    ----------
    model_file :
        The path to the serialized model file. This can be an XGBoost
        binary or JSON file, a LightGBM text file, or a Treelite checkpoint
        file. If the model_type parameter is not passed, an attempt will be
        made to load the file based on its extension.
    model_type : {"xgboost_ubj", "xgboost_json", "xgboost_legacy", "lightgbm",
        "treelite_checkpoint", None}, default=None
        The serialization format for the model file. If None, a best-effort
        guess will be made based on the file extension.
    device: {"auto", "gpu", "cpu"}, default="auto"
        Whether to use GPU or CPU for inferencing. If set to "auto", GPU will
        be selected if it is available.
    layout : {"breadth_first", "depth_first", "layered"}, default="depth_first"
        The in-memory layout to be used during inference for nodes of the
        forest model. This parameter is available purely for runtime
        optimization. For performance-critical applications, it is
        recommended that available layouts be tested with realistic batch
        sizes to determine the optimal value.
    default_chunk_size : int or None, default=None
        If set, predict calls without a specified chunk size will use
        this default value.
    align_bytes : int or None, default=None
        Pad each tree with empty nodes until its in-memory size is a multiple
        of the given value. If None, use 0 for GPU and 64 for CPU.
    precision : {"single", "double", None}, default="single"
        Use the given floating point precision for evaluating the model. If
        None, use the native precision of the model. Note that
        single-precision execution is substantially faster than
        double-precision execution, so double-precision is recommended
        only for models trained and double precision and when exact
        conformance between results from cuForest and the original training
        framework is of paramount importance.
    device_id : int or None, default=None
        For GPU execution, the device on which to load and execute this
        model. For CPU execution, this value is currently ignored.
    raft_handle : pylibraft.common.handle or None
        For GPU execution, the RAFT handle containing the stream or stream
        pool to use during loading and inference. If not given, a new
        handle will be constructed.
    """
    model_path = pathlib.Path(model_file)
    if not model_path.exists():
        raise ValueError(f"Model file {model_file} does not exist")
    if model_type is None:
        extension = model_path.suffix
        if extension == ".json":
            model_type = "xgboost_json"
        elif extension == ".ubj":
            model_type = "xgboost_ubj"
        elif extension == ".model":
            model_type = "xgboost"
        elif extension == ".txt":
            model_type = "lightgbm"
        else:
            model_type = "treelite_checkpoint"
    if model_type == "treelite_checkpoint":
        tl_model = treelite.frontend.Model.deserialize(model_path)
    elif model_type == "xgboost_ubj":
        tl_model = treelite.frontend.load_xgboost_model(model_path, format_choice="ubjson")
    elif model_type == "xgboost_json":
        tl_model = treelite.frontend.load_xgboost_model(model_path, format_choice="json")
    elif model_type == "xgboost":
        tl_model = treelite.frontend.load_xgboost_model_legacy_binary(model_path)
    elif model_type == "lightgbm":
        tl_model = treelite.frontend.load_lightgbm_model(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return ForestInference(
        raft_handle=raft_handle,
        treelite_model=tl_model,
        layout=layout,
        default_chunk_size=default_chunk_size,
        align_bytes=align_bytes,
        precision=precision,
        device=device,
        device_id=device_id,
    )


def load_from_sklearn(
    skl_model: Any,
    *,
    device: str = "auto",
    layout: str = "depth_first",
    default_chunk_size: Optional[int] = None,
    align_bytes: Optional[int] = None,
    precision: Optional[str] = None,
    device_id: Optional[int] = None,
    raft_handle: Optional[RaftHandle] = None,
) -> ForestInference:
    """Load a Scikit-Learn forest model to cuForest

    Parameters
    ----------
    skl_model
        The Scikit-Learn forest model to load.
    device: {"auto", "gpu", "cpu"}, default="auto"
        Whether to use GPU or CPU for inferencing. If set to "auto", GPU will
        be selected if it is available.
    layout : {"breadth_first", "depth_first", "layered"}, default="depth_first"
        The in-memory layout to be used during inference for nodes of the
        forest model. This parameter is available purely for runtime
        optimization. For performance-critical applications, it is
        recommended that available layouts be tested with realistic batch
        sizes to determine the optimal value.
    default_chunk_size : int or None, default=None
        If set, predict calls without a specified chunk size will use
        this default value.
    align_bytes : int or None, default=None
        Pad each tree with empty nodes until its in-memory size is a multiple
        of the given value. If None, use 0 for GPU and 64 for CPU.
    precision : {"single", "double", None}, default="single"
        Use the given floating point precision for evaluating the model. If
        None, use the native precision of the model. Note that
        single-precision execution is substantially faster than
        double-precision execution, so double-precision is recommended
        only for models trained and double precision and when exact
        conformance between results from cuForest and the original training
        framework is of paramount importance.
    device_id : int or None, default=None
        For GPU execution, the device on which to load and execute this
        model. For CPU execution, this value is currently ignored.
    raft_handle : pylibraft.common.handle or None
        For GPU execution, the RAFT handle containing the stream or stream
        pool to use during loading and inference. If not given, a new
        handle will be constructed.
    """
    tl_model = treelite.sklearn.import_model(skl_model)
    return ForestInference(
        raft_handle=raft_handle,
        treelite_model=tl_model,
        layout=layout,
        default_chunk_size=default_chunk_size,
        align_bytes=align_bytes,
        precision=precision,
        device=device,
        device_id=device_id,
    )


def load_from_treelite_model(
    tl_model: treelite.Model,
    *,
    device: str = "auto",
    layout: str = "depth_first",
    default_chunk_size: Optional[int] = None,
    align_bytes: Optional[int] = None,
    precision: Optional[str] = None,
    device_id: Optional[int] = None,
    raft_handle: Optional[RaftHandle] = None,
) -> ForestInference:
    """Load a Treelite forest model to cuForest

    Parameters
    ----------
    tl_model :
        The Treelite model to load.
    device: {"auto", "gpu", "cpu"}, default="auto"
        Whether to use GPU or CPU for inferencing. If set to "auto", GPU will
        be selected if it is available.
    layout : {"breadth_first", "depth_first", "layered"}, default="depth_first"
        The in-memory layout to be used during inference for nodes of the
        forest model. This parameter is available purely for runtime
        optimization. For performance-critical applications, it is
        recommended that available layouts be tested with realistic batch
        sizes to determine the optimal value.
    default_chunk_size : int or None, default=None
        If set, predict calls without a specified chunk size will use
        this default value.
    align_bytes : int or None, default=None
        Pad each tree with empty nodes until its in-memory size is a multiple
        of the given value. If None, use 0 for GPU and 64 for CPU.
    precision : {"single", "double", None}, default="single"
        Use the given floating point precision for evaluating the model. If
        None, use the native precision of the model. Note that
        single-precision execution is substantially faster than
        double-precision execution, so double-precision is recommended
        only for models trained and double precision and when exact
        conformance between results from cuForest and the original training
        framework is of paramount importance.
    device_id : int or None, default=None
        For GPU execution, the device on which to load and execute this
        model. For CPU execution, this value is currently ignored.
    raft_handle : pylibraft.common.handle or None
        For GPU execution, the RAFT handle containing the stream or stream
        pool to use during loading and inference. If not given, a new
        handle will be constructed.
    """
    return ForestInference(
        raft_handle=raft_handle,
        treelite_model=tl_model,
        layout=layout,
        default_chunk_size=default_chunk_size,
        align_bytes=align_bytes,
        precision=precision,
        device=device,
        device_id=device_id,
    )
