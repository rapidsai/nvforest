from enum import Enum
from typing import Optional

import treelite
from cuda.bindings import runtime

from cuforest._base import ForestInferenceClassifier, ForestInferenceRegressor
from cuforest._handle import Handle
from cuforest._typing import DataType
from cuforest.detail.forest_inference import ForestInferenceImpl


# TaskType enum class from Treelite
class TaskTypeEnum(Enum):
    kBinaryClf = 0
    kRegressor = 1
    kMultiClf = 2
    kLearningToRank = 3
    kIsolationForest = 4


def infer_is_classifier(treelite_model: treelite.Model) -> bool:
    header = treelite_model.get_header_accessor()
    return header.get_field("task_type") in (
        TaskTypeEnum.kBinaryClf.value,
        TaskTypeEnum.kMultiClf.value,
    )


def detect_current_device(require: bool) -> Optional[int]:
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


def infer_device(
    device: str,
    device_id: Optional[int],
) -> tuple[str, int]:
    if device == "auto":
        # Auto mode: Use GPU if available; use CPU otherwise.
        device_id = detect_current_device(require=False)
        device = "cpu" if device_id is None else "gpu"
    elif device == "gpu" and device_id is None:
        # If no device ID is explicitly given, use the currently
        # active device
        device_id = detect_current_device(require=True)
    assert device in ("gpu", "cpu")
    return device, device_id


class ClassifierMixin:
    def get_class_assignment(
        self,
        raw_out: DataType,
        threshold: Optional[float] = None,
    ) -> DataType:
        if self.forest.row_postprocessing == "max_index":
            return raw_out[:, 0]
        if len(raw_out.shape) < 2 or raw_out.shape[1] == 1:
            threshold = 0.5 if threshold is None else threshold
            result = (raw_out > threshold).astype("int")
        else:
            result = raw_out.argmax(axis=1)
        return result


class CPUForestInferenceClassifier(ForestInferenceClassifier, ClassifierMixin):
    def __init__(
        self,
        *,
        treelite_model: treelite.Model,
        handle: Optional[Handle] = None,
        layout: str = "depth_first",
        default_chunk_size: Optional[int] = None,
        align_bytes: Optional[int] = None,
        precision: Optional[str] = None,
    ):
        if not infer_is_classifier(treelite_model):
            raise ValueError(f"treelite_model must be a classifier.")
        self.forest = ForestInferenceImpl(
            treelite_model=treelite_model,
            device="cpu",
            device_id=-1,
            handle=handle,
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
            precision=precision,
        )

    def predict(
        self,
        X: DataType,
        *,
        chunk_size: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> DataType:
        raw_out = self.forest.predict(X, chunk_size=chunk_size)
        return self.get_class_assignment(raw_out, threshold=threshold)

    def predict_proba(
        self,
        X: DataType,
        *,
        chunk_size: Optional[int] = None,
    ) -> DataType:
        return self.forest.predict(X, chunk_size=chunk_size)

    def predict_per_tree(
        self,
        X: DataType,
        *,
        chunk_size: Optional[int] = None,
    ) -> DataType:
        return self.forest.predict_per_tree(X, chunk_size=chunk_size)

    def apply(
        self,
        X: DataType,
        *,
        chunk_size: Optional[int] = None,
    ) -> DataType:
        return self.forest.apply(X, chunk_size=chunk_size)

    @property
    def num_outputs(self) -> int:
        return self.forest.num_outputs

    @property
    def num_trees(self) -> int:
        return self.forest.num_trees

    @property
    def layout(self) -> str:
        return self.forest.layout

    @property
    def align_bytes(self) -> Optional[int]:
        return self.forest.align_bytes

    @property
    def precision(self) -> Optional[str]:
        return self.forest.precision


class CPUForestInferenceRegressor(ForestInferenceRegressor):
    def __init__(
        self,
        *,
        treelite_model: treelite.Model,
        handle: Optional[Handle] = None,
        layout: str = "depth_first",
        default_chunk_size: Optional[int] = None,
        align_bytes: Optional[int] = None,
        precision: Optional[str] = None,
    ):
        if infer_is_classifier(treelite_model):
            raise ValueError(f"treelite_model must be a regressor.")
        self.forest = ForestInferenceImpl(
            treelite_model=treelite_model,
            device="cpu",
            device_id=-1,
            handle=handle,
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
            precision=precision,
        )

    def predict(
        self,
        X: DataType,
        *,
        chunk_size: Optional[int] = None,
    ) -> DataType:
        return self.forest.predict(X, chunk_size=chunk_size)

    def predict_per_tree(
        self,
        X: DataType,
        *,
        chunk_size: Optional[int] = None,
    ) -> DataType:
        return self.forest.predict_per_tree(X, chunk_size=chunk_size)

    def apply(
        self,
        X: DataType,
        *,
        chunk_size: Optional[int] = None,
    ) -> DataType:
        return self.forest.apply(X, chunk_size=chunk_size)

    @property
    def num_outputs(self) -> int:
        return self.forest.num_outputs

    @property
    def num_trees(self) -> int:
        return self.forest.num_trees

    @property
    def layout(self) -> str:
        return self.forest.layout

    @property
    def align_bytes(self) -> Optional[int]:
        return self.forest.align_bytes

    @property
    def precision(self) -> Optional[str]:
        return self.forest.precision


class GPUForestInferenceClassifier(ForestInferenceClassifier, ClassifierMixin):
    def __init__(
        self,
        *,
        treelite_model: treelite.Model,
        handle: Optional[Handle] = None,
        layout: str = "depth_first",
        default_chunk_size: Optional[int] = None,
        align_bytes: Optional[int] = None,
        precision: Optional[str] = None,
        device_id: int,
    ):
        if not infer_is_classifier(treelite_model):
            raise ValueError(f"treelite_model must be a classifier.")
        self.forest = ForestInferenceImpl(
            treelite_model=treelite_model,
            device="gpu",
            device_id=device_id,
            handle=handle,
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
            precision=precision,
        )

    def predict(
        self,
        X: DataType,
        *,
        chunk_size: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> DataType:
        raw_out = self.forest.predict(X, chunk_size=chunk_size)
        return self.get_class_assignment(raw_out, threshold=threshold)

    def predict_proba(
        self,
        X: DataType,
        *,
        chunk_size: Optional[int] = None,
    ) -> DataType:
        return self.forest.predict(X, chunk_size=chunk_size)

    def predict_per_tree(
        self,
        X: DataType,
        *,
        chunk_size: Optional[int] = None,
    ) -> DataType:
        return self.forest.predict_per_tree(X, chunk_size=chunk_size)

    def apply(
        self,
        X: DataType,
        *,
        chunk_size: Optional[int] = None,
    ) -> DataType:
        return self.forest.apply(X, chunk_size=chunk_size)

    @property
    def num_outputs(self) -> int:
        return self.forest.num_outputs

    @property
    def num_trees(self) -> int:
        return self.forest.num_trees

    @property
    def layout(self) -> str:
        return self.forest.layout

    @property
    def align_bytes(self) -> Optional[int]:
        return self.forest.align_bytes

    @property
    def precision(self) -> Optional[str]:
        return self.forest.precision


class GPUForestInferenceRegressor(ForestInferenceRegressor):
    def __init__(
        self,
        *,
        treelite_model: treelite.Model,
        handle: Optional[Handle] = None,
        layout: str = "depth_first",
        default_chunk_size: Optional[int] = None,
        align_bytes: Optional[int] = None,
        precision: Optional[str] = None,
        device_id: int,
    ):
        if infer_is_classifier(treelite_model):
            raise ValueError(f"treelite_model must be a regressor.")
        self.forest = ForestInferenceImpl(
            treelite_model=treelite_model,
            device="gpu",
            device_id=device_id,
            handle=handle,
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
            precision=precision,
        )

    def predict(
        self,
        X: DataType,
        *,
        chunk_size: Optional[int] = None,
    ) -> DataType:
        return self.forest.predict(X, chunk_size=chunk_size)

    def predict_per_tree(
        self,
        X: DataType,
        *,
        chunk_size: Optional[int] = None,
    ) -> DataType:
        return self.forest.predict_per_tree(X, chunk_size=chunk_size)

    def apply(
        self,
        X: DataType,
        *,
        chunk_size: Optional[int] = None,
    ) -> DataType:
        return self.forest.apply(X, chunk_size=chunk_size)

    @property
    def num_outputs(self) -> int:
        return self.forest.num_outputs

    @property
    def num_trees(self) -> int:
        return self.forest.num_trees

    @property
    def layout(self) -> str:
        return self.forest.layout

    @property
    def align_bytes(self) -> Optional[int]:
        return self.forest.align_bytes

    @property
    def precision(self) -> Optional[str]:
        return self.forest.precision
