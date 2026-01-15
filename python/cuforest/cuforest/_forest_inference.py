#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import itertools
from enum import Enum
from time import perf_counter
from typing import Optional

import numpy as np
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
            f"Failed to detect a GPU device. Diagnostic:\n    {name}: {msg}"
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


class _AutoIterations:
    """Used to generate sequence of iterations (1, 2, 5, 10, 20, 50...) during
    optimization."""

    def __init__(self):
        self.invocations = 0
        self.sequence = (1, 2, 5)

    def next(self):
        result = (
            10 ** (self.invocations // len(self.sequence))
        ) * self.sequence[self.invocations % len(self.sequence)]
        self.invocations += 1
        return result


class OptimizeMixin:
    """Mixin class that provides the optimize method for ForestInference classes."""

    def _optimize(
        self,
        *,
        data=None,
        batch_size: int = 1024,
        unique_batches: int = 10,
        timeout: float = 0.2,
        predict_method: str = "predict",
        max_chunk_size: Optional[int] = None,
        seed: int = 0,
    ):
        """
        Find the optimal layout and chunk size for this model.

        The optimal value for layout and chunk size depends on the model,
        batch size, and available hardware. In order to get the most
        realistic performance distribution, example data can be provided. If
        it is not, random data will be generated based on the indicated batch
        size. After finding the optimal layout, the model will be reloaded if
        necessary. The optimal chunk size will be used to set the default chunk
        size used if none is passed to the predict call.

        Parameters
        ----------
        data
            Example data either of shape unique_batches x batch_size x features
            or batch_size x features or None. If None, random data will be
            generated instead.
        batch_size : int
            If example data is not provided, random data with this many rows
            per batch will be used.
        unique_batches : int
            The number of unique batches to generate if random data are used.
            Increasing this number decreases the chance that the optimal
            configuration will be skewed by a single batch with unusual
            performance characteristics.
        timeout : float
            Time in seconds to target for optimization. The optimization loop
            will be repeatedly run a number of times increasing in the sequence
            1, 2, 5, 10, 20, 50, ... until the time taken is at least the given
            value. Note that for very large batch sizes and large models, the
            total elapsed time may exceed this timeout; it is a soft target for
            elapsed time. Setting the timeout to zero will run through the
            indicated number of unique batches exactly once. Defaults to 0.2s.
        predict_method : str
            If desired, optimization can occur over one of the prediction
            method variants (e.g. "predict_per_tree") rather than the
            default `predict` method. To do so, pass the name of the method
            here.
        max_chunk_size : int or None
            The maximum chunk size to explore during optimization. If not
            set, a value will be picked based on the current device type.
            Setting this to a lower value will reduce the optimization search
            time but may not result in optimal performance.
        seed : int
            The random seed used for generating example data if none is
            provided.
        """
        is_gpu = self.forest.device == "gpu"

        if data is None:
            rng = np.random.default_rng(seed)
            dtype = self.forest.get_dtype()
            data = rng.uniform(
                np.finfo(dtype).min / 2,
                np.finfo(dtype).max / 2,
                (unique_batches, batch_size, self.forest.num_features),
            ).astype(dtype)
            if is_gpu:
                import cupy as cp

                data = cp.asarray(data)
        else:
            if is_gpu:
                import cupy as cp

                data = cp.asarray(data)
            else:
                data = np.asarray(data)

        if len(data.shape) == 3:
            unique_batches, batch_size, features = data.shape
        else:
            unique_batches = 1
            batch_size, features = data.shape
            data = data[np.newaxis, ...]  # Add batch dimension

        if max_chunk_size is None:
            max_chunk_size = 32 if is_gpu else 512

        max_chunk_size = min(max_chunk_size, batch_size)

        infer = getattr(self, predict_method)

        optimal_layout = "depth_first"
        optimal_chunk_size = 1

        valid_layouts = ("depth_first", "breadth_first", "layered")
        valid_chunk_sizes = []
        chunk_size = 1
        while chunk_size <= max_chunk_size:
            valid_chunk_sizes.append(chunk_size)
            chunk_size *= 2

        all_params = list(itertools.product(valid_layouts, valid_chunk_sizes))
        auto_iterator = _AutoIterations()
        loop_start = perf_counter()

        while True:
            optimal_time = float("inf")
            iterations = auto_iterator.next()

            for layout, chunk_size in all_params:
                # Set layout (triggers model reload if changed)
                self.layout = layout

                # Warmup run
                infer(data[0], chunk_size=chunk_size)

                # Timed runs
                elapsed = float("inf")
                for _ in range(iterations):
                    start = perf_counter()
                    for iter_index in range(unique_batches):
                        infer(data[iter_index], chunk_size=chunk_size)
                    elapsed = min(elapsed, perf_counter() - start)

                if elapsed < optimal_time:
                    optimal_time = elapsed
                    optimal_layout = layout
                    optimal_chunk_size = chunk_size

            if perf_counter() - loop_start > timeout:
                break

        self.layout = optimal_layout
        self.default_chunk_size = optimal_chunk_size


class CPUForestInferenceClassifier(
    ForestInferenceClassifier, ClassifierMixin, OptimizeMixin
):
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
            raise ValueError("treelite_model must be a classifier.")
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

    def optimize(
        self,
        *,
        data=None,
        batch_size: int = 1024,
        unique_batches: int = 10,
        timeout: float = 0.2,
        predict_method: str = "predict",
        max_chunk_size: Optional[int] = None,
        seed: int = 0,
    ):
        self._optimize(
            data=data,
            batch_size=batch_size,
            unique_batches=unique_batches,
            timeout=timeout,
            predict_method=predict_method,
            max_chunk_size=max_chunk_size,
            seed=seed,
        )

    @property
    def num_outputs(self) -> int:
        return self.forest.num_outputs

    @property
    def num_trees(self) -> int:
        return self.forest.num_trees

    @property
    def align_bytes(self) -> Optional[int]:
        return self.forest.align_bytes

    @property
    def precision(self) -> Optional[str]:
        return self.forest.precision

    @property
    def default_chunk_size(self) -> Optional[int]:
        return self.forest.default_chunk_size

    @default_chunk_size.setter
    def default_chunk_size(self, value: Optional[int]):
        self.forest.default_chunk_size = value

    @property
    def layout(self) -> str:
        return self.forest.layout

    @layout.setter
    def layout(self, value: str):
        self.forest.layout = value


class CPUForestInferenceRegressor(ForestInferenceRegressor, OptimizeMixin):
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
            raise ValueError("treelite_model must be a regressor.")
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

    def optimize(
        self,
        *,
        data=None,
        batch_size: int = 1024,
        unique_batches: int = 10,
        timeout: float = 0.2,
        predict_method: str = "predict",
        max_chunk_size: Optional[int] = None,
        seed: int = 0,
    ):
        self._optimize(
            data=data,
            batch_size=batch_size,
            unique_batches=unique_batches,
            timeout=timeout,
            predict_method=predict_method,
            max_chunk_size=max_chunk_size,
            seed=seed,
        )

    @property
    def num_outputs(self) -> int:
        return self.forest.num_outputs

    @property
    def num_trees(self) -> int:
        return self.forest.num_trees

    @property
    def align_bytes(self) -> Optional[int]:
        return self.forest.align_bytes

    @property
    def precision(self) -> Optional[str]:
        return self.forest.precision

    @property
    def default_chunk_size(self) -> Optional[int]:
        return self.forest.default_chunk_size

    @default_chunk_size.setter
    def default_chunk_size(self, value: Optional[int]):
        self.forest.default_chunk_size = value

    @property
    def layout(self) -> str:
        return self.forest.layout

    @layout.setter
    def layout(self, value: str):
        self.forest.layout = value


class GPUForestInferenceClassifier(
    ForestInferenceClassifier, ClassifierMixin, OptimizeMixin
):
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
            raise ValueError("treelite_model must be a classifier.")
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

    def optimize(
        self,
        *,
        data=None,
        batch_size: int = 1024,
        unique_batches: int = 10,
        timeout: float = 0.2,
        predict_method: str = "predict",
        max_chunk_size: Optional[int] = None,
        seed: int = 0,
    ):
        self._optimize(
            data=data,
            batch_size=batch_size,
            unique_batches=unique_batches,
            timeout=timeout,
            predict_method=predict_method,
            max_chunk_size=max_chunk_size,
            seed=seed,
        )

    @property
    def num_outputs(self) -> int:
        return self.forest.num_outputs

    @property
    def num_trees(self) -> int:
        return self.forest.num_trees

    @property
    def align_bytes(self) -> Optional[int]:
        return self.forest.align_bytes

    @property
    def precision(self) -> Optional[str]:
        return self.forest.precision

    @property
    def default_chunk_size(self) -> Optional[int]:
        return self.forest.default_chunk_size

    @default_chunk_size.setter
    def default_chunk_size(self, value: Optional[int]):
        self.forest.default_chunk_size = value

    @property
    def layout(self) -> str:
        return self.forest.layout

    @layout.setter
    def layout(self, value: str):
        self.forest.layout = value


class GPUForestInferenceRegressor(ForestInferenceRegressor, OptimizeMixin):
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
            raise ValueError("treelite_model must be a regressor.")
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

    def optimize(
        self,
        *,
        data=None,
        batch_size: int = 1024,
        unique_batches: int = 10,
        timeout: float = 0.2,
        predict_method: str = "predict",
        max_chunk_size: Optional[int] = None,
        seed: int = 0,
    ):
        self._optimize(
            data=data,
            batch_size=batch_size,
            unique_batches=unique_batches,
            timeout=timeout,
            predict_method=predict_method,
            max_chunk_size=max_chunk_size,
            seed=seed,
        )

    @property
    def num_outputs(self) -> int:
        return self.forest.num_outputs

    @property
    def num_trees(self) -> int:
        return self.forest.num_trees

    @property
    def align_bytes(self) -> Optional[int]:
        return self.forest.align_bytes

    @property
    def precision(self) -> Optional[str]:
        return self.forest.precision

    @property
    def default_chunk_size(self) -> Optional[int]:
        return self.forest.default_chunk_size

    @default_chunk_size.setter
    def default_chunk_size(self, value: Optional[int]):
        self.forest.default_chunk_size = value

    @property
    def layout(self) -> str:
        return self.forest.layout

    @layout.setter
    def layout(self, value: str):
        self.forest.layout = value
