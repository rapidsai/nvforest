# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Optional

from cuforest._typing import DataType


class ForestInference(ABC):
    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @property
    @abstractmethod
    def num_outputs(self) -> int:
        pass

    @property
    @abstractmethod
    def num_trees(self) -> int:
        pass

    @property
    @abstractmethod
    def layout(self) -> str:
        pass

    @layout.setter
    @abstractmethod
    def layout(self, value: str):
        pass

    @property
    @abstractmethod
    def default_chunk_size(self) -> Optional[int]:
        pass

    @default_chunk_size.setter
    @abstractmethod
    def default_chunk_size(self, value: Optional[int]):
        pass

    @property
    @abstractmethod
    def align_bytes(self) -> Optional[int]:
        pass

    @property
    @abstractmethod
    def precision(self) -> Optional[str]:
        pass

    @property
    @abstractmethod
    def is_classifier(self) -> bool:
        pass

    @abstractmethod
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
        pass


class ForestInferenceClassifier(ForestInference):
    @property
    def is_classifier(self) -> bool:
        return True

    @abstractmethod
    def predict(
        self,
        X: DataType,
        *,
        chunk_size: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> DataType:
        """
        Predict the class for each row.

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
        pass

    @abstractmethod
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
        pass


class ForestInferenceRegressor(ForestInference):
    @property
    def is_classifier(self):
        return False

    @abstractmethod
    def predict(
        self,
        X: DataType,
        *,
        chunk_size: Optional[int] = None,
    ) -> DataType:
        """
        Predict the output for each row.

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
        pass
