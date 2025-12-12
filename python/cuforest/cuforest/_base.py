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
