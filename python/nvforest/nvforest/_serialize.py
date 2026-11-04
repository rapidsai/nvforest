# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
from typing import Any, BinaryIO, Optional

from nvforest._base import ForestInference
from nvforest._forest_inference import (
    CPUForestInferenceClassifier,
    CPUForestInferenceRegressor,
    GPUForestInferenceClassifier,
    GPUForestInferenceRegressor,
    infer_device,
)
from nvforest._typing import StreamLike
from nvforest.detail.forest_inference import ForestInferenceImpl


def _get_model_parameters(inference_obj: ForestInference) -> dict[str, Any]:
    return {
        "is_classifier": inference_obj.is_classifier,
        "layout": inference_obj.forest._layout,
        "align_bytes": inference_obj.forest.align_bytes,
        "use_double_precision": inference_obj.forest._use_double_precision,
        "treelite_model_bytes": inference_obj.forest.treelite_model_bytes,
        "default_chunk_size": inference_obj.forest.default_chunk_size,
        "precision": inference_obj.forest.precision,
    }


def _build_inference_object(
    model_parameters: dict[str, Any],
    *,
    device: str = "auto",
    device_id: Optional[int] = None,
    stream: Optional[StreamLike] = None,
) -> ForestInference:
    device, device_id = infer_device(device, device_id)
    forest = ForestInferenceImpl(
        treelite_model=model_parameters["treelite_model_bytes"],
        device=device,
        device_id=device_id,
        stream=stream,
        layout=model_parameters["layout"],
        default_chunk_size=model_parameters["default_chunk_size"],
        align_bytes=model_parameters["align_bytes"],
        precision=model_parameters["precision"],
    )
    if device == "gpu":
        cls = (
            GPUForestInferenceClassifier
            if model_parameters["is_classifier"]
            else GPUForestInferenceRegressor
        )
    else:
        cls = (
            CPUForestInferenceClassifier
            if model_parameters["is_classifier"]
            else CPUForestInferenceRegressor
        )
    result = cls.__new__(cls)
    result.forest = forest
    return result


def serialize(inference_obj: ForestInference, file_obj: BinaryIO) -> None:
    model_parameters = _get_model_parameters(inference_obj)
    pickle.dump(model_parameters, file_obj)


def serialize_bytes(inference_obj: ForestInference) -> bytes:
    model_parameters = _get_model_parameters(inference_obj)
    return pickle.dumps(model_parameters)


def deserialize(
    file_obj: BinaryIO,
    *,
    device: str = "auto",
    device_id: Optional[int] = None,
    stream: Optional[StreamLike] = None,
) -> ForestInference:
    model_parameters = pickle.load(file_obj)
    return _build_inference_object(
        model_parameters,
        device=device,
        device_id=device_id,
        stream=stream,
    )


def deserialize_bytes(
    inference_obj_bytes: bytes,
    *,
    device: str = "auto",
    device_id: Optional[int] = None,
    stream: Optional[StreamLike] = None,
) -> ForestInference:
    model_parameters = pickle.loads(inference_obj_bytes)
    return _build_inference_object(
        model_parameters,
        device=device,
        device_id=device_id,
        stream=stream,
    )
