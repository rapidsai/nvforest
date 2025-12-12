import pathlib
from typing import Any, Optional, Union

import treelite

from cuforest._base import ForestInference
from cuforest._forest_inference import (
    CPUForestInferenceClassifier,
    CPUForestInferenceRegressor,
    GPUForestInferenceClassifier,
    GPUForestInferenceRegressor,
    infer_device,
    infer_is_classifier,
)
from cuforest._handle import Handle


def get_forest_inference_class(device, is_classifier) -> type:
    match (device, is_classifier):
        case ("cpu", True):
            return CPUForestInferenceClassifier
        case ("cpu", False):
            return CPUForestInferenceRegressor
        case ("gpu", True):
            return GPUForestInferenceClassifier
        case ("gpu", False):
            return GPUForestInferenceRegressor
        case _:
            raise ValueError(f"Unknown device configuration: '{device}'")


def make_forest_inference_object(
    *,
    treelite_model: treelite.Model,
    device: str,
    device_id: Optional[int],
    handle: Optional[Handle],
    layout: str,
    default_chunk_size: Optional[int],
    align_bytes: Optional[int],
    precision: Optional[str],
) -> ForestInference:
    device, device_id = infer_device(device, device_id)
    is_classifier = infer_is_classifier(treelite_model)

    kwargs = dict(
        treelite_model=treelite_model,
        handle=handle,
        layout=layout,
        default_chunk_size=default_chunk_size,
        align_bytes=align_bytes,
        precision=precision,
    )
    if device == "gpu":
        kwargs["device_id"] = device_id

    return get_forest_inference_class(device, is_classifier)(**kwargs)


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
    handle: Optional[Handle] = None,
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
    precision : {"single", "double", None}, default=None
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
    handle : cuforest.Handle or None
        For GPU execution, the cuForest handle containing the stream or stream
        pool to use during loading and inference. If not given, a new
        handle will be constructed.
    """
    model_path = pathlib.Path(model_file)
    if not model_path.exists():
        raise FileNotFoundError(f"Error: Model file '{model_file}' not found.")
    if model_type is None:
        match model_path.suffix:
            case ".json":
                model_type = "xgboost_json"
            case ".ubj":
                model_type = "xgboost_ubj"
            case ".model":
                model_type = "xgboost"
            case ".txt":
                model_type = "lightgbm"
            case _:
                model_type = "treelite_checkpoint"
    match model_type:
        case "treelite_checkpoint":
            tl_model = treelite.frontend.Model.deserialize(model_path)
        case "xgboost_ubj":
            tl_model = treelite.frontend.load_xgboost_model(
                model_path, format_choice="ubjson"
            )
        case "xgboost_json":
            tl_model = treelite.frontend.load_xgboost_model(
                model_path, format_choice="json"
            )
        case "xgboost":
            tl_model = treelite.frontend.load_xgboost_model_legacy_binary(
                model_path
            )
        case "lightgbm":
            tl_model = treelite.frontend.load_lightgbm_model(model_path)
        case _:
            raise ValueError(f"Unknown model type: {model_type}")

    return make_forest_inference_object(
        treelite_model=tl_model,
        device=device,
        device_id=device_id,
        handle=handle,
        layout=layout,
        default_chunk_size=default_chunk_size,
        align_bytes=align_bytes,
        precision=precision,
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
    handle: Optional[Handle] = None,
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
    handle : cuforest.Handle or None
        For GPU execution, the cuForest handle containing the stream or stream
        pool to use during loading and inference. If not given, a new
        handle will be constructed.
    """
    tl_model = treelite.sklearn.import_model(skl_model)

    return make_forest_inference_object(
        treelite_model=tl_model,
        device=device,
        device_id=device_id,
        handle=handle,
        layout=layout,
        default_chunk_size=default_chunk_size,
        align_bytes=align_bytes,
        precision=precision,
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
    handle: Optional[Handle] = None,
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
    handle : cuforest.Handle or None
        For GPU execution, the cuForest handle containing the stream or stream
        pool to use during loading and inference. If not given, a new
        handle will be constructed.
    """
    return make_forest_inference_object(
        treelite_model=tl_model,
        device=device,
        device_id=device_id,
        handle=handle,
        layout=layout,
        default_chunk_size=default_chunk_size,
        align_bytes=align_bytes,
        precision=precision,
    )
