#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pickle

import cupy as cp
import numpy as np
import pytest
from cuda.bindings import runtime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import nvforest


def _get_device_count():
    status, device_count = runtime.cudaGetDeviceCount()
    return device_count if status == runtime.cudaError_t.cudaSuccess else 0


device_count = _get_device_count()


def _make_sklearn_model(model_class):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 4)).astype(np.float32)
    if model_class is RandomForestClassifier:
        y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)
    else:
        y = X[:, 0] - 2 * X[:, 1]
    return (
        model_class(n_estimators=3, max_depth=3, random_state=0).fit(X, y),
        X,
    )


def _to_numpy(array):
    return cp.asnumpy(array) if isinstance(array, cp.ndarray) else array


def _predict(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return model.predict(X)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "gpu",
            marks=pytest.mark.skipif(
                device_count == 0, reason="Requires a GPU"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "model_class", [RandomForestClassifier, RandomForestRegressor]
)
def test_pickle_round_trip(device, model_class):
    skl_model, X = _make_sklearn_model(model_class)
    model = nvforest.load_from_sklearn(
        skl_model,
        device=device,
        layout="breadth_first",
        default_chunk_size=2,
        align_bytes=128,
        precision="single",
    )
    expected = _to_numpy(_predict(model, X))

    restored = pickle.loads(pickle.dumps(model))

    assert type(restored) is type(model)
    assert restored.layout == "breadth_first"
    assert restored.default_chunk_size == 2
    assert restored.align_bytes == 128
    assert restored.precision == "single"
    assert restored.forest.device == device
    np.testing.assert_array_equal(_to_numpy(_predict(restored, X)), expected)


@pytest.mark.skipif(device_count == 0, reason="Requires a GPU")
def test_pickle_falls_back_to_gpu_zero(monkeypatch):
    skl_model, _ = _make_sklearn_model(RandomForestClassifier)
    model = nvforest.load_from_sklearn(skl_model, device="gpu", device_id=0)
    model.forest.device_id = 1
    serialized = pickle.dumps(model)
    monkeypatch.setattr(
        runtime,
        "cudaGetDeviceCount",
        lambda: (runtime.cudaError_t.cudaSuccess, 1),
    )

    restored = pickle.loads(serialized)

    assert isinstance(restored, nvforest.GPUForestInferenceClassifier)
    assert restored.forest.device == "gpu"
    assert restored.device_id == 0


@pytest.mark.skipif(device_count == 0, reason="Requires a GPU")
def test_pickle_falls_back_to_cpu(monkeypatch):
    skl_model, X = _make_sklearn_model(RandomForestRegressor)
    model = nvforest.load_from_sklearn(skl_model, device="gpu", device_id=0)
    expected = _to_numpy(model.predict(X))
    serialized = pickle.dumps(model)
    no_device = (runtime.cudaError_t.cudaErrorNoDevice, None)
    monkeypatch.setattr(runtime, "cudaGetDeviceCount", lambda: no_device)
    monkeypatch.setattr(runtime, "cudaGetDevice", lambda: no_device)

    restored = pickle.loads(serialized)

    assert isinstance(restored, nvforest.CPUForestInferenceRegressor)
    assert restored.forest.device == "cpu"
    assert restored.device_id == -1
    np.testing.assert_allclose(restored.predict(X), expected)
