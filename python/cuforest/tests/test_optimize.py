# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for the optimize() method and mutable layout/chunk_size functionality.

import numpy as np
import pytest

# Import XGBoost before scikit-learn to work around a libgomp bug
# See https://github.com/dmlc/xgboost/issues/7110
xgb = pytest.importorskip("xgboost")

import cupy as cp  # noqa: E402
from sklearn.datasets import make_classification, make_regression  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

import cuforest  # noqa: E402


def _get_numpy_array(x):
    if isinstance(x, cp.ndarray):
        return x.get()
    return x


def _simulate_data(
    m,
    n,
    k=2,
    n_informative="auto",
    random_state=None,
    classification=True,
    bias=0.0,
):
    if n_informative == "auto":
        n_informative = n // 5
    if classification:
        features, labels = make_classification(
            n_samples=m,
            n_features=n,
            n_informative=n_informative,
            n_redundant=n - n_informative,
            n_classes=k,
            random_state=random_state,
        )
    else:
        features, labels = make_regression(
            n_samples=m,
            n_features=n,
            n_informative=n_informative,
            n_targets=1,
            bias=bias,
            random_state=random_state,
        )
    return (
        np.c_[features].astype(np.float32),
        np.c_[labels].astype(np.float32).flatten(),
    )


def _build_and_save_xgboost(
    model_path,
    X_train,
    y_train,
    classification=True,
    num_rounds=5,
    n_classes=2,
    xgboost_params=None,
):
    """Train small xgboost classifier and saves it to model_path"""
    dtrain = xgb.DMatrix(X_train, label=y_train)

    params = {"eval_metric": "error", "max_depth": 25, "device": "cuda"}

    if classification:
        if n_classes == 2:
            params["objective"] = "binary:logistic"
        else:
            params["num_class"] = n_classes
            params["objective"] = "multi:softprob"
    else:
        params["objective"] = "reg:squarederror"
        params["base_score"] = 0.0

    xgboost_params = {} if xgboost_params is None else xgboost_params
    params.update(xgboost_params)
    bst = xgb.train(params, dtrain, num_rounds)
    bst.save_model(model_path)
    return bst


# absolute tolerance for cuForest predict_proba
# False is binary classification, True is multiclass
proba_atol = {False: 3e-7, True: 3e-6}


@pytest.fixture(scope="module")
def small_classifier_model(tmp_path_factory):
    """Create a small classifier model for testing."""
    X, y = _simulate_data(500, 10, random_state=43210, classification=True)
    model_path = str(tmp_path_factory.mktemp("models") / "small_class.ubj")
    bst = _build_and_save_xgboost(model_path, X, y)
    dtrain = xgb.DMatrix(X, label=y)
    xgb_preds = bst.predict(dtrain)
    return model_path, X, xgb_preds


@pytest.mark.parametrize("device", ("cpu", "gpu"))
def test_optimize_classifier(device, small_classifier_model):
    """Test that optimize() runs and sets layout and default_chunk_size."""
    model_path, X, xgb_preds = small_classifier_model
    fm = cuforest.load_model(model_path, device=device)

    # Run optimization with a short timeout
    fm.optimize(data=X, timeout=0.1)

    # Verify that layout is set to a valid value
    assert fm.layout in ("depth_first", "breadth_first", "layered")

    # Verify that default_chunk_size is set to a power of 2
    assert fm.default_chunk_size is not None
    assert fm.default_chunk_size > 0
    assert (
        fm.default_chunk_size & (fm.default_chunk_size - 1)
    ) == 0  # power of 2

    # Verify predictions still work after optimization
    cuforest_proba = _get_numpy_array(fm.predict_proba(X))
    cuforest_proba = np.reshape(cuforest_proba, xgb_preds.shape)
    np.testing.assert_almost_equal(cuforest_proba, xgb_preds)


@pytest.mark.parametrize("device", ("cpu", "gpu"))
def test_optimize_regressor(device, tmp_path):
    """Test optimize() on a regression model."""
    n_rows = 500
    n_columns = 20

    X, y = _simulate_data(
        n_rows, n_columns, random_state=43210, classification=False
    )

    model_path = tmp_path / "xgb_reg.ubj"
    bst = _build_and_save_xgboost(
        model_path, X, y, classification=False, num_rounds=5
    )

    fm = cuforest.load_model(model_path, device=device)

    fm.optimize(data=X, timeout=0.1)

    # Verify that layout and chunk_size are set
    assert fm.layout in ("depth_first", "breadth_first", "layered")
    assert fm.default_chunk_size is not None
    assert fm.default_chunk_size > 0

    # Verify predictions still work
    dtest = xgb.DMatrix(X)
    xgb_preds = bst.predict(dtest)
    fil_preds = _get_numpy_array(fm.predict(X))
    fil_preds = np.reshape(fil_preds, xgb_preds.shape)
    np.testing.assert_almost_equal(fil_preds, xgb_preds, decimal=4)


@pytest.mark.parametrize("device", ("cpu", "gpu"))
def test_optimize_without_data(device, small_classifier_model):
    """Test that optimize() can generate random data when none is provided."""
    model_path, X, xgb_preds = small_classifier_model
    fm = cuforest.load_model(model_path, device=device)

    # Run optimization without providing data
    fm.optimize(batch_size=100, timeout=0.1, seed=42)

    # Verify that optimization set the values
    assert fm.layout in ("depth_first", "breadth_first", "layered")
    assert fm.default_chunk_size is not None

    # Verify predictions still work
    cuforest_proba = _get_numpy_array(fm.predict_proba(X))
    cuforest_proba = np.reshape(cuforest_proba, xgb_preds.shape)
    np.testing.assert_almost_equal(cuforest_proba, xgb_preds)


@pytest.mark.parametrize("device", ("cpu", "gpu"))
def test_optimize_with_predict_per_tree(device, small_classifier_model):
    """Test optimize() with a different predict method."""
    model_path, X, xgb_preds = small_classifier_model
    fm = cuforest.load_model(model_path, device=device)

    # Optimize for predict_per_tree method
    fm.optimize(data=X, timeout=0.1, predict_method="predict_per_tree")

    # Verify that optimization ran successfully
    assert fm.layout in ("depth_first", "breadth_first", "layered")
    assert fm.default_chunk_size is not None


@pytest.mark.parametrize("device", ("cpu", "gpu"))
def test_layout_setter(device, small_classifier_model):
    """Test that layout can be changed after model creation."""
    model_path, X, xgb_preds = small_classifier_model
    fm = cuforest.load_model(model_path, device=device, layout="depth_first")

    # Verify initial layout
    assert fm.layout == "depth_first"

    # Change layout
    fm.layout = "breadth_first"
    assert fm.layout == "breadth_first"

    # Verify predictions still work after layout change
    cuforest_proba = _get_numpy_array(fm.predict_proba(X))
    cuforest_proba = np.reshape(cuforest_proba, xgb_preds.shape)
    np.testing.assert_almost_equal(cuforest_proba, xgb_preds)

    # Change to layered
    fm.layout = "layered"
    assert fm.layout == "layered"

    # Verify predictions still work
    cuforest_proba = _get_numpy_array(fm.predict_proba(X))
    cuforest_proba = np.reshape(cuforest_proba, xgb_preds.shape)
    np.testing.assert_almost_equal(cuforest_proba, xgb_preds)


@pytest.mark.parametrize("device", ("cpu", "gpu"))
def test_default_chunk_size_setter(device, small_classifier_model):
    """Test that default_chunk_size can be set."""
    model_path, X, xgb_preds = small_classifier_model
    fm = cuforest.load_model(model_path, device=device)

    fm.default_chunk_size = 16
    assert fm.default_chunk_size == 16

    # Predictions should use the default chunk size
    cuforest_proba = _get_numpy_array(fm.predict_proba(X))
    cuforest_proba = np.reshape(cuforest_proba, xgb_preds.shape)
    np.testing.assert_almost_equal(cuforest_proba, xgb_preds)


def test_optimize_sklearn_classifier():
    """Test optimize() on a sklearn model loaded directly."""
    n_rows = 500
    n_columns = 20
    n_classes = 3

    X, y = _simulate_data(
        n_rows, n_columns, n_classes, random_state=43210, classification=True
    )

    skl_model = RandomForestClassifier(
        n_estimators=10, max_depth=5, random_state=42
    )
    skl_model.fit(X, y)

    fm = cuforest.load_from_sklearn(skl_model, device="cpu")

    fm.optimize(data=X, timeout=0.1)

    # Verify optimization worked
    assert fm.layout in ("depth_first", "breadth_first", "layered")
    assert fm.default_chunk_size is not None

    # Verify predictions match sklearn
    skl_proba = skl_model.predict_proba(X)
    cuforest_proba = _get_numpy_array(fm.predict_proba(X))
    cuforest_proba = np.reshape(cuforest_proba, skl_proba.shape)
    np.testing.assert_allclose(
        cuforest_proba, skl_proba, atol=proba_atol[True]
    )
