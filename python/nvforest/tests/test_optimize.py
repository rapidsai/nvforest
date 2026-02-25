# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for the optimize() method.

import numpy as np
import pytest

# Import XGBoost before scikit-learn to work around a libgomp bug
# See https://github.com/dmlc/xgboost/issues/7110
xgb = pytest.importorskip("xgboost")

import cupy as cp  # noqa: E402
from sklearn.datasets import make_classification, make_regression  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

import nvforest  # noqa: E402


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


# absolute tolerance for nvForest predict_proba
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
    """Test that optimize() returns a new instance with optimal settings."""
    model_path, X, xgb_preds = small_classifier_model
    fm = nvforest.load_model(model_path, device=device)

    # Run optimization with a short timeout - returns a new instance
    fm_opt = fm.optimize(data=X, timeout=0.1)

    # Verify fm_opt is a new instance
    assert fm_opt is not fm

    # Verify that layout is set to a valid value
    assert fm_opt.layout in ("depth_first", "breadth_first", "layered")

    # Verify that default_chunk_size is set to a power of 2
    assert fm_opt.default_chunk_size is not None
    assert fm_opt.default_chunk_size > 0
    assert (
        fm_opt.default_chunk_size & (fm_opt.default_chunk_size - 1)
    ) == 0  # power of 2

    # Verify predictions still work on optimized model
    nvforest_proba = _get_numpy_array(fm_opt.predict_proba(X))
    nvforest_proba = np.reshape(nvforest_proba, xgb_preds.shape)
    np.testing.assert_almost_equal(nvforest_proba, xgb_preds)


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

    fm = nvforest.load_model(model_path, device=device)

    # optimize() returns a new instance
    fm_opt = fm.optimize(data=X, timeout=0.1)

    # Verify fm_opt is a new instance
    assert fm_opt is not fm

    # Verify that layout and chunk_size are set
    assert fm_opt.layout in ("depth_first", "breadth_first", "layered")
    assert fm_opt.default_chunk_size is not None
    assert fm_opt.default_chunk_size > 0

    # Verify predictions still work
    dtest = xgb.DMatrix(X)
    xgb_preds = bst.predict(dtest)
    fil_preds = _get_numpy_array(fm_opt.predict(X))
    fil_preds = np.reshape(fil_preds, xgb_preds.shape)
    np.testing.assert_almost_equal(fil_preds, xgb_preds, decimal=4)


@pytest.mark.parametrize("device", ("cpu", "gpu"))
def test_optimize_without_data(device, small_classifier_model):
    """Test that optimize() can generate random data when none is provided."""
    model_path, X, xgb_preds = small_classifier_model
    fm = nvforest.load_model(model_path, device=device)

    # Run optimization without providing data - returns a new instance
    fm_opt = fm.optimize(batch_size=100, timeout=0.1, seed=42)

    # Verify that optimization set the values
    assert fm_opt.layout in ("depth_first", "breadth_first", "layered")
    assert fm_opt.default_chunk_size is not None

    # Verify predictions still work on the new instance
    nvforest_proba = _get_numpy_array(fm_opt.predict_proba(X))
    nvforest_proba = np.reshape(nvforest_proba, xgb_preds.shape)
    np.testing.assert_almost_equal(nvforest_proba, xgb_preds)


@pytest.mark.parametrize("device", ("cpu", "gpu"))
def test_optimize_with_predict_per_tree(device, small_classifier_model):
    """Test optimize() with a different predict method."""
    model_path, X, xgb_preds = small_classifier_model
    fm = nvforest.load_model(model_path, device=device)

    # Optimize for predict_per_tree method - returns a new instance
    fm_opt = fm.optimize(
        data=X, timeout=0.1, predict_method="predict_per_tree"
    )

    # Verify that optimization ran successfully
    assert fm_opt.layout in ("depth_first", "breadth_first", "layered")
    assert fm_opt.default_chunk_size is not None


@pytest.mark.parametrize("device", ("cpu", "gpu"))
def test_model_layout_immutable(device, small_classifier_model):
    """Test that layout is immutable (read-only property)."""
    model_path, X, xgb_preds = small_classifier_model
    fm = nvforest.load_model(model_path, device=device, layout="depth_first")

    # Verify initial layout
    assert fm.layout == "depth_first"

    # Attempting to set layout should raise AttributeError
    with pytest.raises(AttributeError):
        fm.layout = "breadth_first"


@pytest.mark.parametrize("device", ("cpu", "gpu"))
def test_load_with_different_layouts(device, small_classifier_model):
    """Test loading models with different layout settings."""
    model_path, X, xgb_preds = small_classifier_model

    # Load with each layout type
    for layout in ("depth_first", "breadth_first", "layered"):
        fm = nvforest.load_model(model_path, device=device, layout=layout)
        assert fm.layout == layout

        # Verify predictions work with each layout
        nvforest_proba = _get_numpy_array(fm.predict_proba(X))
        nvforest_proba = np.reshape(nvforest_proba, xgb_preds.shape)
        np.testing.assert_almost_equal(nvforest_proba, xgb_preds)


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

    fm = nvforest.load_from_sklearn(skl_model, device="cpu")

    # optimize() returns a new instance
    fm_opt = fm.optimize(data=X, timeout=0.1)

    # Verify fm_opt is a new instance
    assert fm_opt is not fm

    # Verify optimization worked
    assert fm_opt.layout in ("depth_first", "breadth_first", "layered")
    assert fm_opt.default_chunk_size is not None

    # Verify predictions match sklearn
    skl_proba = skl_model.predict_proba(X)
    nvforest_proba = _get_numpy_array(fm_opt.predict_proba(X))
    nvforest_proba = np.reshape(nvforest_proba, skl_proba.shape)
    np.testing.assert_allclose(
        nvforest_proba, skl_proba, atol=proba_atol[True]
    )


@pytest.mark.parametrize("device", ("cpu", "gpu"))
def test_original_model_unchanged_after_optimize(
    device, small_classifier_model
):
    """Test that the original model is unchanged after calling optimize()."""
    model_path, X, xgb_preds = small_classifier_model
    fm = nvforest.load_model(model_path, device=device, layout="depth_first")

    original_layout = fm.layout
    original_chunk_size = fm.default_chunk_size

    # optimize() returns a new instance
    fm_opt = fm.optimize(data=X, timeout=0.1)

    # Original model should be unchanged
    assert fm.layout == original_layout
    assert fm.default_chunk_size == original_chunk_size

    # Optimized model may have different settings
    assert fm_opt is not fm
