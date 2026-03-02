# nvForest - Highly Optimized Decision Tree Inference

nvForest is a highly-optimized and lightweight [RAPIDS](https://rapids.ai/) library that enables fast inference for decision tree models on NVIDIA GPUs and CPUs. It does not train models; it runs inference on models trained elsewhere (e.g., XGBoost, LightGBM, scikit-learn, or cuML).

nvForest uses [Treelite](https://treelite.readthedocs.io/) as the common format for importing tree models. You can load a model from a file or from an in-memory scikit-learn or Treelite object, then run predictions with a scikit-learn-like API. Setting `device="auto"` lets you deploy the same script on machines with or without GPUs.

As an example, the following Python snippet loads an XGBoost model and runs inference on GPU:

```python
import nvforest

# Load XGBoost model for GPU inference
fm = nvforest.load_model("/path/to/xgboost_model.ubj", device="gpu",
                         model_type="xgboost_ubj")

# Run inference (X can be a NumPy array or CuPy array)
pred = fm.predict(X)
```

Load a scikit-learn random forest model and get class probabilities:

```python
import nvforest
from sklearn.ensemble import RandomForestClassifier

# Train with scikit-learn (or load a saved model)
skl_model = RandomForestClassifier(...)
skl_model.fit(X_train, y_train)

# Load into nvForest for fast GPU inference
fm = nvforest.load_from_sklearn(skl_model, device="gpu")
class_probs = fm.predict_proba(X)
```

For more examples and the full API, see the [Getting started](docs/source/getting_started.rst) guide and the [Python API documentation](docs/source/python_api.rst).

### Supported Models

| Source | Formats |
| --- | --- |
| **XGBoost** | UBJSON, JSON, legacy binary |
| **LightGBM** | Text (`.txt`) |
| **scikit-learn** | In-memory (RandomForest, ExtraTrees, GradientBoosting) |
| **cuML** | Via Treelite export |
| **Treelite** | Checkpoint / in-memory `treelite.Model` |

### Inference Modes

| Method | Description |
| --- | --- |
| `predict(X)` | Standard predictions (class labels or regression values) |
| `predict_proba(X)` | Class probabilities (classification only) |
| `apply(X)` | Leaf indices per tree |
| `predict_per_tree(X)` | Prediction from each tree in the ensemble |

You can tune performance with `layout` (e.g., `depth_first`, `breadth_first`) and `chunk_size`; use `fm.optimize()` to auto-tune.

---

## Installation

See [the RAPIDS Release Selector](https://docs.rapids.ai/install#selector) for the command line to install either nightly or official release nvForest packages via conda, pip, or Docker.

## Build/Install from Source

See the build [guide](BUILD.md).

## Contributing

We welcome contributions. For guidelines and how to get started, see the [RAPIDS contributing guide](https://docs.rapids.ai/contributing).

## Contact

Find out more on the [RAPIDS site](https://rapids.ai/community.html).

## Open GPU Data Science

The RAPIDS suite of open source software libraries aims to enable execution of end-to-end data science and analytics pipelines entirely on GPUs. It relies on NVIDIA® CUDA® primitives for low-level compute optimization, exposing GPU parallelism and high-bandwidth memory through user-friendly Python interfaces.
