# nvForest - Highly Optimized Decision Tree Inference

nvForest is a highly-optimized and lightweight library that enables fast inference for decision tree models on NVIDIA GPUs and CPUs. It does not train models; it runs inference on models trained elsewhere (e.g., XGBoost, LightGBM, scikit-learn, or cuML).

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

By default, nvForest creates a new
[CUDA stream](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-streams)
for the model and retains it. To use a custom stream, pass in a :external+cuda-python:py:class:`cuda.core.Stream` or
other stream-like objects (*):

(*) A stream-like object is an object that exposes a method named `__cuda_stream__`. See [Stream Protocol](
https://nvidia.github.io/cuda-python/cuda-core/latest/interoperability.html#cuda-stream-protocol) for more details.

```python
from cuda.core import Device

device = Device(0)
device.set_current()
stream = device.create_stream()

fm = nvforest.load_model(
    "/path/to/xgboost_model.ubj",
    device="gpu",
    device_id=0,
    stream=stream,
)
pred = fm.predict(X)  # Note: automatically syncs the stream
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

You can install nvForest using Pip or Conda.

```console
# Using Pip: need a suffix corresponding to your CUDA version, e.g. for CUDA 13:
$ pip install nvforest-cu13
```

```console
# Using Conda: need to specify the rapidsai channel
$ conda install -c rapidsai -c conda-forge nvforest
```

### System Requirements

Please see the [Installation Guide](https://docs.nvidia.com/datascience/install/#system-requirements)
for NVIDIA CUDA-X libraries for data science for information about supported operating systems,
GPU drivers, and CUDA versions.

## Build/Install from Source

See the build [guide](BUILD.md).

## Contact

Find out more: [NVIDIA CUDA-X for Data Science](https://developer.nvidia.com/topics/ai/data-science/cuda-x-for-data-science)

## NVIDIA CUDA-X Libraries for Data Science

The NVIDIA CUDA-X libraries for data science aim to enable execution of end-to-end data science and analytics pipelines entirely on GPUs. It relies on NVIDIA® CUDA® primitives for low-level compute optimization, exposing GPU parallelism and high-bandwidth memory through user-friendly Python interfaces.
