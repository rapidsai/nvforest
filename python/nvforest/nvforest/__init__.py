#
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

# If libnvforest was installed as a wheel, we must request it to load the library symbols.
# Otherwise, we assume that the library was installed in a system path that ld can find.
try:
    import libnvforest
except ModuleNotFoundError:
    pass
else:
    libnvforest.load_library()
    del libnvforest

from nvforest._factory import (
    load_from_sklearn,
    load_from_treelite_model,
    load_model,
)
from nvforest._forest_inference import (
    CPUForestInferenceClassifier,
    CPUForestInferenceRegressor,
    GPUForestInferenceClassifier,
    GPUForestInferenceRegressor,
)
from nvforest._handle import DeviceResources
from nvforest._version import __git_commit__, __version__

__all__ = [
    "CPUForestInferenceClassifier",
    "CPUForestInferenceRegressor",
    "DeviceResources",
    "GPUForestInferenceClassifier",
    "GPUForestInferenceRegressor",
    "Handle",
    "load_model",
    "load_from_sklearn",
    "load_from_treelite_model",
    "__git_commit__",
    "__version__",
]


def __getattr__(name):
    if name == "Handle":
        import warnings

        warnings.warn(
            "nvforest.Handle was renamed to nvforest.DeviceResources in 26.06 "
            "and will be removed in 26.08.",
            FutureWarning,
            stacklevel=2,
        )
        return DeviceResources
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
