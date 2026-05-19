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
from nvforest._handle import Handle
from nvforest._version import __git_commit__, __version__

__all__ = [
    "CPUForestInferenceClassifier",
    "CPUForestInferenceRegressor",
    "GPUForestInferenceClassifier",
    "GPUForestInferenceRegressor",
    "Handle",
    "load_model",
    "load_from_sklearn",
    "load_from_treelite_model",
    "__git_commit__",
    "__version__",
]
