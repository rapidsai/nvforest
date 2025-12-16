#
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

# If libcuforest was installed as a wheel, we must request it to load the library symbols.
# Otherwise, we assume that the library was installed in a system path that ld can find.
try:
    import libcuforest
except ModuleNotFoundError:
    pass
else:
    libcuforest.load_library()
    del libcuforest

from cuforest._factory import (
    load_from_sklearn,
    load_from_treelite_model,
    load_model,
)
from cuforest._forest_inference import (
    CPUForestInferenceClassifier,
    CPUForestInferenceRegressor,
    GPUForestInferenceClassifier,
    GPUForestInferenceRegressor,
)
from cuforest._handle import Handle
from cuforest._version import __git_commit__, __version__

__all__ = [
    "CPUForestInferenceClassifier",
    "CPUForestInferenceRegressor",
    "GPUForestInferenceClassifier",
    "GPUForestInferenceRegressor",
    "Handle",
    "load_model",
    "load_from_sklearn",
    "load_from_treelite_model",
]
