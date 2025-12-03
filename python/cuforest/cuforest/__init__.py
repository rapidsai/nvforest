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

from cuforest._version import __git_commit__, __version__

from cuforest.forest_inference import ForestInference, load_model, load_from_sklearn, load_from_treelite_model

__all__ = ["ForestInference", "load_model", "load_from_sklearn", "load_from_treelite_model"]
