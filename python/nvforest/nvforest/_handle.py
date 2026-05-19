#
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import warnings

from pylibraft.common.handle import DeviceResources as RaftDeviceResources

DeviceResources = RaftDeviceResources

__all__ = ["DeviceResources", "Handle"]


def __getattr__(name):
    if name == "Handle":
        warnings.warn(
            "nvforest.Handle was renamed to nvforest.DeviceResources in 26.06 "
            "and will be removed in 26.08.",
            FutureWarning,
            stacklevel=2,
        )
        return DeviceResources
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
