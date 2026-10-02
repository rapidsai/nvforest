#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from typing import TYPE_CHECKING, Protocol, Union, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import cupy

DataType = Union[np.ndarray, "cupy.ndarray"]


@runtime_checkable
class StreamLike(Protocol):
    """Duck typing for all stream-like objects"""

    def __cuda_stream__(self) -> tuple[int, int]: ...
