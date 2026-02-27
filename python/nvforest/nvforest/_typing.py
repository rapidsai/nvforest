#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    import cupy

DataType = Union[np.ndarray, "cupy.ndarray"]
