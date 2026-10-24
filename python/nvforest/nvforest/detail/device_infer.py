# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from cuda.bindings import runtime


def detect_current_device(require: bool) -> Optional[int]:
    """
    Query the currently active GPU.

    Parameters
    ----------
    require:
        Whether to raise an exception when no GPU is available.

    Returns
    -------
    int or None
        ID of the currently active GPU device, or None if no GPU is available.
    """
    status, current_device_id = runtime.cudaGetDevice()
    if status != runtime.cudaError_t.cudaSuccess:
        if not require:
            return None
        _, name = runtime.cudaGetErrorName(status)
        _, msg = runtime.cudaGetErrorString(status)
        name, msg = name.decode("utf-8"), msg.decode("utf-8")
        raise RuntimeError(
            f"Failed to detect a GPU device. Diagnostic:\n    {name}: {msg}"
        )
    return current_device_id


def infer_device(
    device: str,
    device_id: Optional[int],
) -> tuple[str, int]:
    if device == "auto":
        # Auto mode: Use GPU if available; use CPU otherwise.
        device_id = detect_current_device(require=False)
        device = "cpu" if device_id is None else "gpu"
    elif device == "gpu" and device_id is None:
        # If no device ID is explicitly given, use the currently
        # active device
        device_id = detect_current_device(require=True)
    assert device in ("gpu", "cpu")
    return device, device_id
