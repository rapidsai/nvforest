# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Optional

from cuda.core import Stream
from pylibraft.common.handle import Handle as RaftHandle

from nvforest._typing import StreamLike


def _get_stream_from_raft_handle(handle: RaftHandle) -> Stream:
    cdef int stream_ptr = handle.c_obj.get_stream()
    return Stream.from_handle(stream_ptr)


def _handle_deprecated_handle_arg(
    *,
    handle: Optional[RaftHandle] = None,
    stream: Optional[StreamLike] = None,
) -> Optional[StreamLike]:
    if handle is None:
        return stream

    if stream is not None:
        raise ValueError("Cannot set `stream` and `handle` at the same time.")
    warnings.warn(
        "`handle` parameter is deprecated and will be removed in 26.12. "
        "Please use `stream` instead",
        FutureWarning,
        stacklevel=2,
    )
    stream = _get_stream_from_raft_handle(handle)
    return stream
