#
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
cdef extern from "nvforest/detail/raft_proto/cuda_stream.hpp" namespace "raft_proto" nogil:
    cdef cppclass cuda_stream:
        pass
