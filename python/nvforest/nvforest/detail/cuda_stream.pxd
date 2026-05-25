#
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
cdef extern from "nvforest/cuda_stream.hpp" namespace "nvforest" nogil:
    cdef cppclass cuda_stream:
        pass
