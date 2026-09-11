#
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from cuda.bindings.cyruntime cimport cudaStream_t


cdef extern from "nvforest/cuda_stream.hpp" namespace "nvforest" nogil:
    ctypedef cudaStream_t cuda_stream
