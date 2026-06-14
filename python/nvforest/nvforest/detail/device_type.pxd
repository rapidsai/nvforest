#
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
cdef extern from "nvforest/device_type.hpp" namespace "nvforest" nogil:
    cdef enum device_type:
        cpu "nvforest::device_type::cpu",
        gpu "nvforest::device_type::gpu"
