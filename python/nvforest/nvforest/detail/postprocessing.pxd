#
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

cdef extern from "nvforest/postproc_ops.hpp" namespace "nvforest" nogil:
    cdef enum row_op:
        row_disable "nvforest::row_op::disable",
        softmax "nvforest::row_op::softmax",
        max_index "nvforest::row_op::max_index"
    cdef enum element_op:
        elem_disable "nvforest::element_op::disable",
        signed_square "nvforest::element_op::signed_square",
        hinge "nvforest::element_op::hinge",
        sigmoid "nvforest::element_op::sigmoid",
        exponential "nvforest::element_op::exponential",
        logarithm_one_plus_exp "nvforest::element_op::logarithm_one_plus_exp"
