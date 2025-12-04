#
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

cdef extern from "cuforest/postproc_ops.hpp" namespace "cuforest" nogil:
    cdef enum row_op:
        row_disable "cuforest::row_op::disable",
        softmax "cuforest::row_op::softmax",
        max_index "cuforest::row_op::max_index"
    cdef enum element_op:
        elem_disable "cuforest::element_op::disable",
        signed_square "cuforest::element_op::signed_square",
        hinge "cuforest::element_op::hinge",
        sigmoid "cuforest::element_op::sigmoid",
        exponential "cuforest::element_op::exponential",
        logarithm_one_plus_exp "cuforest::element_op::logarithm_one_plus_exp"
