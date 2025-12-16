#
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

cdef extern from "cuforest/infer_kind.hpp" namespace "cuforest" nogil:
    cdef enum class infer_kind:
        default_kind "cuforest::infer_kind::default_kind"
        per_tree "cuforest::infer_kind::per_tree"
        leaf_id "cuforest::infer_kind::leaf_id"
