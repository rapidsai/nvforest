#
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

cdef extern from "nvforest/infer_kind.hpp" namespace "nvforest" nogil:
    cdef enum class infer_kind:
        default_kind "nvforest::infer_kind::default_kind"
        per_tree "nvforest::infer_kind::per_tree"
        leaf_id "nvforest::infer_kind::leaf_id"
