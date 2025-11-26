#
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

cdef extern from "cuforest/tree_layout.hpp" namespace "cuforest" nogil:
    cdef enum class tree_layout:
        depth_first "cuforest::tree_layout::depth_first",
        breadth_first "cuforest::tree_layout::breadth_first",
        layered_children_together "cuforest::tree_layout::layered_children_together"
