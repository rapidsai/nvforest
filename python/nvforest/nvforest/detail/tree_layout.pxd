#
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

cdef extern from "nvforest/tree_layout.hpp" namespace "nvforest" nogil:
    cdef enum class tree_layout:
        depth_first "nvforest::tree_layout::depth_first",
        breadth_first "nvforest::tree_layout::breadth_first",
        layered_children_together "nvforest::tree_layout::layered_children_together"
