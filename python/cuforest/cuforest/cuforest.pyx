#
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

from libc.stdint cimport uint32_t
from libcpp cimport bool

from cuforest.detail.raft_proto.handle cimport handle_t as raft_proto_handle_t
from cuforest.detail.raft_proto.optional cimport nullopt, optional
from cuforest.postprocessing cimport element_op, row_op

cdef extern from "cuforest/forest_model.hpp" namespace "cuforest" nogil:
    cdef cppclass forest_model:
        void predict[io_t](
            const raft_proto_handle_t&,
            io_t*,
            io_t*,
            size_t,
            raft_proto_device_t,
            raft_proto_device_t,
            infer_kind,
            optional[uint32_t]
        ) except +

        bool is_double_precision() except +
        size_t num_features() except +
        size_t num_outputs() except +
        size_t num_trees() except +
        bool has_vector_leaves() except +
        row_op row_postprocessing() except +
        element_op elem_postprocessing() except +

cdef extern from "cuforest/treelite_importer.hpp" namespace "cuforest" nogil:
    forest_model import_from_treelite_handle(
        TreeliteModelHandle,
        fil_tree_layout,
        uint32_t,
        optional[bool],
        raft_proto_device_t,
        int,
        raft_proto_stream_t
    ) except +
