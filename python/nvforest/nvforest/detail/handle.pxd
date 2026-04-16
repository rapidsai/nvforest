#
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

from libcpp.memory cimport unique_ptr

from nvforest.detail.raft_proto.cuda_stream cimport (
    cuda_stream as raft_proto_stream_t,
)


cdef extern from "nvforest/handle.hpp" namespace "nvforest" nogil:
    cdef cppclass handle_t:
        handle_t() except +
        raft_proto_stream_t get_next_usable_stream() except +
        void synchronize() except+

cdef class Handle:
    cdef unique_ptr[handle_t] c_obj
