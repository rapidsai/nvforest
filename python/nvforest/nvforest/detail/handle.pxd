#
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

from pylibraft.common.handle cimport handle_t as raft_handle_t

from nvforest.detail.cuda_stream cimport cuda_stream as nvforest_stream_t


cdef extern from "nvforest/handle.hpp" namespace "nvforest" nogil:
    cdef cppclass handle_t:
        handle_t() except +
        handle_t(const raft_handle_t* handle_ptr) except +
        handle_t(const raft_handle_t& handle) except +
        nvforest_stream_t get_next_usable_stream() except +
        void synchronize() except+
