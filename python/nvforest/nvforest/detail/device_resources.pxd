#
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

from libcpp.memory cimport unique_ptr

from nvforest.detail.raft_proto.cuda_stream cimport (
    cuda_stream as raft_proto_stream_t,
)


cdef extern from "nvforest/device_resources.hpp" namespace "nvforest" nogil:
    cdef cppclass device_resources:
        device_resources() except +
        raft_proto_stream_t get_next_usable_stream() except +
        void synchronize() except +

cdef class DeviceResources:
    cdef unique_ptr[device_resources] c_obj
