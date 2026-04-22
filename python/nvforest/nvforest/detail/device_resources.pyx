#
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

cdef class DeviceResources:
    def __cinit__(self, c_obj=None):
        self.c_obj.reset(new device_resources())

    def __getstate__(self):
        return object()

    def __setstate__(self, state):
        self.c_obj.reset(new device_resources())

    def get_c_obj(self):
        return <size_t> self.c_obj.get()
