#
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

cdef class Handle:
    def __cinit__(self):
        self.c_obj.reset(new handle_t())

    def __getstate__(self):
        return object()

    def __setstate__(self, state):
        self.c_obj.reset(new handle_t())

    def getHandle(self):
        return <size_t> self.c_obj.get()
