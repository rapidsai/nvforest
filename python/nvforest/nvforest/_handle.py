#
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

from pylibraft.common.handle import DeviceResources as RaftDeviceResources

# For now, nvforest.handle.Handle is an alias of pylibraft.common.handle.DeviceResources
Handle = RaftDeviceResources
