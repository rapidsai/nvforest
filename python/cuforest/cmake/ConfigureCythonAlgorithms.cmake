# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

function(add_module_gpu_default FILENAME)
  list(APPEND cython_sources ${FILENAME})
  set(cython_sources
    ${cython_sources}
    PARENT_SCOPE
  )
endfunction()
