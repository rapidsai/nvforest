#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

python -m pytest \
  --cache-clear \
  "$@" \
  python/cuforest/tests/

