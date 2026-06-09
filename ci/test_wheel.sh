#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

LIBNVFOREST_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libnvforest nvforest --cuda "$RAPIDS_CUDA_VERSION")")
NVFOREST_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python nvforest nvforest --stable --cuda "$RAPIDS_CUDA_VERSION")")
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
mkdir -p "${RAPIDS_TESTS_DIR}"

# generate constraints, the constraints will limit the version of the
# dependencies that can be installed later on when installing the wheel
rapids-generate-pip-constraints test_python "${PIP_CONSTRAINT}"

# Install just minimal dependencies first
rapids-pip-retry install \
  --prefer-binary \
  --constraint "${PIP_CONSTRAINT}" \
  "${LIBNVFOREST_WHEELHOUSE}"/libnvforest*.whl \
  "${NVFOREST_WHEELHOUSE}"/nvforest*.whl

# Try to import nvforest with just a minimal install"
rapids-logger "Importing nvforest with minimal dependencies"
python -c "import nvforest"

# notes:
#
#   * echo to expand wildcard before adding `[test]` requires for pip
#   * need to provide --constraint="${PIP_CONSTRAINT}" because that environment variable is
#     ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
  --prefer-binary \
  --constraint "${PIP_CONSTRAINT}" \
  "${LIBNVFOREST_WHEELHOUSE}"/libnvforest*.whl \
  "$(echo "${NVFOREST_WHEELHOUSE}"/nvforest*.whl)[test]"

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest nvforest"
timeout 1h python -m pytest \
  --cache-clear \
  --numprocesses=8 \
  --dist=worksteal \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-nvforest.xml" \
  python/nvforest/tests/

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
