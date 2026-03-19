#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
NVFOREST_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name wheel_python nvforest --stable --cuda "$RAPIDS_CUDA_VERSION")")
LIBNVFOREST_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libnvforest_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
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
