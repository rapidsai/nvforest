#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CUFOREST_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cuforest_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
LIBCUFOREST_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcuforest_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
mkdir -p "${RAPIDS_TESTS_DIR}"

# generate constraints, the constraints will limit the version of the
# dependencies that can be installed later on when installing the wheel
rapids-generate-pip-constraints test_python ./constraints.txt

# Install just minimal dependencies first
rapids-pip-retry install \
  "${LIBCUFOREST_WHEELHOUSE}"/libcuforest*.whl \
  "${CUFOREST_WHEELHOUSE}"/cuforest*.whl \
  --constraint ./constraints.txt \
  --constraint "${PIP_CONSTRAINT}"

# Try to import cuforest with just a minimal install"
rapids-logger "Importing cuforest with minimal dependencies"
python -c "import cuforest"

# notes:
#
#   * echo to expand wildcard before adding `[test]` requires for pip
#   * need to provide --constraint="${PIP_CONSTRAINT}" because that environment variable is
#     ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
   "${LIBCUFOREST_WHEELHOUSE}"/libcuforest*.whl \
  "$(echo "${CUFOREST_WHEELHOUSE}"/cuforest*.whl)[test]" \
  --constraint ./constraints.txt \
  --constraint "${PIP_CONSTRAINT}"

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest cuforest"
timeout 1h python -m pytest \
  --cache-clear \
  --numprocesses=8 \
  --dist=worksteal \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuforest.xml" \
  python/cuforest/tests/

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

