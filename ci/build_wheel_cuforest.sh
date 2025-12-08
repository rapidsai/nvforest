#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

package_name="cuforest"
package_dir="python/cuforest"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Download the libcuforest wheel built in the previous step and make it
# available for pip to find.
#
# Using env variable PIP_CONSTRAINT (initialized by 'rapids-init-pip') is necessary to ensure the constraints
# are used when creating the isolated build environment.
LIBCUFOREST_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcuforest_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
echo "libcuforest-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBCUFOREST_WHEELHOUSE}"/libcuforest_*.whl)" >> "${PIP_CONSTRAINT}"

EXCLUDE_ARGS=(
  --exclude "libcuforest++.so"
  --exclude "libraft.so"
  --exclude "libcublas.so.*"
  --exclude "libcublasLt.so.*"
  --exclude "libcufft.so.*"
  --exclude "libcurand.so.*"
  --exclude "libcusolver.so.*"
  --exclude "libcusparse.so.*"
  --exclude "libnvJitLink.so.*"
  --exclude "librapids_logger.so"
  --exclude "librmm.so"
)

export SKBUILD_CMAKE_ARGS="-DDISABLE_DEPRECATION_WARNINGS=ON"
./ci/build_wheel.sh "${package_name}" "${package_dir}"

# repair wheels and write to the location that artifact-uploading code expects to find them
python -m auditwheel repair \
    "${EXCLUDE_ARGS[@]}" \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    ${package_dir}/dist/*

./ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

