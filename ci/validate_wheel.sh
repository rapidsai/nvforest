#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

package_dir=$1
wheel_dir_relative_path=$2

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"

cd "${package_dir}"

rapids-logger "validate packages with 'pydistcheck'"

PYDISTCHECK_ARGS=(
    --inspect
)

if [[ "${package_dir}" == "python/libcuforest" ]]; then
    if [[ "${RAPIDS_CUDA_MAJOR}" == "12" ]]; then
        PYDISTCHECK_ARGS+=(
            --max-allowed-size-compressed '90M'
        )
    else
        PYDISTCHECK_ARGS+=(
            --max-allowed-size-compressed '40M'
        )
    fi
elif [[ "${package_dir}" != "python/cuforest" ]]; then
    rapids-echo-stderr "unrecognized package_dir: '${package_dir}'"
    exit 1
fi

pydistcheck \
    "${PYDISTCHECK_ARGS[@]}" \
    "$(echo "${wheel_dir_relative_path}"/*.whl)"

rapids-logger "validate packages with 'twine'"

twine check \
    --strict \
    "$(echo "${wheel_dir_relative_path}"/*.whl)"
