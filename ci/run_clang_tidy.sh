#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking run_clang_tidy.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

source rapids-configure-sccache

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Generate clang-tidy environment"
rapids-dependency-file-generator \
  --output conda \
  --file-key clang_tidy \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n clang_tidy

# Temporarily allow unbound variables for conda activation.
set +u
conda activate clang_tidy
set -u

rapids-print-env

LIBCUFOREST_BUILD_DIR=${LIBCUFOREST_BUILD_DIR:=${PWD}/cpp/build}

cmake \
    -S cpp \
    -B "${LIBCUFOREST_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_CUFOREST_TESTS=OFF \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

rapids-logger "Run clang-tidy"
python cpp/scripts/run-clang-tidy.py

