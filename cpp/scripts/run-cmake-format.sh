#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# This script is a wrapper for cmakelang that may be used with pre-commit. The
# wrapping is necessary because RAPIDS libraries split configuration for
# cmakelang linters between a local config file and a second config file that's
# shared across all of RAPIDS via rapids-cmake.

status=0
if [ -z ${CUFOREST_ROOT:+PLACEHOLDER} ]; then
    CUFOREST_BUILD_DIR=$(git rev-parse --show-toplevel 2>&1)/cpp/build
    status=$?
else
    CUFOREST_BUILD_DIR=${CUFOREST_ROOT}
fi

if ! [ ${status} -eq 0 ]; then
    if [[ ${CUFOREST_BUILD_DIR} == *"not a git repository"* ]]; then
        echo "This script must be run inside the cuforest repository, or the CUFOREST_ROOT environment variable must be set."
    else
        echo "Script failed with unknown error attempting to determine project root:"
        echo "${CUFOREST_BUILD_DIR}"
    fi
    exit 1
fi

DEFAULT_FORMAT_FILE_LOCATIONS=(
  "${CUFOREST_BUILD_DIR:-${HOME}}/_deps/rapids-cmake-src/cmake-format-rapids-cmake.json"
)

if [ -z ${RAPIDS_CMAKE_FORMAT_FILE:+PLACEHOLDER} ]; then
    for file_path in "${DEFAULT_FORMAT_FILE_LOCATIONS[@]}"; do
        if [ -f "${file_path}" ]; then
            RAPIDS_CMAKE_FORMAT_FILE=${file_path}
            break
        fi
    done
fi

if [ -z ${RAPIDS_CMAKE_FORMAT_FILE:+PLACEHOLDER} ]; then
  echo "The rapids-cmake cmake-format configuration file was not found at any of the default search locations: "
  echo ""
  ( IFS=$'\n'; echo "${DEFAULT_FORMAT_FILE_LOCATIONS[*]}" )
  echo ""
  echo "Try setting the environment variable RAPIDS_CMAKE_FORMAT_FILE to the path to the config file."
  exit 0
else
  echo "Using format file ${RAPIDS_CMAKE_FORMAT_FILE}"
fi

if [[ $1 == "cmake-format" ]]; then
  cmake-format -i --config-files cpp/cmake/config.json "${RAPIDS_CMAKE_FORMAT_FILE}" -- "${@:2}"
elif [[ $1 == "cmake-lint" ]]; then
  OUTPUT=$(cmake-lint --config-files cpp/cmake/config.json "${RAPIDS_CMAKE_FORMAT_FILE}" -- "${@:2}")
  status=$?

  if ! [ ${status} -eq 0 ]; then
    echo "${OUTPUT}"
  fi
  exit ${status}
fi

