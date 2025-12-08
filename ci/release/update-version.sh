#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

## Usage
# bash update-version.sh <new_version>

set -euo pipefail

# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG=$1

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')
CURRENT_MAJOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[1]}')
CURRENT_MINOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[2]}')
CURRENT_PATCH=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[3]}')
CURRENT_SHORT_TAG=${CURRENT_MAJOR}.${CURRENT_MINOR}

# Get <major>.<minor> for next version
NEXT_MAJOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[2]}')
NEXT_PATCH=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[3]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}

NEXT_UCXX_SHORT_TAG="$(curl -sL https://version.gpuci.io/rapids/${NEXT_SHORT_TAG})"

echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for cross-platform differences
function sed_runner() {
    sed -i.bak ''"$1"'' $2 && rm -f ${2}.bak
}

# Update RAPIDS_BRANCH
sed_runner "s/branch-${CURRENT_SHORT_TAG}/branch-${NEXT_SHORT_TAG}/g" RAPIDS_BRANCH

# VERSION file
echo "${NEXT_FULL_TAG}" > VERSION

# Python version updates (pyproject.toml, version files)
for FILE in python/*/pyproject.toml; do
  sed_runner "s/version = \".*\"/version = \"${NEXT_FULL_TAG}\"/g" "${FILE}"
done

# __init__.py and _version.py files
find python -name "__init__.py" -o -name "_version.py" | while read -r FILE; do
  sed_runner "s/__version__ = \".*\"/__version__ = \"${NEXT_FULL_TAG}\"/g" "${FILE}"
done

# VERSION files in Python packages
find python -name "VERSION" | while read -r FILE; do
  echo "${NEXT_FULL_TAG}" > "${FILE}"
done

# dependencies.yaml
sed_runner "s/cuforest==.*/cuforest==${NEXT_SHORT_TAG}.*,>=0.0.0a0/g" dependencies.yaml
sed_runner "s/libcuforest==.*/libcuforest==${NEXT_SHORT_TAG}.*,>=0.0.0a0/g" dependencies.yaml
sed_runner "s/libcuforest-tests==.*/libcuforest-tests==${NEXT_SHORT_TAG}.*,>=0.0.0a0/g" dependencies.yaml

# RAPIDS dependencies
for DEP in rmm librmm pylibraft libraft libraft-headers; do
  sed_runner "s/${DEP}==.*/${DEP}==${NEXT_SHORT_TAG}.*,>=0.0.0a0/g" dependencies.yaml
done

echo "Version update complete"

