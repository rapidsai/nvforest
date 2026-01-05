#
# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import pytest

# =============================================================================
# Pytest Hooks and Configuration
# =============================================================================


def pytest_addoption(parser):
    """Add custom command line options for pytest.

    This function adds three custom options to control test execution:
    - --run_stress: Run stress tests
    - --run_quality: Run quality tests
    - --run_unit: Run unit tests
    - --run_memleak: Run memleak tests
    """
    group = parser.getgroup("cuForest Custom Options")

    group.addoption(
        "--run_stress",
        action="store_true",
        default=False,
        help=(
            "Runs tests marked with 'stress'. These are the most intense "
            "tests that take the longest to run and are designed to stress "
            "the hardware's compute resources."
        ),
    )

    group.addoption(
        "--run_quality",
        action="store_true",
        default=False,
        help=(
            "Runs tests marked with 'quality'. These tests are more "
            "computationally intense than 'unit', but less than 'stress'"
        ),
    )

    group.addoption(
        "--run_unit",
        action="store_true",
        default=False,
        help=(
            "Runs tests marked with 'unit'. These are the quickest tests "
            "that are focused on accuracy and correctness."
        ),
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options.

    This function:
    1. Checks for hypothesis tests without examples
    2. Selectively skip tests based on selected categories (unit/quality/stress)
    """

    # Handle test categories (unit/quality/stress)
    should_run_quality = config.getoption("--run_quality")
    should_run_stress = config.getoption("--run_stress")

    # Run unit is implied if no --run_XXX is set
    should_run_unit = config.getoption("--run_unit") or not (
        should_run_quality or should_run_stress
    )

    # Mark the tests as "skip" if needed
    if not should_run_unit:
        skip_unit = pytest.mark.skip(
            reason="Unit tests run with --run_unit flag."
        )
        for item in items:
            if "unit" in item.keywords:
                item.add_marker(skip_unit)

    if not should_run_quality:
        skip_quality = pytest.mark.skip(
            reason="Quality tests run with --run_quality flag."
        )
        for item in items:
            if "quality" in item.keywords:
                item.add_marker(skip_quality)

    if not should_run_stress:
        skip_stress = pytest.mark.skip(
            reason="Stress tests run with --run_stress flag."
        )
        for item in items:
            if "stress" in item.keywords:
                item.add_marker(skip_stress)
