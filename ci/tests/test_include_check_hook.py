# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual include hook in a new cache with real C++ fixtures."""

import os
import shutil
import subprocess  # nosec B404 -- fixed local Git and pre-commit commands.
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "invalid_directory", [None, "cpp/include", "cpp/src", "cpp/tests"]
)
def test_include_hook_preserves_scope_without_installing_a_package(
    tmp_path, invalid_directory
):
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    hook = next(
        hook
        for repository in config["repos"]
        if repository["repo"] == "local"
        for hook in repository["hooks"]
        if hook["id"] == "include-check"
    )
    fixture = tmp_path / "source"
    fixture.mkdir()
    scripts = fixture / "cpp/scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(
        ROOT / "cpp/scripts/include_checker.py", scripts / "include_checker.py"
    )
    (fixture / "README.md").write_text(
        "Fixture trigger outside checked source directories.\n"
    )
    for directory in ("cpp/include", "cpp/src", "cpp/tests"):
        path = fixture / directory
        path.mkdir(exist_ok=True)
        include = (
            '#include "vector"'
            if invalid_directory == directory
            else "#include <vector>"
        )
        (path / "fixture.hpp").write_text(f"#pragma once\n{include}\n")
    configuration = fixture / ".pre-commit-config.yaml"
    configuration.write_text(
        yaml.safe_dump({"repos": [{"repo": "local", "hooks": [hook]}]})
    )
    environment = {
        "PATH": os.environ["PATH"],
        "PRE_COMMIT_HOME": str(tmp_path / "cache"),
    }
    git = shutil.which("git")
    pre_commit = shutil.which("pre-commit")
    if not git or not pre_commit:
        pytest.fail(
            "Git and pre-commit must be installed for the actual hook contract"
        )
    subprocess.run(  # nosec B603 -- resolved Git executable and fixed fixture argv.
        [git, "init", "-q", str(fixture)], env=environment, check=True
    )
    subprocess.run(  # nosec B603 -- stage only the isolated synthetic fixture tree.
        [git, "add", "--all"], cwd=fixture, env=environment, check=True
    )
    result = subprocess.run(  # nosec B603 -- resolved pre-commit executable, synthetic source tree.
        [pre_commit, "run", "include-check", "--files", "README.md"],
        cwd=fixture,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    output = result.stdout + result.stderr
    if invalid_directory is None:
        assert result.returncode == 0, output  # nosec B101 -- pytest assertion.
        assert "Passed" in output  # nosec B101 -- pytest assertion.
    else:
        assert result.returncode != 0, output  # nosec B101 -- pytest assertion.
        assert f"{invalid_directory}/fixture.hpp" in output  # nosec B101 -- pytest assertion.
        assert "use #include <...>" in output  # nosec B101 -- pytest assertion.
    # A fresh cache must execute the stdlib checker without invoking pip/setuptools.
    assert "Installing environment" not in output  # nosec B101 -- pytest assertion.
    assert not list((tmp_path / "cache").glob("**/py_env-*"))  # nosec B101 -- pytest assertion.
