# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: nightly slow selection must deselect the live LLM adapter suite.

The documentation generation suite carries `slow` and fails closed without
LLM_DEEPSEEK_R1_URL (OMN-21147). Hosted runners configure no endpoint, so its
autouse fixture errored every night instead of exercising a live adapter.

Falsifier: remove the new --deselect line from nightly-tests.yml and
test_slow_job_deselects_live_adapter_suite and
test_workflow_selection_collects_no_live_adapter_test fail.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_ADAPTER_SUITE = (
    "tests/integration/adapters/llm/"
    "test_adapter_documentation_generation_integration.py"
)


def _slow_job_pytest_args() -> list[str]:
    steps: list[dict[str, str]] = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "nightly-tests.yml").read_text(
            encoding="utf-8"
        )
    )["jobs"]["slow-tests"]["steps"]
    script = next(
        step["run"] for step in steps if step["name"] == "Run slow and chaos tests"
    )
    for line in script.replace("\\\n", " ").splitlines():
        tokens = shlex.split(line, comments=True)
        if tokens[:3] == ["uv", "run", "pytest"]:
            return tokens[3:]
    raise AssertionError("slow-tests must invoke uv run pytest")


def _workflow_collection_args(*, remove_live_adapter_deselect: bool) -> list[str]:
    args = iter(_slow_job_pytest_args())
    argv: list[str] = []
    for arg in args:
        if arg == "tests/":
            argv.append("tests/integration/adapters/llm/")
        elif arg == "-m":
            assert next(args) == "$FILTER"
            argv.extend(["-m", "slow or chaos"])
        elif arg == "-n":
            next(args)
        elif arg == "-v" or arg.startswith("--junitxml="):
            continue
        elif arg == "--deselect":
            path = next(args)
            if remove_live_adapter_deselect and path == LIVE_ADAPTER_SUITE:
                continue
            argv.extend([arg, path])
        else:
            argv.append(arg)
    return argv


def _collect_selection(
    *, remove_live_adapter_deselect: bool
) -> subprocess.CompletedProcess[str]:
    argv = _workflow_collection_args(
        remove_live_adapter_deselect=remove_live_adapter_deselect
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "--no-cov",
            *argv,
        ],
        cwd=REPO_ROOT,
        env={
            key: value
            for key, value in os.environ.items()
            if key != "LLM_DEEPSEEK_R1_URL"
        },
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_slow_job_deselects_live_adapter_suite() -> None:
    args = _slow_job_pytest_args()
    assert ("--deselect", LIVE_ADAPTER_SUITE) in pairwise(args)


def test_slow_job_keeps_two_way_registration_deselect() -> None:
    args = _slow_job_pytest_args()
    assert (
        "--deselect",
        "tests/integration/registration/e2e/test_two_way_registration_e2e.py",
    ) in pairwise(args)


def test_live_adapter_suite_is_slow_and_fails_closed() -> None:
    source = (REPO_ROOT / LIVE_ADAPTER_SUITE).read_text(encoding="utf-8")
    assert "pytest.mark.slow" in source
    assert "Set LLM_DEEPSEEK_R1_URL to run the live adapter suite" in source


def test_workflow_selection_collects_no_live_adapter_test() -> None:
    result = _collect_selection(remove_live_adapter_deselect=False)
    assert result.returncode in (0, 5), result.stdout + result.stderr
    assert not any(
        "test_adapter_documentation_generation_integration.py::" in line
        for line in result.stdout.splitlines()
    ), result.stdout


def test_live_adapter_suite_is_selected_without_the_deselect() -> None:
    result = _collect_selection(remove_live_adapter_deselect=True)
    assert result.returncode in (0, 5), result.stdout + result.stderr
    assert any(
        "test_adapter_documentation_generation_integration.py::" in line
        for line in result.stdout.splitlines()
    ), result.stdout
