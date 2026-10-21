# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: nightly integration selection must deselect the live LLM adapter suite.

The documentation generation suite is marked as integration by its conftest and
fails closed without LLM_DEEPSEEK_R1_URL (OMN-21147). This workflow configures no
endpoint, so its autouse fixture errored every night instead of exercising a live
adapter.

Falsifier: remove the new --deselect line from nightly-integration.yml and
test_integration_job_deselects_live_adapter_suite and
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


def _integration_job_pytest_args() -> list[str]:
    steps: list[dict[str, str]] = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "nightly-integration.yml").read_text(
            encoding="utf-8"
        )
    )["jobs"]["integration-tests"]["steps"]
    script = next(
        step["run"] for step in steps if step["name"] == "Run integration tests"
    )
    for line in script.replace("\\\n", " ").splitlines():
        tokens = shlex.split(line, comments=True)
        if tokens[:3] == ["uv", "run", "pytest"]:
            return tokens[3:]
    raise AssertionError("integration-tests must invoke uv run pytest")


def _workflow_collection_args(*, remove_live_adapter_deselect: bool) -> list[str]:
    args = iter(_integration_job_pytest_args())
    argv: list[str] = []
    for arg in args:
        if arg == "tests/integration/":
            argv.append("tests/integration/adapters/llm/")
        elif arg == "-m":
            marker = next(args)
            assert marker == "integration and not bastion"
            argv.extend([arg, marker])
        elif arg == "-v" or arg.startswith(
            ("--tb=", "--junitxml=", "--timeout=", "--timeout-method=")
        ):
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


def test_integration_job_deselects_live_adapter_suite() -> None:
    args = _integration_job_pytest_args()
    assert ("--deselect", LIVE_ADAPTER_SUITE) in pairwise(args)


def test_integration_job_keeps_existing_deselects() -> None:
    args = _integration_job_pytest_args()
    for suite in (
        "tests/integration/runtime/test_delegation_dispatch_port_consumer_kwarg_parity.py",
        "tests/integration/registration/e2e/test_two_way_registration_e2e.py",
    ):
        assert ("--deselect", suite) in pairwise(args)


def test_live_adapter_suite_is_integration_and_fails_closed() -> None:
    source = (REPO_ROOT / LIVE_ADAPTER_SUITE).read_text(encoding="utf-8")
    assert "pytest.mark.slow" in source
    conftest = (REPO_ROOT / "tests/integration/conftest.py").read_text(encoding="utf-8")
    assert "integration_marker = pytest.mark.integration" in conftest
    assert "item.add_marker(integration_marker)" in conftest
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
