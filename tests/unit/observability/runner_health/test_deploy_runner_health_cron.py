# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for runner health cron installation in deploy-runners.sh (OMN-11277)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DEPLOY_SCRIPT = Path(__file__).parents[4] / "scripts" / "deploy-runners.sh"


def _read_script() -> str:
    return DEPLOY_SCRIPT.read_text(encoding="utf-8")


@pytest.mark.unit
def test_deploy_script_exists() -> None:
    assert DEPLOY_SCRIPT.exists(), f"deploy-runners.sh not found at {DEPLOY_SCRIPT}"


@pytest.mark.unit
def test_install_health_cron_function_present() -> None:
    content = _read_script()
    assert "install_health_cron()" in content


@pytest.mark.unit
def test_health_cron_runs_every_three_minutes() -> None:
    content = _read_script()
    match = re.search(r"\*/3 \* \* \* \*.*cli_runner_health", content)
    assert match is not None, (
        "Cron schedule '*/3 * * * *' with cli_runner_health not found"
    )


@pytest.mark.unit
def test_health_cron_uses_emit_and_alert_flags() -> None:
    content = _read_script()
    match = re.search(r"cli_runner_health.*--emit.*--alert", content)
    assert match is not None, "cli_runner_health --emit --alert not found in cron line"


@pytest.mark.unit
def test_health_cron_sources_env_file() -> None:
    content = _read_script()
    assert (
        "~/.omnibase/.env" in content
        or r"\${HOME}/.omnibase/.env" in content
        or ". ~/.omnibase/.env" in content
    )


@pytest.mark.unit
def test_health_cron_logs_to_tmp() -> None:
    content = _read_script()
    assert "/tmp/runner-health.log" in content  # noqa: S108


@pytest.mark.unit
def test_health_cron_is_idempotent() -> None:
    content = _read_script()
    assert "grep -v" in content and "runner-health-check" in content, (
        "Idempotent installation using marker grep -v 'runner-health-check' not found"
    )


@pytest.mark.unit
def test_install_health_cron_called_in_deploy_with_retry() -> None:
    content = _read_script()
    in_retry_fn = re.search(
        r"deploy_with_retry\(\).*?install_health_cron",
        content,
        re.DOTALL,
    )
    assert in_retry_fn is not None, (
        "install_health_cron not called in deploy_with_retry()"
    )
