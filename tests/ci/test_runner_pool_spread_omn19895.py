# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19895: the fleet probe fails when a required runner label sits on one host.

Operator rulings 2026-09-28T01:58:06Z and 01:58:17Z: a check must not rely on a
specific machine. A workflow that names no machine still relies on one when
every runner of its label is on one host. These tests run the probe step body
of ``.github/workflows/runner-pool-spread.yml`` over the repository's real
``config/runner_fleet.yaml`` and a runner-registry fixture.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "runner-pool-spread.yml"
STEP = "Every required runner label sits on at least two hosts"


def _step() -> dict[str, Any]:
    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    for step in doc["jobs"]["runner-pool-spread"]["steps"]:
        if step.get("name") == STEP:
            return cast("dict[str, Any]", step)
    raise AssertionError(f"step {STEP!r} not found")


def _runner(name: str, *labels: str, status: str = "online") -> dict[str, Any]:
    return {"name": name, "status": status, "labels": [{"name": x} for x in labels]}


def _run(
    tmp_path: Path, runners: list[dict[str, Any]]
) -> subprocess.CompletedProcess[str]:
    fixture = tmp_path / "runners.json"
    fixture.write_text(json.dumps(runners), encoding="utf-8")
    step = _step()
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "RUNNER_POOL_RUNNERS_FILE": str(fixture),
        **{k: str(v) for k, v in step["env"].items() if not str(v).startswith("${{")},
    }
    return subprocess.run(
        ["bash", "-c", step["run"]],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _spread_verify_and_plane() -> list[dict[str, Any]]:
    # Two declared hosts each for the verify and customer-plane labels, by the
    # runner-name prefixes config/runner_fleet.yaml declares.
    return [
        _runner("omnipc2-verify-runner-1", "self-hosted", "omnibase-verify"),
        _runner("omninode-mini-runner-1", "self-hosted", "omnibase-verify"),
        _runner("omninode-runner-7", "self-hosted", "omnibase-customer-plane"),
        _runner("omnipc2-verify-runner-2", "self-hosted", "omnibase-customer-plane"),
    ]


def test_runner_pool_spread_fails_when_every_ci_runner_shares_one_host(
    tmp_path: Path,
) -> None:
    runners = [
        _runner(f"omninode-runner-{n}", "self-hosted", "omnibase-ci")
        for n in range(1, 61)
    ] + _spread_verify_and_plane()
    result = _run(tmp_path, runners)
    assert result.returncode == 1, result.stdout
    assert "omnibase-ci" in result.stdout.split("::error::", 1)[1]


def test_runner_pool_spread_passes_when_ci_runners_span_two_hosts(
    tmp_path: Path,
) -> None:
    runners = (
        [
            _runner(f"omninode-runner-{n}", "self-hosted", "omnibase-ci")
            for n in range(1, 45)
        ]
        + [
            _runner(f"omnipc2-verify-runner-{n}", "self-hosted", "omnibase-ci")
            for n in range(3, 19)
        ]
        + _spread_verify_and_plane()
    )
    result = _run(tmp_path, runners)
    assert result.returncode == 0, result.stdout + result.stderr


def test_runner_pool_spread_counts_an_undeclared_runner_toward_no_host(
    tmp_path: Path,
) -> None:
    """An unknown host is not a second host."""
    runners = (
        [_runner("omninode-runner-1", "self-hosted", "omnibase-ci")]
        + [_runner("mystery-box-1", "self-hosted", "omnibase-ci")]
        + _spread_verify_and_plane()
    )
    result = _run(tmp_path, runners)
    assert result.returncode == 1, result.stdout
    assert "undeclared" in result.stdout


def test_runner_pool_spread_ignores_offline_runners(tmp_path: Path) -> None:
    runners = (
        [_runner("omninode-runner-1", "self-hosted", "omnibase-ci")]
        + [
            _runner(
                "omnipc2-verify-runner-4",
                "self-hosted",
                "omnibase-ci",
                status="offline",
            )
        ]
        + _spread_verify_and_plane()
    )
    result = _run(tmp_path, runners)
    assert result.returncode == 1, result.stdout


def test_runner_pool_spread_is_red_when_the_registry_cannot_be_read(
    tmp_path: Path,
) -> None:
    step = _step()
    result = subprocess.run(
        ["bash", "-c", step["run"]],
        cwd=REPO_ROOT,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "SPREAD_LABELS": "omnibase-ci",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "RED, not a skip" in result.stdout


def test_runner_pool_spread_runs_off_the_fleet_it_watches() -> None:
    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert doc["jobs"]["runner-pool-spread"]["runs-on"] == "ubuntu-latest"
    triggers = doc.get("on", doc.get(True))
    assert "schedule" in triggers and "workflow_dispatch" in triggers


def test_runner_pool_spread_reads_each_hosts_pools(tmp_path: Path) -> None:
    """A host row's `pools:` prefixes count toward that host.

    The live spread (2026-09-28): 44 omninode-runner on .201 and 16
    omnipc2-ci-runner on .202 for omnibase-ci, and a customer-plane runner on
    each of .201 and .202. Every one of those prefixes except omninode-runner is
    declared under a host's `pools:` list, so a probe that reads only the host
    row's own prefix sees one host and goes red on a spread fleet.
    """
    runners = (
        [
            _runner(f"omninode-runner-{n}", "self-hosted", "omnibase-ci")
            for n in range(1, 45)
        ]
        + [
            _runner(f"omnipc2-ci-runner-{n}", "self-hosted", "omnibase-ci")
            for n in range(1, 17)
        ]
        + [
            _runner("omninode-verify-runner-1", "self-hosted", "omnibase-verify"),
            _runner("omnipc2-verify-runner-1", "self-hosted", "omnibase-verify"),
            _runner(
                "omninode-customer-plane-runner-1",
                "self-hosted",
                "omnibase-customer-plane",
            ),
            _runner(
                "omnipc2-customer-plane-runner-1",
                "self-hosted",
                "omnibase-customer-plane",
            ),
        ]
    )
    result = _run(tmp_path, runners)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "undeclared" not in result.stdout, result.stdout
