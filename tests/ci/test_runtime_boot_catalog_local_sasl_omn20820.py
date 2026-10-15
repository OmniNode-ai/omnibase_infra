# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The laptop-profile CI job gates on the broker's SASL requirement (OMN-20820).

``runtime-boot (mode=catalog-local)`` boots the ``local`` bundle, whose broker
now requires authentication. Two steps carry that gate:

* ``make check-sasl-local``: one anonymous probe, which must be refused, and one
  probe as the local superuser, which must be accepted. Removing
  ``redpanda-local-sasl`` from the bundle fails the render or the anonymous
  probe; stripping the credentials from the second probe fails it. The target's
  own behaviour is pinned in ``tests/unit/infra/test_catalog_local_profile.py``
  (``-k sasl``); here the workflow is pinned to call it.
* every one-shot of the render exited 0, ``redpanda-local-sasl`` among them,
  run here as shipped against a stub Docker.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO / ".github" / "workflows" / "reusable-runtime-boot.yml"
_JOB = "boot-catalog-local"
_SASL_STEP = "Broker requires SASL - anonymous client refused, local superuser accepted"
_ONE_SHOT_STEP = "Every laptop one-shot exited 0"
_PROJECT = "omnibase-infra-local"

_DOCKER_STUB = """#!/usr/bin/env bash
# Stub of `docker inspect --format ... <container>` for the one-shot step.
set -u
[ "$1" = "inspect" ] || { echo "stub docker: unexpected $*" >&2; exit 2; }
container="${!#}"
echo "$container" >> "$STUB_SEEN"
if [ "$container" = "$STUB_MISSING" ]; then
  echo "Error: No such object: $container" >&2
  exit 1
fi
if [ "$container" = "$STUB_FAILED" ]; then
  echo "exited exit=1"
else
  echo "exited exit=0"
fi
"""


def _steps() -> list[dict[str, Any]]:
    workflow: dict[str, Any] = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"][_JOB]["steps"]
    assert isinstance(steps, list)
    return steps


def _step(name: str) -> dict[str, Any]:
    matches = [s for s in _steps() if s.get("name") == name]
    assert len(matches) == 1, f"expected exactly one step named {name!r}"
    return matches[0]


def test_job_runs_the_sasl_check_after_the_boot_and_before_the_probes() -> None:
    names = [s.get("name") for s in _steps()]
    boot = names.index("Boot the laptop profile (make up-local)")
    check = names.index(_SASL_STEP)
    probe = names.index("Delegate-skill command topic has a live consumer group")
    assert boot < check < probe
    run = _step(_SASL_STEP)["run"]
    assert "make check-sasl-local" in run
    assert '"$LOCAL_ENV_FILE"' in run


def _run_one_shot_step(
    tmp_path: Path, *, failed: str = "", missing: str = ""
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(_DOCKER_STUB, encoding="utf-8")
    docker.chmod(docker.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    script = tmp_path / "step.sh"
    script.write_text(_step(_ONE_SHOT_STEP)["run"], encoding="utf-8")
    seen = tmp_path / "seen.txt"
    seen.write_text("", encoding="utf-8")
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "LOCAL_PROJECT": _PROJECT,
        "STUB_SEEN": str(seen),
        "STUB_FAILED": f"{_PROJECT}-{failed}" if failed else "",
        "STUB_MISSING": f"{_PROJECT}-{missing}" if missing else "",
    }
    result = subprocess.run(
        ["bash", "-e", str(script)],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    return result, seen.read_text(encoding="utf-8").split()


def test_one_shot_step_passes_and_covers_the_sasl_one_shot(tmp_path: Path) -> None:
    result, seen = _run_one_shot_step(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{_PROJECT}-redpanda-local-sasl" in seen
    assert f"{_PROJECT}-redpanda-partition-cap" in seen
    assert f"{_PROJECT}-local-tenant-registry-seed" in seen


def test_one_shot_step_fails_when_the_sasl_one_shot_exited_non_zero(
    tmp_path: Path,
) -> None:
    result, _ = _run_one_shot_step(tmp_path, failed="redpanda-local-sasl")
    assert result.returncode != 0
    assert "redpanda-local-sasl=exited exit=1" in result.stdout


def test_one_shot_step_fails_when_the_sasl_one_shot_never_ran(tmp_path: Path) -> None:
    result, _ = _run_one_shot_step(tmp_path, missing="redpanda-local-sasl")
    assert result.returncode != 0
    assert "redpanda-local-sasl=missing" in result.stdout
