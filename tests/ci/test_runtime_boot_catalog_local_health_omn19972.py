# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Execute the laptop-profile health step against a stub Docker (OMN-19972).

OMN-19972's demo half adds the projection API, the LLM call and cost writer and
the consumer-health projection to the ``local`` bundle, and its acceptance says
the ``runtime-boot (mode=catalog-local)`` job asserts each added service
healthy -- falsified by the step passing with a service stopped.

The step derives the services to wait for from the ``local`` render itself
(every service with a healthcheck that is not a one-shot), so a service added
to the bundle later is covered without editing the workflow. These tests run
the step's own ``run:`` text, unmodified: the derivation runs for real through
``uv run``, and ``docker`` and ``sleep`` are stubbed on PATH.
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
_STEP = "Every health-checked laptop service reports healthy"
_PROJECT = "omnibase-infra-local"
_ADDED = (
    "projection-api",
    "omnimarket-projection-llm-cost",
    "consumer-health-projection",
)

_DOCKER_STUB = """#!/usr/bin/env bash
# Stub of `docker inspect --format ... <container>` for the health step.
set -u
[ "$1" = "inspect" ] || { echo "stub docker: unexpected $*" >&2; exit 2; }
container="${!#}"
echo "$container" >> "$STUB_SEEN"
if [ "$container" = "$STUB_MISSING" ]; then
  echo "Error: No such object: $container" >&2
  exit 1
fi
case "$*" in
  *RestartCount*) echo 0 ;;
  *Health.Status*)
    if [ "$container" = "$STUB_UNHEALTHY" ]; then echo unhealthy;
    elif [ "$container" = "$STUB_STARTING" ]; then echo starting;
    else echo healthy; fi ;;
  *) echo "stub docker: unexpected inspect $*" >&2; exit 2 ;;
esac
"""


def _step_script() -> str:
    workflow: dict[str, Any] = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"][_JOB]["steps"]
    matches = [s for s in steps if s.get("name") == _STEP]
    assert len(matches) == 1, f"expected exactly one step named {_STEP!r}"
    run = matches[0]["run"]
    assert isinstance(run, str)
    return run


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _run_step(
    tmp_path: Path, *, unhealthy: str = "", missing: str = "", starting: str = ""
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(bin_dir / "docker", _DOCKER_STUB)
    _write_executable(bin_dir / "sleep", "#!/usr/bin/env bash\nexit 0\n")
    script = tmp_path / "step.sh"
    script.write_text(_step_script(), encoding="utf-8")
    seen = tmp_path / "seen.txt"
    seen.write_text("", encoding="utf-8")

    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "LOCAL_PROJECT": _PROJECT,
        "STUB_SEEN": str(seen),
        "STUB_UNHEALTHY": f"{_PROJECT}-{unhealthy}" if unhealthy else "",
        "STUB_MISSING": f"{_PROJECT}-{missing}" if missing else "",
        "STUB_STARTING": f"{_PROJECT}-{starting}" if starting else "",
    }
    # GitHub runs a `run:` block as `bash -e {0}`, from the repository root.
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


def _show(result: subprocess.CompletedProcess[str]) -> str:
    return f"exit {result.returncode}\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"


def test_step_passes_and_checks_every_added_service(tmp_path: Path) -> None:
    result, seen = _run_step(tmp_path)
    assert result.returncode == 0, _show(result)
    for svc in _ADDED:
        assert f"{_PROJECT}-{svc}" in seen, (svc, seen)


def test_step_fails_when_an_added_service_is_unhealthy(tmp_path: Path) -> None:
    result, _ = _run_step(tmp_path, unhealthy="projection-api")
    assert result.returncode == 1, _show(result)
    assert "projection-api" in result.stdout


def test_step_fails_when_an_added_service_never_started(tmp_path: Path) -> None:
    result, _ = _run_step(tmp_path, missing="omnimarket-projection-llm-cost")
    assert result.returncode == 1, _show(result)
    assert "omnimarket-projection-llm-cost" in result.stdout


def test_step_fails_when_an_added_service_never_leaves_starting(tmp_path: Path) -> None:
    result, _ = _run_step(tmp_path, starting="consumer-health-projection")
    assert result.returncode == 1, _show(result)
    assert "consumer-health-projection" in result.stdout
