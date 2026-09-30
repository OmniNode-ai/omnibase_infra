# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A runner without Docker Compose fails the compose runtime boot (OMN-20147).

``reusable-runtime-boot.yml`` used to answer a missing Compose front end by
writing a skip reason to ``$GITHUB_ENV`` and exiting 0, and every later step was
gated on that reason being empty. The job then completed green having booted
nothing (OMN-18811 AC4). Now that ``Runtime Boot Smoke (compose)`` is a strict
CI Summary gate on ``merge_group`` (OMN-20147), a green that ran nothing would
pass the one real-runtime check in the queue, so the step must fail instead.

These tests run the step's own shell body, extracted from the workflow file,
under bash with a PATH that holds either no Compose at all or a stub ``docker``
that answers ``compose version``. They do not re-state the script.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "reusable-runtime-boot.yml"
RESOLVE_STEP = "Resolve Docker Compose command"
BASH = "/bin/bash"


def _workflow() -> dict[str, Any]:
    loaded = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _boot_steps() -> list[dict[str, Any]]:
    steps = _workflow()["jobs"]["boot"]["steps"]
    assert isinstance(steps, list)
    return steps


def _resolve_script() -> str:
    matched = [s for s in _boot_steps() if s.get("name") == RESOLVE_STEP]
    assert len(matched) == 1, f"expected exactly one {RESOLVE_STEP!r} step"
    run = matched[0].get("run")
    assert isinstance(run, str) and run.strip()
    return run


def _run_resolve(tmp_path: Path, bin_dir: Path) -> tuple[int, str, str]:
    github_env = tmp_path / "github_env"
    github_env.write_text("", encoding="utf-8")
    proc = subprocess.run(
        [BASH, "-c", _resolve_script()],
        env={
            "PATH": str(bin_dir),
            "GITHUB_ENV": str(github_env),
            "HOME": str(tmp_path),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr, github_env.read_text("utf-8")


@pytest.mark.skipif(not os.access(BASH, os.X_OK), reason="needs /bin/bash")
def test_compose_absent_fails_the_step(tmp_path: Path) -> None:
    """No `docker compose` and no `docker-compose`: the step exits non-zero."""
    empty_bin = tmp_path / "bin"
    empty_bin.mkdir()

    code, output, github_env = _run_resolve(tmp_path, empty_bin)

    assert code != 0, (
        "compose-absent runner must FAIL the runtime boot, not complete green "
        f"having run nothing (OMN-18811 AC4); output was:\n{output}"
    )
    assert "::error::" in output, "the failure must be an annotated error"
    assert "RUNTIME_BOOT_SKIP_REASON" not in github_env
    assert "DOCKER_COMPOSE_CMD" not in github_env


@pytest.mark.skipif(not os.access(BASH, os.X_OK), reason="needs /bin/bash")
def test_compose_present_resolves_the_command(tmp_path: Path) -> None:
    """Positive control: a `docker` that answers `compose version` passes."""
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    stub = stub_bin / "docker"
    stub.write_text(
        '#!/bin/bash\n[ "$1" = compose ] && [ "$2" = version ] && exit 0\nexit 1\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)

    code, output, github_env = _run_resolve(tmp_path, stub_bin)

    assert code == 0, output
    assert "DOCKER_COMPOSE_CMD=docker compose" in github_env


def test_no_step_is_gated_on_a_skip_reason() -> None:
    """Nothing in the workflow can route around the boot on a skip variable."""
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "RUNTIME_BOOT_SKIP_REASON" not in text
