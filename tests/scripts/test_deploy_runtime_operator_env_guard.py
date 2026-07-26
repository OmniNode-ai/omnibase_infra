# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Behavioral tests for deploy-runtime.sh's operator-env preflight (OMN-14958).

Live failure being eliminated: deploy run 29977968728 (release-train-lab.yml,
job `Deploy triggering tag to its lane`) died inside the containerized
omninode-deploy-runner at deploy-runtime.sh's `source ~/.omnibase/.env` with
bash's bare

    deploy-runtime.sh: line 38: /home/runner/.omnibase/.env: No such file or
    directory

-- an unnamed crash, before any build/compose action, because the runner
container's $HOME carries no operator env. The fix parameterizes the path
(OMNIBASE_OPERATOR_ENV_FILE, default ${HOME}/.omnibase/.env) and turns the
missing-file case into a NAMED, actionable precondition failure (exit 64,
token OPERATOR_ENV_MISSING).

These tests run the real script (no simulation): the guard fires before any
side-effecting code, so executing it with a bare $HOME is safe.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runtime.sh"


def _run(
    tmp_home: Path,
    extra_env: dict[str, str] | None = None,
    args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {
        k: v
        for k, v in os.environ.items()
        # Hermetic: the guard must see only what the test provides.
        if k not in {"OMNIBASE_OPERATOR_ENV_FILE", "HEALTH_CHECK_URL", "INFRA_HOST"}
    }
    env["HOME"] = str(tmp_home)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(DEPLOY_SCRIPT), *(args or ["--help"])],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.unit
def test_missing_default_operator_env_is_named_exit_64(tmp_path: Path) -> None:
    """No ~/.omnibase/.env and no override: NAMED error, exit 64 -- never the
    bare `source: No such file or directory` crash (exit 1) that killed the
    runner deploy before any build action."""
    home = tmp_path / "home"
    home.mkdir()
    result = _run(home)
    assert result.returncode == 64, (
        f"expected exit 64 (named precondition failure), got {result.returncode}; "
        f"stderr: {result.stderr}"
    )
    assert "OPERATOR_ENV_MISSING" in result.stderr
    # The message must name the path it looked for and the override knob.
    assert str(home / ".omnibase" / ".env") in result.stderr
    assert "OMNIBASE_OPERATOR_ENV_FILE" in result.stderr
    assert "No such file or directory" not in result.stderr


@pytest.mark.unit
def test_missing_explicit_operator_env_is_named_exit_64(tmp_path: Path) -> None:
    """An explicit OMNIBASE_OPERATOR_ENV_FILE pointing at a missing file must
    fail the same named way, citing the explicit path."""
    home = tmp_path / "home"
    home.mkdir()
    missing = tmp_path / "nope" / "operator.env"
    result = _run(home, extra_env={"OMNIBASE_OPERATOR_ENV_FILE": str(missing)})
    assert result.returncode == 64, result.stderr
    assert "OPERATOR_ENV_MISSING" in result.stderr
    assert str(missing) in result.stderr


@pytest.mark.unit
def test_present_operator_env_passes_guard(tmp_path: Path) -> None:
    """A provisioned env file (the runner's /run/omnibase-operator.env case)
    passes the guard: --help completes exit 0 with no OPERATOR_ENV_MISSING."""
    home = tmp_path / "home"
    home.mkdir()
    op_env = tmp_path / "operator.env"
    # INFRA_HOST is consumed by the script's own fail-fast HEALTH_CHECK_URL
    # derivation further down -- a minimal-but-real operator env.
    op_env.write_text("INFRA_HOST=127.0.0.1\n", encoding="utf-8")
    result = _run(home, extra_env={"OMNIBASE_OPERATOR_ENV_FILE": str(op_env)})
    assert result.returncode == 0, (
        f"--help should exit 0 past the guard; stderr: {result.stderr}"
    )
    assert "OPERATOR_ENV_MISSING" not in result.stderr
