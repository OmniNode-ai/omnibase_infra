# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev-lane refresh applies the overlay embedding address over the operator file (OMN-24081).

Runs ``scripts/runtime_build/refresh_dev_lane.sh`` in ``--print-rollback-cmd``
mode (read-only, no docker call) inside a sandbox, and reads what the script's
own log line prints, so the overlay lookup, the operator env file and the
rendered address are exercised together.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from tests.helpers.lane_census_fixtures import NEUTRAL_CONFIG_ROOT

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "runtime_build" / "refresh_dev_lane.sh"

_POP_NAMES = {
    "DEPLOY_AGENT_OVERLAY_HOME_CLONE",
    "DEPLOY_AGENT_STATE_DIR",
    "ONEX_LANE_OVERLAY_DIR",
    "LLM_EMBEDDING_URL",
    "LANE_MANIFEST",
    "OMNIBASE_INFRA_COMPOSE_PROJECT",
    "ONEX_LANE_LOCK_HELD",
}


def _build_env(
    tmp_path: Path,
    *,
    overlay_root: Path,
    operator_env: Path,
    home: Path,
    bin_dir: Path,
) -> dict[str, str]:
    env = scrub_git_location_env(os.environ)
    for name in _POP_NAMES:
        env.pop(name, None)
    env["PATH"] = f"{bin_dir}:{os.environ['PATH']}"
    env["HOME"] = str(home)
    env["OMNI_HOME"] = str(tmp_path / "ws")
    env["OMNIBASE_OPERATOR_ENV_FILE"] = str(operator_env)
    env["ONEX_SKILL_OVERLAY_ROOTS"] = str(overlay_root)
    env["ONEX_WORKSPACE_CONFIG_ROOT"] = str(NEUTRAL_CONFIG_ROOT)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO / "scripts" / "deploy-agent"), str(_REPO / "src")]
    )
    return env


def _make_sandbox(tmp_path: Path) -> tuple[dict[str, str], Path]:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "omnibase_infra").symlink_to(_REPO)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "onex-runtime-deploy"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)

    overlay_root = tmp_path / "overlay-root"
    (overlay_root / "llm_endpoints").mkdir(parents=True)
    (overlay_root / "llm_endpoints" / "overlay.yaml").write_text(
        "schema_version: 1\nendpoint_urls:\n  embeddings-201: http://192.0.2.50:8100\n"
    )

    operator_env = tmp_path / "operator.env"
    operator_env.write_text("LLM_EMBEDDING_URL=http://192.0.2.50:8002\n")

    home = tmp_path / "home"
    home.mkdir()

    env = _build_env(
        tmp_path,
        overlay_root=overlay_root,
        operator_env=operator_env,
        home=home,
        bin_dir=bin_dir,
    )
    return env, home


@pytest.fixture
def sandbox(tmp_path: Path) -> tuple[dict[str, str], Path]:
    return _make_sandbox(tmp_path)


def test_golden_the_refresh_applies_the_overlay_address_over_the_operator_file(
    sandbox: tuple[dict[str, str], Path],
    tmp_path: Path,
) -> None:
    env, _home = sandbox
    result = subprocess.run(
        ["bash", str(_SCRIPT), "--print-rollback-cmd"],
        env=env,
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0
    assert "LLM_EMBEDDING_URL=http://192.0.2.50:8100" in combined
    assert "http://192.0.2.50:8002" not in combined


def test_error_a_refresh_with_no_overlay_stops_before_anything_is_built(
    sandbox: tuple[dict[str, str], Path],
    tmp_path: Path,
) -> None:
    env, _home = sandbox
    env.pop("ONEX_SKILL_OVERLAY_ROOTS", None)
    env["DEPLOY_AGENT_OVERLAY_HOME_CLONE"] = str(tmp_path / "no-clone")
    result = subprocess.run(
        ["bash", str(_SCRIPT), "--print-rollback-cmd"],
        env=env,
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 64
    assert "LANE_ENV_FRAGMENT_REFUSED" in combined


def test_error_an_overlay_without_the_embedding_slot_is_refused(
    tmp_path: Path,
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "omnibase_infra").symlink_to(_REPO)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "onex-runtime-deploy"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)

    overlay_root = tmp_path / "overlay-root"
    (overlay_root / "llm_endpoints").mkdir(parents=True)
    (overlay_root / "llm_endpoints" / "overlay.yaml").write_text(
        "schema_version: 1\nendpoint_urls:\n  some-other-slot: http://192.0.2.50:8100\n"
    )

    operator_env = tmp_path / "operator.env"
    operator_env.write_text("LLM_EMBEDDING_URL=http://192.0.2.50:8002\n")

    home = tmp_path / "home"
    home.mkdir()

    env = _build_env(
        tmp_path,
        overlay_root=overlay_root,
        operator_env=operator_env,
        home=home,
        bin_dir=bin_dir,
    )
    result = subprocess.run(
        ["bash", str(_SCRIPT), "--print-rollback-cmd"],
        env=env,
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 64
    assert "embeddings-201" in combined
