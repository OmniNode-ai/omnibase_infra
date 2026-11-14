# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The stability-lane refresh applies the overlay embedding address over the operator file (OMN-24119).

Runs ``scripts/runtime_build/refresh_stability_lane.sh`` in plan mode (no ``--execute``) inside a
sandbox of three tagged clones, a release pin, stub ``docker``/``curl``/``onex-runtime-deploy``
programs that log every call, and an operator env file that still names the removed embedding
server. The script's own log line says which address the render applied, and the stub call log
says whether anything was built. The exemplar is
``test_refresh_dev_lane_overlay_embedding_omn24081``. Every address is synthetic (192.0.2.0/24).
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "runtime_build" / "refresh_stability_lane.sh"

#: The lane id the script hardcodes; its pin file is named for it. Read from the script so no
#: lane id is spelled in this public repository.
_LANE_MATCH = re.search(r'^readonly LANE="([^"]+)"$', _SCRIPT.read_text(), re.M)
assert _LANE_MATCH is not None
LANE = _LANE_MATCH.group(1)

STALE_ADDRESS = "http://192.0.2.50:8002"
OVERLAY_ADDRESS = "http://192.0.2.50:8100"
SLOT_ID = "embeddings-201"

_POP_NAMES = {
    "DEPLOY_AGENT_OVERLAY_HOME_CLONE",
    "DEPLOY_AGENT_STATE_DIR",
    "ONEX_LANE_OVERLAY_DIR",
    "LLM_EMBEDDING_URL",
    "LANE_MANIFEST",
    "OMNIBASE_INFRA_COMPOSE_PROJECT",
    "ONEX_LANE_LOCK_HELD",
    "ONEX_DEPLOY_LANE_ENV_EXPORTS",
    "STABILITY_DESIRED_STATE",
}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.test", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        env=scrub_git_location_env(os.environ),
    )


def _tagged_clone(path: Path) -> None:
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "commit", "-q", "--allow-empty", "-m", "x")
    _git(path, "tag", "v1.2.3")


def _stub(path: Path, log: Path) -> None:
    path.write_text(f'#!/bin/sh\necho "$0 $*" >> "{log}"\nexit 1\n', encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _make_sandbox(
    tmp_path: Path, endpoint_urls: dict[str, str] | None
) -> tuple[dict[str, str], Path]:
    """The environment to run the script in, and the file the stubs log their calls to."""
    ws = tmp_path / "ws"
    ws.mkdir()
    for name in ("omnibase_infra", "omnibase_core", "omnimarket"):
        _tagged_clone(ws / name)

    lane_overlays = tmp_path / "cfg" / "config" / "lab" / "lane-overlays"
    lane_overlays.mkdir(parents=True)
    (lane_overlays / f"{LANE}.release-pin.yaml").write_text(
        f"schema_version: 1\nlane: {LANE}\npin:\n"
        "  omnibase_infra: v1.2.3\n  omnibase_core: v1.2.3\n  omnimarket: v1.2.3\n",
        encoding="utf-8",
    )
    (lane_overlays / "lane-under-test.bifrost.yaml").write_text(
        yaml.safe_dump(
            {
                "lane": "lane-under-test",
                "backends": [
                    {
                        "backend_id": "local-embedding",
                        "endpoint_url": f"{STALE_ADDRESS}/v1/embeddings",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    calls = tmp_path / "stub-calls.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("docker", "curl", "onex-runtime-deploy"):
        _stub(bin_dir / name, calls)

    operator_env = tmp_path / "operator.env"
    operator_env.write_text(f"LLM_EMBEDDING_URL={STALE_ADDRESS}\n", encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()

    env = scrub_git_location_env(os.environ)
    for name in _POP_NAMES:
        env.pop(name, None)
    env["PATH"] = f"{bin_dir}:{os.environ['PATH']}"
    env["HOME"] = str(home)
    env["OMNI_HOME"] = str(ws)
    env["DEPLOY_RUNTIME"] = str(bin_dir / "onex-runtime-deploy")
    env["OMNIBASE_OPERATOR_ENV_FILE"] = str(operator_env)
    env["ONEX_WORKSPACE_CONFIG_ROOT"] = str(tmp_path / "cfg")
    env["ONEX_DEPLOY_REASON"] = "OMN-24119 test"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO / "scripts" / "deploy-agent"), str(_REPO / "src")]
    )
    if endpoint_urls is None:
        env.pop("ONEX_SKILL_OVERLAY_ROOTS", None)
        env["DEPLOY_AGENT_OVERLAY_HOME_CLONE"] = str(tmp_path / "no-clone")
    else:
        overlay_root = tmp_path / "overlay-root"
        (overlay_root / "llm_endpoints").mkdir(parents=True)
        (overlay_root / "llm_endpoints" / "overlay.yaml").write_text(
            yaml.safe_dump({"schema_version": 1, "endpoint_urls": endpoint_urls}),
            encoding="utf-8",
        )
        env["ONEX_SKILL_OVERLAY_ROOTS"] = str(overlay_root)
    return env, calls


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_SCRIPT)],
        env=env,
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.integration
def test_golden_the_stability_refresh_applies_the_overlay_address_over_the_operator_file(
    tmp_path: Path,
) -> None:
    env, _calls = _make_sandbox(tmp_path, {SLOT_ID: OVERLAY_ADDRESS})
    result = _run(env)
    combined = result.stdout + result.stderr
    # Later steps need a real docker, so the exit code is not asserted: the log line is.
    assert "LLM endpoint env applied from the overlays:" in combined
    assert f"LLM_EMBEDDING_URL={OVERLAY_ADDRESS}" in combined
    assert STALE_ADDRESS not in combined


@pytest.mark.integration
def test_error_an_overlay_without_the_embedding_slot_stops_before_anything_is_built(
    tmp_path: Path,
) -> None:
    env, calls = _make_sandbox(tmp_path, {"some-other-slot": OVERLAY_ADDRESS})
    result = _run(env)
    combined = result.stdout + result.stderr
    assert result.returncode == 64
    assert "LANE_ENV_FRAGMENT_REFUSED" in combined
    assert SLOT_ID in combined
    assert not calls.exists()


@pytest.mark.integration
def test_error_no_overlay_stops_before_anything_is_built(tmp_path: Path) -> None:
    env, calls = _make_sandbox(tmp_path, None)
    result = _run(env)
    combined = result.stdout + result.stderr
    assert result.returncode == 64
    assert "LANE_ENV_FRAGMENT_REFUSED" in combined
    assert not calls.exists()
