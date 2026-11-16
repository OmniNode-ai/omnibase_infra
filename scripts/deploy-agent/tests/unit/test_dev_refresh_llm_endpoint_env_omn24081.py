# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev lane's LLM endpoint env exports take the embedding address from the overlay (OMN-24081).

The operator env file keeps whatever a person last typed. On the dev host it still names the
removed embedding server; the warm refresh sources it and must not end up with that address.
These tests render the exports the refresh script sources after the operator env, run them
through bash over a stale operator env file, and read what the shell ends up holding. The
exemplar is ``test_llm_endpoint_env_compose_omn24025``. Every address is synthetic
(192.0.2.0/24).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml
from deploy_agent import executor as executor_mod

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

pytestmark = [pytest.mark.unit, pytest.mark.llm_endpoint_env_render]

STALE_ADDRESS = "http://192.0.2.50:8002"
OVERLAY_ADDRESS = "http://192.0.2.50:8100"
EMBEDDING_PATH = "/v1/embeddings"
SLOT_ID = "embeddings-201"
ENV_STATE_DIR = "DEPLOY_AGENT_STATE_DIR"


def dev_llm_endpoint_env_exports(
    environ: Mapping[str, str], search_root: Path, state_dir: Path
) -> str:
    """What the refresh script sources: the node module's ``refresh_exports`` over the executor."""
    guard = executor_mod.load_dev_boot_guard(executor_mod._LLM_ENV_MODULE)
    exports: str = guard.refresh_exports(executor_mod, environ, search_root, state_dir)
    return exports


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.test", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        env=scrub_git_location_env(os.environ),
    )


def _overlay_home(root: Path, endpoint_urls: dict[str, str]) -> Path:
    repo = root / "overlay-home"
    repo.mkdir()
    _git(repo, "init", "-q")
    overlay = repo / "overlays" / "llm_endpoints" / "overlay.yaml"
    overlay.parent.mkdir(parents=True)
    overlay.write_text(
        yaml.safe_dump({"schema_version": 1, "endpoint_urls": endpoint_urls}),
        encoding="utf-8",
    )
    _git(repo, "add", str(overlay.relative_to(repo)))
    _git(repo, "commit", "-q", "-m", "overlay")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    return repo


def _lane_overlays(root: Path) -> Path:
    lane_dir = root / "lane-overlays"
    lane_dir.mkdir()
    (lane_dir / "dev.bifrost.yaml").write_text(
        yaml.safe_dump(
            {
                "lane": "dev",
                "backends": [
                    {
                        "backend_id": "local-embedding",
                        "endpoint_url": f"{STALE_ADDRESS}{EMBEDDING_PATH}",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return lane_dir


def test_golden_exports_take_the_address_from_the_overlay(tmp_path: Path) -> None:
    lane_dir = _lane_overlays(tmp_path)
    home = _overlay_home(tmp_path, {SLOT_ID: OVERLAY_ADDRESS})
    state = tmp_path / "state"
    state.mkdir()
    environ: Mapping[str, str] = {
        "LLM_EMBEDDING_URL": STALE_ADDRESS,
        "ONEX_LANE_OVERLAY_DIR": str(lane_dir),
        "DEPLOY_AGENT_OVERLAY_HOME_CLONE": str(home),
        ENV_STATE_DIR: str(state),
    }
    exports = dev_llm_endpoint_env_exports(environ, tmp_path / "no-clones", state)
    script = (
        f"{exports}\n"
        "printf 'LLM_EMBEDDING_URL=%s\\n' \"$LLM_EMBEDDING_URL\"\n"
        "printf 'BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL=%s\\n' \"$BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL\"\n"
        "printf 'ONEX_LANE_OVERLAY_DIR=%s\\n' \"$ONEX_LANE_OVERLAY_DIR\"\n"
    )
    result = subprocess.run(
        ["bash", "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    )
    lines = result.stdout.strip().splitlines()
    assert lines[0] == f"LLM_EMBEDDING_URL={OVERLAY_ADDRESS}"
    assert (
        lines[1]
        == f"BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL={OVERLAY_ADDRESS}{EMBEDDING_PATH}"
    )
    rendered_dir = state / "rendered-lane-overlays"
    assert lines[2] == f"ONEX_LANE_OVERLAY_DIR={rendered_dir}"
    rendered_yaml = rendered_dir / "dev.bifrost.yaml"
    assert rendered_yaml.is_file()
    data = yaml.safe_load(rendered_yaml.read_text(encoding="utf-8"))
    assert data["backends"][0]["endpoint_url"] == f"{OVERLAY_ADDRESS}{EMBEDDING_PATH}"


def test_seam_the_refresh_shell_ends_with_the_overlays_address_not_the_operator_files(
    tmp_path: Path,
) -> None:
    lane_dir = _lane_overlays(tmp_path)
    home = _overlay_home(tmp_path, {SLOT_ID: OVERLAY_ADDRESS})
    state = tmp_path / "state"
    state.mkdir()
    environ: Mapping[str, str] = {
        "LLM_EMBEDDING_URL": STALE_ADDRESS,
        "ONEX_LANE_OVERLAY_DIR": str(lane_dir),
        "DEPLOY_AGENT_OVERLAY_HOME_CLONE": str(home),
        ENV_STATE_DIR: str(state),
    }
    exports = dev_llm_endpoint_env_exports(environ, tmp_path / "no-clones", state)
    operator_env = tmp_path / "operator.env"
    operator_env.write_text(
        f"export LLM_EMBEDDING_URL={STALE_ADDRESS}\n", encoding="utf-8"
    )
    script = (
        f"set -a\n"
        f"source {operator_env}\n"
        f"set +a\n"
        f"printf 'LLM_EMBEDDING_URL=%s\\n' \"$LLM_EMBEDDING_URL\"\n"
        f"source <(printf '%s' \"$EXPORTS\")\n"
        f"printf 'LLM_EMBEDDING_URL=%s\\n' \"$LLM_EMBEDDING_URL\"\n"
    )
    result = subprocess.run(
        ["bash", "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env={**scrub_git_location_env(os.environ), "EXPORTS": exports},
    )
    lines = result.stdout.strip().splitlines()
    assert lines[0] == f"LLM_EMBEDDING_URL={STALE_ADDRESS}"
    assert lines[1] == f"LLM_EMBEDDING_URL={OVERLAY_ADDRESS}"


def test_error_an_overlay_without_the_embedding_slot_is_refused_by_name(
    tmp_path: Path,
) -> None:
    lane_dir = _lane_overlays(tmp_path)
    home = _overlay_home(tmp_path, {"some-other-slot": OVERLAY_ADDRESS})
    state = tmp_path / "state"
    state.mkdir()
    environ: Mapping[str, str] = {
        "LLM_EMBEDDING_URL": STALE_ADDRESS,
        "ONEX_LANE_OVERLAY_DIR": str(lane_dir),
        "DEPLOY_AGENT_OVERLAY_HOME_CLONE": str(home),
        ENV_STATE_DIR: str(state),
    }
    with pytest.raises(RuntimeError, match=SLOT_ID):
        dev_llm_endpoint_env_exports(environ, tmp_path / "no-clones", state)


def test_error_an_unnamed_overlay_clone_is_refused(tmp_path: Path) -> None:
    lane_dir = _lane_overlays(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    environ: Mapping[str, str] = {
        "LLM_EMBEDDING_URL": STALE_ADDRESS,
        "ONEX_LANE_OVERLAY_DIR": str(lane_dir),
        ENV_STATE_DIR: str(state),
    }
    with pytest.raises(RuntimeError, match="llm_endpoints overlay"):
        dev_llm_endpoint_env_exports(environ, tmp_path / "no-clones", state)


def test_the_refresh_script_renders_after_the_operator_env_and_before_the_build() -> (
    None
):
    script_path = (
        Path(__file__).resolve().parents[3] / "runtime_build" / "refresh_dev_lane.sh"
    )
    text = script_path.read_text(encoding="utf-8")
    assert text.index('source "${OMNIBASE_OPERATOR_ENV_FILE}"') < text.index(
        "default_instance_lane_env_exports"
    )
    assert text.index("default_instance_lane_env_exports") < text.index(
        "llm.refresh_exports"
    )
    assert text.index("llm.refresh_exports") < text.index("=== Build + bring-up")


def test_seam_the_scripts_own_python_renders_the_overlay_address(
    tmp_path: Path,
) -> None:
    """Run the exact program the refresh script hands to its interpreter."""
    repo_root = Path(__file__).resolve().parents[4]
    text = (repo_root / "scripts" / "runtime_build" / "refresh_dev_lane.sh").read_text(
        encoding="utf-8"
    )
    program = re.search(
        r"run_python -c '(.*?)' \\\n\s+\"\$\{REPO_ROOT\}\"", text, re.DOTALL
    )
    assert program is not None
    lane_dir = _lane_overlays(tmp_path)
    home = _overlay_home(tmp_path, {SLOT_ID: OVERLAY_ADDRESS})
    state = tmp_path / "state"
    state.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            program.group(1),
            str(repo_root),
            str(tmp_path / "no-clones"),
            str(state),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **scrub_git_location_env(os.environ),
            "PYTHONPATH": os.pathsep.join(
                [str(repo_root / "scripts" / "deploy-agent"), str(repo_root / "src")]
            ),
            "LLM_EMBEDDING_URL": STALE_ADDRESS,
            "ONEX_LANE_OVERLAY_DIR": str(lane_dir),
            "DEPLOY_AGENT_OVERLAY_HOME_CLONE": str(home),
        },
    )
    assert f"export LLM_EMBEDDING_URL={OVERLAY_ADDRESS}\n" in result.stdout
    assert STALE_ADDRESS not in result.stdout
