# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev lane's compose environment and deploy guard take the embedding address from the overlay (OMN-24025).

The operator store keeps whatever a person last typed. On the dev host it still names the
removed embedding server; a dev deploy must not read it, and no hand edit of the store may
be the way to move the address. These tests build the dev-fenced agent the way the
exemplar ``test_lane_env_overlay_home_omn22478`` does and read what compose and the guard
actually receive. Every address is synthetic (192.0.2.0/24).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from deploy_agent import executor as executor_mod
from deploy_agent.executor import DeployExecutor

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

pytestmark = [pytest.mark.unit, pytest.mark.llm_endpoint_env_render]

REPO_ROOT = Path(__file__).resolve().parents[4]
STALE_ADDRESS = "http://192.0.2.50:8002"
OVERLAY_ADDRESS = "http://192.0.2.50:8100"
EMBEDDING_PATH = "/v1/embeddings"
SLOT_ID = "embeddings-201"
ENV_STATE_DIR = "DEPLOY_AGENT_STATE_DIR"
ENV_OVERLAY_HOME_CLONE = "DEPLOY_AGENT_OVERLAY_HOME_CLONE"


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


@pytest.fixture
def dev_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A dev-fenced agent whose operator store still names the removed address."""
    lane_dir = _lane_overlays(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setattr(executor_mod, "REPO_DIR", str(REPO_ROOT))
    monkeypatch.setenv("ONEX_LANE_OVERLAY_DIR", str(lane_dir))
    monkeypatch.setenv("DEPLOY_AGENT_ALLOWED_LANES", "dev")
    monkeypatch.setenv(ENV_STATE_DIR, str(state))
    monkeypatch.setenv("LLM_EMBEDDING_URL", STALE_ADDRESS)
    monkeypatch.delenv("BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL", raising=False)
    # Only the overlay home these tests build may hold the overlay; the host's own roots and
    # workspace clone must not.
    monkeypatch.delenv("ONEX_SKILL_OVERLAY_ROOTS", raising=False)
    monkeypatch.delenv("ONEX_WORKSPACE_CONFIG_ROOT", raising=False)
    monkeypatch.setattr(executor_mod._LaneOverlayClone, "path", None)
    monkeypatch.setattr(executor_mod, "operator_store_overlay", dict)
    monkeypatch.setattr(executor_mod, "_load_runtime_policy_env", dict)
    home = _overlay_home(tmp_path, {SLOT_ID: OVERLAY_ADDRESS})
    monkeypatch.setenv(ENV_OVERLAY_HOME_CLONE, str(home))
    return {"lane_dir": lane_dir, "state": state, "home": home, "tmp": tmp_path}


def _completed(returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout="", stderr=""
    )


def test_golden_compose_env_takes_the_address_from_the_overlay(
    dev_agent: dict[str, Any],
) -> None:
    env = executor_mod._compose_env()

    assert env["LLM_EMBEDDING_URL"] == OVERLAY_ADDRESS
    assert (
        env["BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL"]
        == f"{OVERLAY_ADDRESS}{EMBEDDING_PATH}"
    )
    rendered_dir = Path(env["ONEX_LANE_OVERLAY_DIR"])
    assert rendered_dir == dev_agent["state"] / "rendered-lane-overlays"
    bound = yaml.safe_load(
        (rendered_dir / "dev.bifrost.yaml").read_text(encoding="utf-8")
    )
    assert bound["backends"][0]["endpoint_url"] == f"{OVERLAY_ADDRESS}{EMBEDDING_PATH}"


def test_control_an_agent_not_fenced_to_dev_keeps_the_stores_address(
    dev_agent: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "DEPLOY_AGENT_ALLOWED_LANES", executor_mod.EnumRuntimeLane.STABILITY_TEST.value
    )

    env = executor_mod._compose_env()

    assert env["LLM_EMBEDDING_URL"] == STALE_ADDRESS
    assert env["ONEX_LANE_OVERLAY_DIR"] == str(dev_agent["lane_dir"].resolve())
    assert "BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL" not in env


def test_control_a_stability_job_of_a_dev_agent_keeps_the_stores_address(
    dev_agent: dict[str, Any],
) -> None:
    with executor_mod.active_deploy_lane(executor_mod.EnumRuntimeLane.STABILITY_TEST):
        env = executor_mod._compose_env()

    assert env["LLM_EMBEDDING_URL"] == STALE_ADDRESS


def test_error_a_dev_deploy_whose_overlay_cannot_be_read_is_refused(
    dev_agent: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENV_OVERLAY_HOME_CLONE)

    with pytest.raises(RuntimeError, match="llm_endpoints overlay"):
        executor_mod._compose_env()


def test_error_a_dev_deploy_whose_overlay_omits_the_slot_is_refused_by_name(
    dev_agent: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    other = dev_agent["tmp"] / "other"
    other.mkdir()
    home = _overlay_home(other, {"some-other-slot": OVERLAY_ADDRESS})
    monkeypatch.setenv(ENV_OVERLAY_HOME_CLONE, str(home))

    with pytest.raises(RuntimeError, match=SLOT_ID):
        executor_mod._compose_env()


def _captured_guard_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[list[str], dict[str, str] | None]]:
    seen: list[tuple[list[str], dict[str, str] | None]] = []

    def fake_run(
        cmd: list[str],
        timeout: int,
        *,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        seen.append((cmd, env))
        return _completed()

    monkeypatch.setattr(executor_mod, "_run", fake_run)
    return seen


def test_golden_guard_checks_the_rendered_environment_and_skips_the_stores_value(
    dev_agent: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    store = dev_agent["tmp"] / "store.env"
    store.write_text(f"LLM_EMBEDDING_URL={STALE_ADDRESS}\n", encoding="utf-8")
    monkeypatch.setenv("OMNIBASE_ENV_FILE", str(store))
    seen = _captured_guard_commands(monkeypatch)

    DeployExecutor().validate_llm_endpoint_env_contract()

    process_cmd, process_env = seen[0]
    assert "--env-file" not in process_cmd
    assert process_env is not None
    assert process_env["LLM_EMBEDDING_URL"] == OVERLAY_ADDRESS
    file_commands = [cmd for cmd, _ in seen if "--env-file" in cmd]
    assert file_commands
    for cmd in file_commands:
        assert cmd[cmd.index("--skip-env-var") + 1] == "LLM_EMBEDDING_URL"


def test_control_guard_outside_the_dev_fence_skips_nothing(
    dev_agent: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "DEPLOY_AGENT_ALLOWED_LANES", executor_mod.EnumRuntimeLane.STABILITY_TEST.value
    )
    store = dev_agent["tmp"] / "store.env"
    store.write_text(f"LLM_EMBEDDING_URL={STALE_ADDRESS}\n", encoding="utf-8")
    monkeypatch.setenv("OMNIBASE_ENV_FILE", str(store))
    seen = _captured_guard_commands(monkeypatch)

    DeployExecutor().validate_llm_endpoint_env_contract()

    assert all("--skip-env-var" not in cmd for cmd, _ in seen)


def test_seam_every_name_the_guard_skips_is_a_contract_slot_env_var(
    dev_agent: dict[str, Any],
) -> None:
    contract = yaml.safe_load(
        (REPO_ROOT / "contracts" / "llm_endpoints.yaml").read_text(encoding="utf-8")
    )
    slot_env_vars = {
        e["url_env_var"] for e in contract["endpoints"] if e.get("url_env_var")
    }

    skipped, _ = executor_mod._llm_endpoint_env_overlay(
        {k: v for k, v in os.environ.items() if k != "ONEX_LANE_OVERLAY_DIR"}
    )

    assert skipped
    assert set(skipped) <= slot_env_vars
