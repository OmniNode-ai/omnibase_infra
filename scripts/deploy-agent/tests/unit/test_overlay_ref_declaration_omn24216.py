# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The signed rebuild command carries the overlays revision the dev lane mounts (OMN-24216).

The command's ``overlay_ref`` names the 40-hex head of an overlays repository
revision the dev lane mounts as its overlay store root for this deploy. It rides
with the rollback declaration a deliberate off-lineage deploy already needs, like
``open_pr`` (OMN-21928); other lanes refuse it, and ``_compose_env`` mounts the
recorded revision's rendered folder only when one was recorded.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
import yaml
from deploy_agent import executor as executor_mod
from deploy_agent.events import (
    EnumRuntimeLane,
    ModelRebuildRequested,
    ModelRollbackDeclaration,
)
from deploy_agent.lane_policy import ENV_ALLOWED_LANES
from pydantic import ValidationError

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SHA = "ab" * 20
ENDPOINT_URL = "http://192.0.2.50:8100"  # synthetic (RFC 5737), like the exemplar's
ROLLBACK = ModelRollbackDeclaration(actor="slot-fix", reason="mount overlay revision")
DECLARED = executor_mod.load_dev_boot_guard(
    "nodes.node_setup_local_provision_effect.runtime_overlay_revision"
).declaration(executor_mod)
CLONE_ENV = str(DECLARED["clone_env"])
STATE_DIR_ENV = str(DECLARED["state_dir_env"])
RENDERED_DIR_NAME = str(DECLARED["rendered_dir_name"])
STORE_ROOT_ENV = str(DECLARED["store_root_env"])
OVERLAY_DIR = str(DECLARED["overlay_dir"])


def _command(lane: str = "dev", **overrides: object) -> ModelRebuildRequested:
    fields: dict[str, object] = {
        "correlation_id": UUID(int=24216),
        "requested_by": "operator-manual",
        "scope": "runtime",
        "runtime_lane": lane,
        "git_ref": SHA,
        "overlay_ref": SHA,
        "rollback": ROLLBACK,
    }
    fields.update(overrides)
    return ModelRebuildRequested.model_validate(fields)


def test_the_dev_command_carries_the_overlay_ref() -> None:
    cmd = _command()
    assert cmd.overlay_ref == SHA
    assert cmd.runtime_lane is EnumRuntimeLane.DEV


def test_a_command_without_an_overlay_ref_is_accepted() -> None:
    cmd = _command(overlay_ref="", rollback=None)
    assert cmd.overlay_ref == ""
    assert cmd.rollback is None


def test_a_stability_command_refuses_an_overlay_ref() -> None:
    with pytest.raises(ValidationError, match="overlay_ref needs runtime_lane dev"):
        _command(EnumRuntimeLane.STABILITY_TEST.value)


def test_an_overlay_ref_needs_a_rollback_declaration() -> None:
    with pytest.raises(ValidationError, match="overlay_ref needs runtime_lane dev"):
        _command(rollback=None)


def test_a_malformed_overlay_ref_is_refused() -> None:
    with pytest.raises(ValidationError, match="overlay_ref"):
        _command(overlay_ref="not-a-sha")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.test", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        env=scrub_git_location_env(os.environ),
    )


def _overlay_home(root: Path) -> tuple[Path, str]:
    """A git repo holding the overlays the dev lane reads, and its head.

    The clone env names one repo for every overlay reader: the contract hotload
    overlay this revision mounts, and the llm_endpoints overlay the dev fence
    refuses to deploy without (one flagged slot, a synthetic address).
    """
    repo = root / "overlay-home"
    repo.mkdir()
    _git(repo, "init", "-q")
    overlay = repo / OVERLAY_DIR / "runtime_contract_hotload" / "overlay.yaml"
    overlay.parent.mkdir(parents=True)
    overlay.write_text(
        yaml.safe_dump({"schema_version": 1, "lanes": {}}, sort_keys=False),
        encoding="utf-8",
    )
    _git(repo, "add", str(overlay.relative_to(repo)))
    endpoint = repo / OVERLAY_DIR / "llm_endpoints" / "overlay.yaml"
    endpoint.parent.mkdir(parents=True)
    endpoint.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "endpoint_urls": {"embeddings-201": ENDPOINT_URL}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _git(repo, "add", str(endpoint.relative_to(repo)))
    _git(repo, "commit", "-q", "-m", "overlay")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    ).stdout.strip()
    return repo, head


@pytest.fixture
def dev_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A dev-fenced agent whose overlays clone lives in tmp_path (OMN-24025 style)."""
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setattr(executor_mod, "REPO_DIR", str(REPO_ROOT))
    monkeypatch.setenv(ENV_ALLOWED_LANES, "dev")
    monkeypatch.setenv(STATE_DIR_ENV, str(state))
    monkeypatch.delenv(STORE_ROOT_ENV, raising=False)
    # Only the overlays repo these tests build may hold it; the host's own
    # roots and workspace clone must not.
    monkeypatch.delenv("ONEX_SKILL_OVERLAY_ROOTS", raising=False)
    monkeypatch.delenv("ONEX_WORKSPACE_CONFIG_ROOT", raising=False)
    monkeypatch.setattr(executor_mod._LaneOverlayClone, "path", None)
    monkeypatch.setattr(executor_mod, "operator_store_overlay", dict)
    monkeypatch.setattr(executor_mod, "_load_runtime_policy_env", dict)
    home, head = _overlay_home(tmp_path)
    monkeypatch.setenv(CLONE_ENV, str(home))
    return {"state": state, "home": home, "head": head}


def test_golden_a_recorded_revision_names_the_rendered_store_root(
    dev_agent: dict[str, Any],
) -> None:
    executor_mod.record_overlay_revision(dev_agent["head"])

    env = executor_mod._compose_env()

    assert env[STORE_ROOT_ENV] == str(
        dev_agent["state"] / RENDERED_DIR_NAME / dev_agent["head"] / OVERLAY_DIR
    )


def test_control_no_record_leaves_no_store_root_in_compose_env(
    dev_agent: dict[str, Any],
) -> None:
    # Failing control: if _compose_env ever set the key without a record call,
    # this delenv-then-assert-absent test fails.
    assert STORE_ROOT_ENV not in os.environ

    env = executor_mod._compose_env()

    assert STORE_ROOT_ENV not in env
