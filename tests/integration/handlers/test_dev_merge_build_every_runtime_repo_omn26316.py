# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration: the overdue check reads real git clones of every composed repository (OMN-26316).

THE RECORD
----------
2026-11-26 on h201: omnimarket#6872 merged at 05:47:54Z and no dev build was requested for
it, because ``check_overdue_dev_merges`` read only the agent's own omnibase_infra clone. The
agent journal also repeated ``Topic onex.evt.deploy.dev-merge-build-overdue.v1 not found in
cluster metadata`` (about 10 per second): the topic was declared nowhere the provisioner reads.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.models.model_dev_merge_build_command import (
    ModelDevMergeBuildCommand,
)
from omnibase_infra.models.model_dev_merge_build_overdue_event import (
    ModelDevMergeBuildOverdueEvent,
)
from omnibase_infra.nodes.node_deploy_job_reading_effect import (
    runtime_deploy_job_reading as caller,
)

pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parents[3]
_NOW = datetime.now(UTC)


class _Sender:
    def __init__(self) -> None:
        self.events: list[ModelDevMergeBuildOverdueEvent] = []
        self.commands: list[ModelDevMergeBuildCommand] = []

    def publish_event(self, event: ModelDevMergeBuildOverdueEvent) -> None:
        self.events.append(event)

    def send_command(self, command: ModelDevMergeBuildCommand) -> str:
        self.commands.append(command)
        return "published"


def _clone(
    path: Path, path_changed: str, subject: str, age: timedelta
) -> tuple[Path, str]:
    path.mkdir()
    stamp = (_NOW - age).isoformat()
    ambient = {
        **os.environ,
        "GIT_COMMITTER_DATE": stamp,
        "GIT_AUTHOR_DATE": stamp,
    }

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
            env=scrub_git_location_env(ambient),
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    target = path / path_changed
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x\n")
    git("add", "-A")
    git("commit", "-q", "-m", subject)
    return path, git("rev-parse", "HEAD")


@pytest.fixture
def lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    infra, infra_sha = _clone(
        tmp_path / "omnibase_infra", "docs/old.md", "old (#1)", timedelta(days=3)
    )
    market, market_sha = _clone(
        tmp_path / "omnimarket",
        "src/omnimarket/nodes/x.py",
        "feat: runtime change (#6872)",
        timedelta(hours=2),
    )
    state = tmp_path / "state"
    state.mkdir()
    execution = SimpleNamespace(
        STATE_DIR=str(state),
        deploy_repo_dir=lambda: str(infra),
        AGENT_CLONE_ROOT=str(_REPO),
        _sibling_clone=lambda repository: tmp_path / repository,
    )
    sender = _Sender()
    monkeypatch.setattr(caller, "_agent", lambda: execution)
    monkeypatch.setattr(caller, "AgentDevMergeSender", lambda *_a: sender)
    monkeypatch.setattr(caller, "_TRACKED_HEAD", "HEAD")
    monkeypatch.setattr(
        caller, "_composed_repositories", lambda _e: ("omnimarket", "omnibase_infra")
    )
    return SimpleNamespace(
        execution=execution,
        sender=sender,
        market_sha=market_sha,
        market=market,
        infra_sha=infra_sha,
    )


def test_real_sibling_clone_merge_is_requested_at_its_own_head(
    lane: SimpleNamespace,
) -> None:
    decisions = caller.check_overdue_dev_merges(force=True)
    assert decisions is not None
    assert {d.merge_sha: d.outcome for d in decisions}[lane.market_sha] == "REQUEST"
    # The command names the sibling repo, but its git_ref is the agent's own clone head (OMN-26703).
    assert [(c.repo, c.git_ref) for c in lane.sender.commands] == [
        ("omnimarket", lane.infra_sha)
    ]


def test_every_composed_clone_unread_is_skipped_not_empty(
    lane: SimpleNamespace, tmp_path: Path
) -> None:
    lane.execution.deploy_repo_dir = lambda: str(tmp_path / "missing_infra")
    lane.execution._sibling_clone = lambda repository: tmp_path / "missing" / repository
    assert caller.check_overdue_dev_merges(force=True) is None
    assert lane.sender.commands == []
