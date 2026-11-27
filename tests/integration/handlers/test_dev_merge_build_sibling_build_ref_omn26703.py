# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A sibling-repository overdue build resets the agent's own clone to its own commit (OMN-26703).

Real git clones: the command's ``git_ref`` must resolve in the omnibase_infra clone the agent
resets, never be the sibling's sha (2026-11-26: ``Git reset --hard 3026421861f6 failed``).
"""

from __future__ import annotations

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


def test_sibling_build_ref_resolves_in_the_agents_own_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    infra, infra_sha = _clone(
        tmp_path / "omnibase_infra", "docs/old.md", "old (#1)", timedelta(days=3)
    )
    _, market_sha = _clone(
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
    caller.check_overdue_dev_merges(force=True)
    assert len(sender.commands) == 1
    ref = sender.commands[0].git_ref
    assert ref == infra_sha != market_sha
    resolved = subprocess.run(
        ["git", "-C", str(infra), "cat-file", "-e", f"{ref}^{{commit}}"],
        check=False,
        capture_output=True,
        env=scrub_git_location_env(dict(os.environ)),
    )
    assert resolved.returncode == 0
