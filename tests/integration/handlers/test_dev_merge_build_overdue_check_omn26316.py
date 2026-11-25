# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The whole overdue dev merge check over a real git history and recorded job files (OMN-26316).

``check_overdue_dev_merges`` is run as the deploy agent runs it: the settings are the node
contract's own, the merges are read with ``git log`` from a temporary repository whose commits
carry real committer times, the jobs are job record files in a state directory, and every
decision and request is the production handler's. Only the bus sender is replaced, by one that
records the commands it is given.

THE RECORD
----------
2026-11-25 on h201: dev was rebuilt at 07:09Z, 07:23Z and 07:36Z because each build asked for
by the overdue check was pinned to the dev head, which the next check did not count as carrying
the other merges, and because a build had just run. The check now asks for one build of the head,
carries every merge below it, and waits when a dev job was accepted inside the coalesce window.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import BaseModel

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.models.model_dev_merge_build_command import (
    ModelDevMergeBuildCommand,
)
from omnibase_infra.models.model_dev_merge_build_overdue_event import (
    ModelDevMergeBuildOverdueEvent,
)
from omnibase_infra.models.model_overdue_dev_merge_build_decision import (
    ModelOverdueDevMergeBuildDecision,
)
from omnibase_infra.nodes.node_deploy_job_reading_effect import (
    runtime_deploy_job_reading as caller,
)

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[3]
_MINUTE = timedelta(minutes=1)
_HOUR = timedelta(hours=1)


class _Sender:
    def __init__(self) -> None:
        self.events: list[ModelDevMergeBuildOverdueEvent] = []
        self.commands: list[ModelDevMergeBuildCommand] = []

    def publish_event(self, event: ModelDevMergeBuildOverdueEvent) -> None:
        self.events.append(event)

    def send_command(self, command: ModelDevMergeBuildCommand) -> str:
        self.commands.append(command)
        return "published"


class _Host:
    """A clone, a state directory and the clock of one check."""

    def __init__(self, root: Path) -> None:
        self.now = datetime.now(UTC).replace(microsecond=0)
        self.repo = root / "clone"
        self.state = root / "state"
        self.repo.mkdir()
        self.state.mkdir()
        self._git("init", "-q")
        self._git("config", "user.email", "t@example.invalid")
        self._git("config", "user.name", "t")
        self.merges: list[str] = []

    def _git(self, *args: str, at: datetime | None = None) -> str:
        dates = (
            {}
            if at is None
            else {
                "GIT_COMMITTER_DATE": at.isoformat(),
                "GIT_AUTHOR_DATE": at.isoformat(),
            }
        )
        return subprocess.run(
            ["git", "-C", str(self.repo), *args],
            check=True,
            capture_output=True,
            text=True,
            env={**scrub_git_location_env(os.environ), **dates},
        ).stdout.strip()

    def merge(self, age: timedelta) -> str:
        """A runtime merge (a source file changes) committed ``age`` before now."""
        number = len(self.merges) + 1
        source = self.repo / "src" / "omnibase_infra" / "module.py"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"VALUE = {number}\n", encoding="utf-8")
        self._git("add", "-A")
        self._git("commit", "-q", "-m", f"merge (#{number})", at=self.now - age)
        sha = self._git("rev-parse", "HEAD")
        self.merges.append(sha)
        return sha

    def job(
        self, name: str, age: timedelta, *, built_sha: str = "", by: str = "agent/x"
    ) -> None:
        """The agent's record of a finished dev job accepted ``age`` before now."""
        record: dict[str, object] = {
            "correlation_id": f"00000000-0000-5000-8000-{name:0>12}",
            "accepted_at": (self.now - age).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": "success",
            "command": {
                "runtime_lane": "dev",
                "git_ref": "origin/dev",
                "requested_by": by,
            },
        }
        if built_sha:
            record["lineage"] = {"resolved_ref": built_sha}
        (self.state / f"{name}.json").write_text(json.dumps(record), encoding="utf-8")


@pytest.fixture
def sender(monkeypatch: pytest.MonkeyPatch) -> _Sender:
    fake = _Sender()
    monkeypatch.setattr(caller, "AgentDevMergeSender", lambda *_a: fake)
    monkeypatch.setattr(caller, "_TRACKED_HEAD", "HEAD")
    monkeypatch.setattr(caller, "_overdue_last", [])
    return fake


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Host:
    built = _Host(tmp_path)
    monkeypatch.setattr(
        caller,
        "_agent",
        lambda: SimpleNamespace(
            STATE_DIR=str(built.state),
            REPO_DIR=str(built.repo),
            AGENT_CLONE_ROOT=str(_ROOT),
        ),
    )
    return built


def _outcomes(decisions: tuple[BaseModel, ...] | None) -> list[tuple[str, str]]:
    assert decisions is not None, (
        "the check was skipped: merges, jobs or contract unread"
    )
    return [
        (
            str(cast("ModelOverdueDevMergeBuildDecision", d).outcome),
            str(cast("ModelOverdueDevMergeBuildDecision", d).reason),
        )
        for d in decisions
    ]


def test_four_old_merges_and_a_newer_build_ask_for_nothing(
    host: _Host, sender: _Sender
) -> None:
    for hours in (6, 5, 4, 3):
        host.merge(hours * _HOUR)
    host.job("a1", 2 * _HOUR, built_sha=host.merges[-1])
    outcomes = _outcomes(caller.check_overdue_dev_merges(force=True))
    assert outcomes == [("SATISFIED", "built_by_job")] * 4
    assert sender.commands == []
    assert sender.events == []


def test_one_merge_over_the_window_with_no_build_asks_once_for_the_head(
    host: _Host, sender: _Sender
) -> None:
    sha = host.merge(90 * _MINUTE)
    decisions = caller.check_overdue_dev_merges(force=True)
    assert _outcomes(decisions) == [("REQUEST", "overdue_no_build")]
    assert [c.git_ref for c in sender.commands] == [sha]
    assert [e.merge_sha for e in sender.events] == [sha]


def test_a_job_accepted_thirty_minutes_ago_coalesces_the_overdue_merge(
    host: _Host, sender: _Sender
) -> None:
    host.merge(90 * _MINUTE)
    host.job("b1", 30 * _MINUTE)
    outcomes = _outcomes(caller.check_overdue_dev_merges(force=True))
    assert outcomes == [("WAIT", "coalesced")]
    assert sender.commands == []
    assert sender.events == []
