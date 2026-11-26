# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One overdue check asks for one build of the current dev head (OMN-26316).

THE RECORD
----------
2026-11-25 on h201: the overdue check asked for a build per old merge inside its
24-hour lookback (3a527226 at 07:15:07Z, c0622a1d at 07:28:59Z, 5be4cfa4 at 07:42:13Z,
4f27dcfa at 07:54:36Z), every one of them at the same ref ``f553b1d6``. One build of the
head carries every merge below it, so one request per check is enough, and a head that the
running build already contains is never requested.
"""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.handlers.handler_dev_merge_build_overdue import request_id_for
from omnibase_infra.models.model_dev_merge_build_command import (
    ModelDevMergeBuildCommand,
)
from omnibase_infra.models.model_dev_merge_build_overdue_event import (
    ModelDevMergeBuildOverdueEvent,
)
from omnibase_infra.models.model_overdue_dev_merge_build_decision import (
    ModelOverdueDevMergeBuildDecision,
)
from omnibase_infra.models.model_overdue_dev_merge_job import ModelOverdueDevMergeJob
from omnibase_infra.nodes.node_deploy_job_reading_effect import (
    runtime_deploy_job_reading as caller,
)

pytestmark = pytest.mark.unit

_SETTINGS = {
    "dev_merge_build_runtime_lane": "dev",
    "dev_merge_build_overdue_topic": "topic-from-settings",
    "dev_merge_build_command_topic": "command-topic-from-settings",
    "build_decision_git_diff_timeout_seconds": 10,
}


class _Sender:
    def __init__(self) -> None:
        self.events: list[ModelDevMergeBuildOverdueEvent] = []
        self.commands: list[ModelDevMergeBuildCommand] = []

    def publish_event(self, event: ModelDevMergeBuildOverdueEvent) -> None:
        self.events.append(event)

    def send_command(self, command: ModelDevMergeBuildCommand) -> str:
        self.commands.append(command)
        return "published"


def _history(tmp_path: Path) -> tuple[SimpleNamespace, list[str]]:
    repo = tmp_path / "clone"
    state = tmp_path / "state"
    repo.mkdir()
    state.mkdir()

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
            env=scrub_git_location_env(os.environ),
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    shas = []
    for number in (1, 2, 3):
        git("commit", "--allow-empty", "-q", "-m", f"merge (#{number})")
        shas.append(git("rev-parse", "HEAD"))
    return SimpleNamespace(STATE_DIR=str(state), REPO_DIR=str(repo)), shas


def _decision(sha: str) -> ModelOverdueDevMergeBuildDecision:
    return ModelOverdueDevMergeBuildDecision.model_validate(
        {
            "outcome": "REQUEST",
            "reason": "overdue_no_build",
            "merge_sha": sha,
            "due_at": datetime(2026, 11, 25, 6, 0, tzinfo=UTC),
            "request_id": request_id_for("omnibase_infra", sha),
        }
    )


@pytest.fixture
def sender(monkeypatch: pytest.MonkeyPatch) -> _Sender:
    for relative in caller._OVERDUE_MODULES:
        caller._load(relative)
    fake = _Sender()
    monkeypatch.setattr(caller, "AgentDevMergeSender", lambda *_a: fake)
    monkeypatch.setattr(caller, "_TRACKED_HEAD", "HEAD")
    return fake


def test_three_overdue_merges_get_one_request_for_the_head(
    tmp_path: Path, sender: _Sender
) -> None:
    execution, shas = _history(tmp_path)
    overdue = [_decision(sha) for sha in shas]
    subjects = {sha: f"merge (#{number})" for number, sha in enumerate(shas, 1)}
    for _ in range(2):
        caller._request_overdue_builds(execution, _SETTINGS, overdue, subjects)
    assert len(sender.commands) == 1
    assert sender.commands[0].git_ref == shas[-1]
    recorded = (Path(execution.STATE_DIR) / caller._REQUESTED_KEYS).read_text().split()
    assert recorded == [str(request_id_for("omnibase_infra", sha)) for sha in shas]


def test_a_head_the_running_build_contains_is_never_requested(
    tmp_path: Path, sender: _Sender
) -> None:
    execution, shas = _history(tmp_path)
    subprocess.run(
        ["git", "-C", execution.REPO_DIR, "reset", "-q", "--hard", shas[1]],
        check=True,
        env=scrub_git_location_env(os.environ),
    )
    running = ModelOverdueDevMergeJob.model_validate(
        {
            "job_id": "00000000-0000-5000-8000-000000000009",
            "runtime_lane": "dev",
            "accepted_at": "2026-11-25T07:00:50Z",
            "status": "success",
            "git_ref": "origin/dev",
            "requested_by": "agent/idle-restore",
            "built_sha": shas[2],
        }
    )
    caller._request_overdue_builds(
        execution,
        _SETTINGS,
        [_decision(shas[0])],
        {shas[0]: "merge (#1)"},
        context=caller._OverdueContext((running,), {}),
    )
    assert sender.commands == []
