# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The overdue check reads every repository the dev build composes (OMN-26316).

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
from omnibase_infra.tools.contract_topic_extractor import ContractTopicExtractor

pytestmark = pytest.mark.unit

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
        REPO_DIR=str(infra),
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


def test_an_omnimarket_merge_with_no_dev_build_reads_overdue(
    lane: SimpleNamespace,
) -> None:
    decisions = caller.check_overdue_dev_merges(force=True)
    assert decisions is not None
    by_sha = {d.merge_sha: d for d in decisions}
    assert by_sha[lane.market_sha].outcome == "REQUEST"
    assert by_sha[lane.market_sha].reason == "overdue_no_build"


def test_the_omnimarket_merge_gets_its_event_and_one_build_request(
    lane: SimpleNamespace,
) -> None:
    caller.check_overdue_dev_merges(force=True)
    assert [e.repository for e in lane.sender.events] == ["omnimarket"]
    assert lane.sender.events[0].pr_number == 6872
    # git_ref is a commit of the agent's own omnibase_infra clone, never the sibling's sha:
    # the agent resets that clone to it (2026-11-26: "Git reset --hard 3026421861f6 failed").
    assert [(c.repo, c.pr_number, c.git_ref) for c in lane.sender.commands] == [
        ("omnimarket", 6872, lane.infra_sha)
    ]
    assert lane.sender.commands[0].git_ref != lane.market_sha


def test_a_dev_job_accepted_after_the_merge_satisfies_it(
    lane: SimpleNamespace,
) -> None:
    built = "a" * 40
    record = {
        "correlation_id": "00000000-0000-5000-8000-000000000001",
        "accepted_at": (_NOW - timedelta(minutes=10)).isoformat(),
        "status": "success",
        "command": {"runtime_lane": "dev", "git_ref": "origin/dev"},
        "lineage": {"resolved_ref": built},
    }
    (Path(lane.execution.STATE_DIR) / "job.json").write_text(json.dumps(record))
    decisions = caller.check_overdue_dev_merges(force=True)
    assert decisions is not None
    by_sha = {d.merge_sha: d for d in decisions}
    assert by_sha[lane.market_sha].outcome == "SATISFIED"
    assert lane.sender.commands == []


def test_the_repositories_come_from_the_agents_declared_composition() -> None:
    source = (
        _REPO / "scripts" / "deploy-agent" / "deploy_agent" / "idle_converge.py"
    ).read_text(encoding="utf-8")
    assert 'FOLLOWED_REPOSITORIES: Final = ("omnimarket", "omnibase_infra")' in source
    body = Path(caller.__file__).read_text(encoding="utf-8")
    assert "idle_converge" in body and "FOLLOWED_REPOSITORIES" in body


def test_the_overdue_topic_is_declared_for_provisioning() -> None:
    entries = ContractTopicExtractor().extract_all(
        contracts_root=_REPO / "src" / "omnibase_infra" / "nodes",
        skill_manifests_roots=[
            _REPO / "src" / "omnibase_infra" / "cli",
            _REPO / "src" / "omnibase_infra" / "services",
        ],
    )
    declared = {entry.topic for entry in entries}
    settings = caller._idle_settings()
    assert settings["dev_merge_build_overdue_topic"] in declared
