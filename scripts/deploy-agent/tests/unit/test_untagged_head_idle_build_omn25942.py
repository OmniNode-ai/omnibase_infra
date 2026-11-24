# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-25942: idle restore and idle converge build a dev head only once its dev-merge tag exists.

At 2026-11-24T02:46:16Z the dev agent's idle restore (job 7dfb1bc9) fetched the dev head
5cafc841f7de 44 seconds after it was pushed. The Dev Merge Tag workflow for that push ran from
02:45:32Z to 02:48:12Z (a median run takes 15.5 s), so the tag did not exist yet; the staged
tree carried no dev-merge tag, the dev-lane first-party audit printed ``FAIL omnibase-infra:
branch (5cafc841f7de) staged from branch dev`` and the build failed with
``runtime_image_build_errored``. The agent now refuses an idle build of a head that no dev tag
on origin names, writes one ledger row per head, and builds the head on the first check after
the tag exists. The audit and its dev-lane context are untouched.

Select the acceptance criteria with ``-k untagged_head_idle_waits_for_tag`` (the build),
``-k untagged_head_ledger_row`` (the row), ``-k signed_request_bypasses_untagged_head`` and
``-k untagged_head_incident_replay`` (the recorded job 7dfb1bc9).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.agent import JOB_READING, DeployAgent
from deploy_agent.executor import load_dev_boot_guard
from deploy_agent.idle_converge import (
    IDLE_CONVERGE_REQUESTER,
    IDLE_RESTORE_REQUESTER,
    EnumRuntimePresence,
    ModelDevTag,
)

from .test_failed_head_gate_omn26493 import (
    _agent,
    _agent_jobs,
    _cells,
    _fake_append,
    _follow_infra_elsewhere,
    _signed,
)

pytestmark = pytest.mark.unit

reading = load_dev_boot_guard(JOB_READING)

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "omn25942"
    / "recorded_untagged_head_7dfb1bc9.json"
)
RECORDED = json.loads(FIXTURE.read_text(encoding="utf-8"))
HEAD = RECORDED["head"]
PREVIOUS = RECORDED["previous_head"]

#: Job 7dfb1bc9 was accepted here, 44 s after the head was pushed and 2 m 40 s before its tag.
FIRST_CHECK = datetime(2026, 11, 24, 2, 46, 16, tzinfo=UTC)
#: The first check after the Dev Merge Tag run for the head ended (02:48:12Z).
TAG_CHECK = datetime(2026, 11, 24, 2, 48, 16, tzinfo=UTC)


def _tag(name: str, sha: str) -> ModelDevTag:
    stamp = datetime.strptime(name.split("-")[1], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    return ModelDevTag(name=name, sha=sha, cut_at=stamp)


#: Origin's dev tags before the Dev Merge Tag run for the head ended: the previous head's last.
TAGS_BEFORE = [_tag(PREVIOUS["tag"], PREVIOUS["sha"])]
#: And after: the head's own tag, as ``git ls-remote --tags`` names it.
TAGS_AFTER = [*TAGS_BEFORE, _tag(RECORDED["tag_for_head"], HEAD)]


@pytest.fixture
def ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The agent's append route and its state directory; the file the rows land in."""
    command, sink = _fake_append(tmp_path)
    settings = reading.load_settings(agent_mod)
    monkeypatch.setenv(settings["append_command_env"], command)
    monkeypatch.setenv(settings["host_name_env"], "lab-host")
    (tmp_path / "agent-state").mkdir(exist_ok=True)
    reading._refs.clear()
    reading._not_routed_read.clear()
    return sink


def _untagged_rows(sink: Path) -> list[dict[str, str]]:
    if not sink.exists():
        return []
    return [
        _cells(line)
        for line in sink.read_text(encoding="utf-8").splitlines()
        if "src=build-decision" in line and "reason=untagged_head" in line
    ]


def _down_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tags: list[list[ModelDevTag] | None],
    *,
    head: str = HEAD,
) -> DeployAgent:
    """A lane whose runtime is down at ``head``; ``tags[0]`` is what origin names now."""
    agent = _agent(tmp_path, monkeypatch, now=FIRST_CHECK, head=head)
    agent._idle_read_dev_tags = lambda repo: tags[0]
    return agent


def _tick(agent: DeployAgent, now: datetime) -> None:
    agent._idle_converge_now = lambda: now
    agent._idle_converge_last_check = None
    agent._maybe_idle_converge()


class TestUntaggedHeadIdleWaitsForTag:
    def test_untagged_head_idle_waits_for_tag_restore(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        agent = _down_agent(tmp_path, monkeypatch, [TAGS_BEFORE])

        _tick(agent, FIRST_CHECK)

        assert _agent_jobs(tmp_path, IDLE_RESTORE_REQUESTER) == [], (
            "the idle restore built a head whose dev-merge tag did not exist yet"
        )

    def test_untagged_head_idle_waits_for_tag_restore_builds_once_tagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        tags = [TAGS_BEFORE]
        agent = _down_agent(tmp_path, monkeypatch, tags)
        _tick(agent, FIRST_CHECK)
        tags[0] = TAGS_AFTER

        _tick(agent, TAG_CHECK)

        (job,) = _agent_jobs(tmp_path, IDLE_RESTORE_REQUESTER)
        assert job.command["git_ref"] == "origin/dev"

    def test_untagged_head_idle_waits_for_tag_converge(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        agent = _down_agent(tmp_path, monkeypatch, [TAGS_BEFORE])
        agent._idle_read_runtime_presence = lambda: (
            EnumRuntimePresence.RUNNING,
            "omninode-runtime is running",
        )
        _follow_infra_elsewhere(agent)

        _tick(agent, FIRST_CHECK)

        assert _agent_jobs(tmp_path, IDLE_CONVERGE_REQUESTER) == [], (
            "the idle converge built a head whose dev-merge tag did not exist yet"
        )

    def test_untagged_head_idle_waits_for_tag_converge_builds_once_tagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        tags = [TAGS_BEFORE]
        agent = _down_agent(tmp_path, monkeypatch, tags)
        agent._idle_read_runtime_presence = lambda: (
            EnumRuntimePresence.RUNNING,
            "omninode-runtime is running",
        )
        _follow_infra_elsewhere(agent)
        _tick(agent, FIRST_CHECK)
        tags[0] = TAGS_AFTER

        _tick(agent, TAG_CHECK)

        assert len(_agent_jobs(tmp_path, IDLE_CONVERGE_REQUESTER)) == 1

    def test_untagged_head_idle_waits_for_tag_builds_a_tagged_head_at_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        agent = _down_agent(tmp_path, monkeypatch, [TAGS_AFTER])

        _tick(agent, FIRST_CHECK)

        assert len(_agent_jobs(tmp_path, IDLE_RESTORE_REQUESTER)) == 1
        assert _untagged_rows(ledger) == []

    def test_untagged_head_idle_waits_for_tag_with_tags_unread_builds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        """An unread tag list proves nothing about the head; the audit stays the backstop."""
        agent = _down_agent(tmp_path, monkeypatch, [None])

        _tick(agent, FIRST_CHECK)

        assert len(_agent_jobs(tmp_path, IDLE_RESTORE_REQUESTER)) == 1

    def test_untagged_head_idle_waits_for_tag_names_the_head_not_a_prefix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        """A tag on the head's parent is not a tag on the head."""
        agent = _down_agent(tmp_path, monkeypatch, [TAGS_BEFORE], head="c" * 40)

        _tick(agent, FIRST_CHECK)

        assert _agent_jobs(tmp_path, IDLE_RESTORE_REQUESTER) == []


class TestUntaggedHeadLedgerRow:
    def test_untagged_head_ledger_row_names_head_and_reason(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        agent = _down_agent(tmp_path, monkeypatch, [TAGS_BEFORE])

        _tick(agent, FIRST_CHECK)

        (row,) = _untagged_rows(ledger)
        assert row["trigger_sha"] == HEAD
        assert row["result"] == "refused"
        assert row["repo"] == "omnibase_infra"

    def test_untagged_head_ledger_row_is_one_per_head_not_one_per_tick(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        agent = _down_agent(tmp_path, monkeypatch, [TAGS_BEFORE])

        for minute in range(30):
            _tick(agent, FIRST_CHECK + timedelta(minutes=minute))

        assert len(_untagged_rows(ledger)) == 1

    def test_untagged_head_ledger_row_is_not_repeated_after_a_restart(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        first = _down_agent(tmp_path, monkeypatch, [TAGS_BEFORE])
        _tick(first, FIRST_CHECK)
        assert len(_untagged_rows(ledger)) == 1

        restarted = _down_agent(tmp_path, monkeypatch, [TAGS_BEFORE])
        _tick(restarted, FIRST_CHECK + timedelta(minutes=1))

        assert len(_untagged_rows(ledger)) == 1

    def test_untagged_head_ledger_row_is_one_for_restore_and_converge_together(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        agent = _down_agent(tmp_path, monkeypatch, [TAGS_BEFORE])
        _follow_infra_elsewhere(agent)
        _tick(agent, FIRST_CHECK)
        agent._idle_read_runtime_presence = lambda: (
            EnumRuntimePresence.RUNNING,
            "omninode-runtime is running",
        )
        _tick(agent, FIRST_CHECK + timedelta(minutes=1))

        assert len(_untagged_rows(ledger)) == 1

    def test_untagged_head_ledger_row_is_one_for_each_head(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        agent = _down_agent(tmp_path, monkeypatch, [TAGS_BEFORE])
        _tick(agent, FIRST_CHECK)
        agent._idle_read_head_ref = lambda repo: "e" * 40
        _tick(agent, FIRST_CHECK + timedelta(minutes=1))

        assert [row["trigger_sha"] for row in _untagged_rows(ledger)] == [
            HEAD,
            "e" * 40,
        ]


class TestSignedRequestBypassesUntaggedHead:
    @pytest.mark.parametrize(
        "requester",
        [
            "operator/pinned-fix/devwindow2-2215",
            "lane/idleskip-0030",
            "gha/omnibase_infra/runtime-rebuild",
        ],
    )
    def test_signed_request_bypasses_untagged_head(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ledger: Path,
        requester: str,
    ) -> None:
        agent = _down_agent(tmp_path, monkeypatch, [TAGS_BEFORE])
        _tick(agent, FIRST_CHECK)
        assert _agent_jobs(tmp_path, IDLE_RESTORE_REQUESTER) == []

        cmd = _signed(requester)
        agent.job_store.accept(cmd.correlation_id, cmd.model_dump(mode="json"))
        agent._execute_command(cmd)

        signed = _agent_jobs(tmp_path, requester)
        assert [job.correlation_id for job in signed] == [cmd.correlation_id]
        assert signed[0].status == "success"


class TestUntaggedHeadIncidentReplay:
    def test_untagged_head_incident_replay(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        """Job 7dfb1bc9: the head pushed at 02:45:32Z, its tag visible after 02:48:12Z.

        The agent checks once a minute from 02:46:16Z. It builds nothing until the tag
        exists, then starts one restore; without the gate it starts the restore at 02:46:16Z.
        """
        assert RECORDED["job"]["requested_by"] == IDLE_RESTORE_REQUESTER
        tag_run_ended = datetime.fromisoformat(
            RECORDED["dev_merge_tag_run"]["updated_at"]
        )
        tags = [TAGS_BEFORE]
        agent = _down_agent(tmp_path, monkeypatch, tags)

        started: list[datetime] = []
        for minute in range(10):
            now = FIRST_CHECK + timedelta(minutes=minute)
            if now > tag_run_ended:
                tags[0] = TAGS_AFTER
            before = len(_agent_jobs(tmp_path, IDLE_RESTORE_REQUESTER))
            _tick(agent, now)
            if len(_agent_jobs(tmp_path, IDLE_RESTORE_REQUESTER)) > before:
                started.append(now)

        assert started == [TAG_CHECK]
        assert len(_untagged_rows(ledger)) == 1
        accepted: Any = _agent_jobs(tmp_path, IDLE_RESTORE_REQUESTER)[0]
        assert accepted.command["git_ref"] == RECORDED["job"]["git_ref"]
