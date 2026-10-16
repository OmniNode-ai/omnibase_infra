# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19998: lab instances follow omnimarket's newest dev tag, beat, and alarm.

Operator RULING 2026-10-16T09:35:49Z: in the lab, omnimarket deploys from a
tag cut on each dev merge, not from a PyPI release, and that is part of the
process. The deploy agent's followed omnimarket head is the newest
``dev-<UTC YYYYMMDDHHMMSS>-<sha12>`` tag on origin; every instance publishes a
``ModelDevTagHeartbeat`` against it, and raises the alarm once the lane has run
behind that tag for two hours.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer
from deploy_agent import agent as agent_mod
from deploy_agent.events import TOPIC_DEV_TAG_HEARTBEAT, ModelDevTagHeartbeat
from deploy_agent.health import create_health_app
from deploy_agent.idle_converge import (
    DEV_TAG_ALARM_AFTER,
    DEV_TAG_HEARTBEAT_INTERVAL,
    ModelDevTag,
    dev_tag_behind_since,
    dev_tag_heartbeat,
    read_dev_tags,
)
from deploy_agent.job_state import JobStore
from deploy_agent.routing import AGENT_CLONE_ROOT
from deploy_agent.tracking_ref import DEV_TAG_FOLLOWED_REPOSITORIES

from .test_idle_converge_omn19509 import _agent

pytestmark = pytest.mark.unit

OLD = "1" * 40
MID = "2" * 40
NEW = "3" * 40
OLD_CUT = datetime(2026, 10, 16, 6, 0, tzinfo=UTC)
MID_CUT = datetime(2026, 10, 16, 7, 0, tzinfo=UTC)
NEW_CUT = datetime(2026, 10, 16, 9, 0, tzinfo=UTC)


def _tag(sha: str, cut: datetime) -> ModelDevTag:
    return ModelDevTag(name=f"dev-{cut:%Y%m%d%H%M%S}-{sha[:12]}", sha=sha, cut_at=cut)


TAGS = [_tag(OLD, OLD_CUT), _tag(MID, MID_CUT), _tag(NEW, NEW_CUT)]


class TestReadDevTags:
    def test_reads_origin_with_ls_remote_and_parses_with_the_build_parser(
        self,
    ) -> None:
        calls: list[list[str]] = []
        out = "\n".join(f"{t.sha}\trefs/tags/{t.name}" for t in reversed(TAGS))

        def run(argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, out, "")

        tags = read_dev_tags(Path("/clone"), AGENT_CLONE_ROOT, run=run)

        assert calls == [
            [
                "git",
                "-C",
                "/clone",
                "ls-remote",
                "--tags",
                "origin",
                "refs/tags/dev-*",
            ]
        ]
        assert tags == TAGS

    def test_a_failed_read_is_unread(self) -> None:
        def run(argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 128, "", "no origin")

        assert read_dev_tags(Path("/clone"), AGENT_CLONE_ROOT, run=run) is None


class TestBehindSince:
    NOW = NEW_CUT + timedelta(hours=3)

    def test_on_the_newest_tag_is_not_behind(self) -> None:
        assert dev_tag_behind_since(TAGS, NEW, MID_CUT, self.NOW) is None

    def test_an_older_tag_is_behind_from_the_next_tag_cut(self) -> None:
        assert dev_tag_behind_since(TAGS, OLD, None, self.NOW) == MID_CUT
        assert dev_tag_behind_since(TAGS, MID, None, self.NOW) == NEW_CUT

    def test_a_commit_that_is_no_tag_is_behind_from_the_first_read(self) -> None:
        assert dev_tag_behind_since(TAGS, "f" * 40, None, self.NOW) == self.NOW
        assert dev_tag_behind_since(TAGS, "f" * 40, OLD_CUT, self.NOW) == OLD_CUT

    def test_a_build_ahead_of_the_newest_tag_is_not_behind(self) -> None:
        assert (
            dev_tag_behind_since(
                TAGS, "f" * 40, OLD_CUT, self.NOW, running_contains_newest=True
            )
            is None
        )

    def test_an_unread_ref_or_no_tags_changes_nothing(self) -> None:
        assert dev_tag_behind_since(TAGS, None, OLD_CUT, self.NOW) == OLD_CUT
        assert dev_tag_behind_since([], OLD, None, self.NOW) is None


class TestHeartbeat:
    def _beat(
        self, running: str | None, behind: datetime | None, now: datetime
    ) -> ModelDevTagHeartbeat:
        return dev_tag_heartbeat(
            instance="dev-201",
            repository="omnimarket",
            runtime_container="omninode-runtime",
            running_ref=running,
            tags=TAGS,
            behind_since=behind,
            now=now,
        )

    def test_the_alarm_is_two_hours(self) -> None:
        assert timedelta(hours=2) == DEV_TAG_ALARM_AFTER

    def test_behind_past_two_hours_alarms(self) -> None:
        beat = self._beat(MID, NEW_CUT, NEW_CUT + timedelta(hours=2, minutes=1))
        assert beat.alarm is True
        assert beat.newest_tag == TAGS[-1].name
        assert beat.newest_tag_sha == NEW
        assert beat.lag_seconds == (2 * 60 + 1) * 60
        assert beat.alarm_after_seconds == 7200

    def test_behind_inside_two_hours_does_not_alarm(self) -> None:
        beat = self._beat(MID, NEW_CUT, NEW_CUT + timedelta(minutes=119))
        assert beat.alarm is False

    def test_on_the_newest_tag_beats_without_alarm(self) -> None:
        beat = self._beat(NEW, None, NEW_CUT + timedelta(days=2))
        assert beat.alarm is False
        assert beat.lag_seconds == 0

    def test_an_unread_running_ref_never_alarms(self) -> None:
        beat = self._beat(None, OLD_CUT, NEW_CUT + timedelta(days=2))
        assert beat.alarm is False
        assert beat.running_ref is None

    def test_the_topic_is_an_event_of_the_deploy_producer(self) -> None:
        assert TOPIC_DEV_TAG_HEARTBEAT == "onex.evt.deploy.dev-tag-heartbeat.v1"


class TestInTheAgent:
    def test_omnimarket_is_the_tag_followed_repository(self) -> None:
        assert DEV_TAG_FOLLOWED_REPOSITORIES == ("omnimarket",)

    def test_the_followed_head_is_the_newest_tag(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("OMNI_HOME", str(tmp_path))
        monkeypatch.setattr(agent_mod, "read_dev_tags", lambda clone, root: TAGS)
        monkeypatch.setattr(agent_mod, "read_dev_head", lambda clone: "9" * 40)
        assert agent_mod._read_dev_head("omnimarket") == NEW

    def test_with_no_tag_yet_the_head_is_the_branch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("OMNI_HOME", str(tmp_path))
        monkeypatch.setattr(agent_mod, "read_dev_tags", lambda clone, root: [])
        monkeypatch.setattr(agent_mod, "read_dev_head", lambda clone: "9" * 40)
        assert agent_mod._read_dev_head("omnimarket") == "9" * 40

    def test_an_unread_tag_list_is_an_unread_head(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("OMNI_HOME", str(tmp_path))
        monkeypatch.setattr(agent_mod, "read_dev_tags", lambda clone, root: None)
        monkeypatch.setattr(agent_mod, "read_dev_head", lambda clone: "9" * 40)
        assert agent_mod._read_dev_head("omnimarket") is None

    def test_every_instance_beats_and_alarms(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        agent = _agent(tmp_path, monkeypatch, omnimarket_to_202=False)
        published: list[dict[str, Any]] = []
        monkeypatch.setattr(
            agent_mod,
            "publish_dev_tag_heartbeat",
            lambda payload, config: published.append(payload) or True,
        )
        now = NEW_CUT + timedelta(hours=2, minutes=5)
        agent._idle_converge_now = lambda: now
        agent._idle_read_running_ref = lambda repo: MID
        agent._idle_read_dev_tags = lambda repo: TAGS
        agent._idle_running_contains = lambda repo, ancestor, running: False

        with caplog.at_level("ERROR", logger=agent_mod.logger.name):
            agent._maybe_idle_converge()

        assert len(published) == 1
        beat = ModelDevTagHeartbeat.model_validate(published[0])
        assert beat.repository == "omnimarket"
        assert beat.running_ref == MID
        assert beat.behind_since == NEW_CUT
        assert beat.alarm is True
        assert "friction_type=dev_tag_lag" in caplog.text
        assert agent._latest_dev_tag_heartbeat()["omnimarket"]["alarm"] is True

    def test_a_down_lane_beats_unread_and_reads_no_revision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(tmp_path, monkeypatch, omnimarket_to_202=False)
        published: list[dict[str, Any]] = []
        monkeypatch.setattr(
            agent_mod,
            "publish_dev_tag_heartbeat",
            lambda payload, config: published.append(payload) or True,
        )
        agent._idle_read_runtime_presence = lambda: (
            agent_mod.EnumRuntimePresence.DOWN,
            "omninode-runtime is exited",
        )
        agent._idle_restore_last_attempt = NEW_CUT  # no restore this check

        def no_running_ref(repo: str) -> str | None:
            raise AssertionError("the heartbeat read a revision on a down lane")

        agent._idle_read_running_ref = no_running_ref
        agent._idle_read_dev_tags = lambda repo: TAGS
        agent._idle_running_contains = lambda repo, ancestor, running: False
        agent._idle_converge_now = lambda: NEW_CUT + timedelta(minutes=5)

        agent._maybe_idle_converge()

        assert len(published) == 1
        assert published[0]["running_ref"] is None
        assert published[0]["alarm"] is False

    def test_the_heartbeat_is_throttled_but_an_alarm_change_is_not(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(tmp_path, monkeypatch, omnimarket_to_202=False)
        published: list[dict[str, Any]] = []
        monkeypatch.setattr(
            agent_mod,
            "publish_dev_tag_heartbeat",
            lambda payload, config: published.append(payload) or True,
        )
        running = {"ref": NEW}
        agent._idle_read_running_ref = lambda repo: running["ref"]
        agent._idle_read_dev_tags = lambda repo: TAGS
        agent._idle_running_contains = lambda repo, ancestor, running: False
        clock = {"now": NEW_CUT + timedelta(minutes=1)}
        agent._idle_converge_now = lambda: clock["now"]

        def check() -> None:
            agent._idle_converge_last_check = None
            agent._maybe_idle_converge()

        check()  # first beat
        clock["now"] += timedelta(minutes=1)
        check()  # inside the interval, same alarm: no beat
        assert len(published) == 1
        clock["now"] += DEV_TAG_HEARTBEAT_INTERVAL
        check()  # the interval passed: a beat
        assert len(published) == 2
        running["ref"] = "f" * 40  # a hand build, off every tag
        agent._dev_tag_behind_since["omnimarket"] = clock["now"] - timedelta(hours=3)
        clock["now"] += timedelta(minutes=1)
        check()  # the alarm changed: a beat at once
        assert len(published) == 3
        assert published[-1]["alarm"] is True


async def test_the_heartbeat_is_served(tmp_path: Path) -> None:
    beat = dev_tag_heartbeat(
        instance="dev-201",
        repository="omnimarket",
        runtime_container="omninode-runtime",
        running_ref=NEW,
        tags=TAGS,
        behind_since=None,
        now=NEW_CUT,
    )
    app = create_health_app(
        job_store=JobStore(tmp_path / "jobs"),
        get_agent_state=lambda: "idle",
        get_dev_tag_heartbeat=lambda: {"omnimarket": beat.model_dump(mode="json")},
    )
    async with TestClient(TestServer(app)) as client:
        response = await client.get("/dev-tag-heartbeat")
        body = await response.json()
    assert body["heartbeat"]["omnimarket"]["newest_tag_sha"] == NEW
    assert body["heartbeat"]["omnimarket"]["alarm"] is False
