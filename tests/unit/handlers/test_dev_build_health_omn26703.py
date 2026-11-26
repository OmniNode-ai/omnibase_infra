# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Each dev build's time to health and whether dev went back, read as an event on the bus (OMN-26703).

Goal item 1 of 2026-11-26: a build answers health within five minutes of its container start,
and dev never goes back to an older build than it ran. The fixture is the record of the
2026-11-26 hand load: the dev container started 19:24:53Z on omnibase_infra 9f77818ea, whose
parent build of 15:01Z ran 2b0c718bb (an ancestor of it).
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.handlers.dev_build_health.handler_dev_build_health import (
    HandlerDevBuildHealth,
)
from omnibase_infra.models.model_dev_build_health_request import (
    ModelDevBuildHealthRequest,
)
from omnibase_infra.models.model_dev_container_reading import ModelDevContainerReading
from omnibase_infra.nodes.node_dev_roll_events_effect import runtime_dev_roll_events
from omnibase_infra.nodes.node_dev_roll_events_effect.dev_roll_events import (
    DevRollEventsPublisher,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_CONTRACT = _ROOT / "src/omnibase_infra/nodes/node_dev_roll_events_effect/contract.yaml"
_STARTED = datetime(2026, 11, 26, 19, 24, 53, tzinfo=UTC)
_OLD = "2b0c718bb7a89f25e69b5adaaa0f3949cda86543"
_NEW = "9f77818ea0c96bc05ecffcbbfb621204bed0e6fa"
_LIMIT = 300


def _reading(
    *,
    healthy_after: timedelta | None = timedelta(seconds=14),
    status: str = "healthy",
    revision: str = _NEW,
    started: datetime = _STARTED,
) -> ModelDevContainerReading:
    return ModelDevContainerReading(
        container="omninode-runtime",
        container_digest="c" * 64,
        started_at=started,
        restart_count=0,
        exit_code=0,
        image_digest="sha256:" + "a" * 64,
        image_ref="omnibase-infra-omninode-runtime",
        image_revision=revision,
        health_status=status,
        first_healthy_at=None if healthy_after is None else started + healthy_after,
    )


def _judge(
    reading: ModelDevContainerReading,
    order: str,
    since: datetime | None = None,
    previous: str = _OLD,
) -> Any:
    return HandlerDevBuildHealth().handle(
        ModelDevBuildHealthRequest.model_validate(
            {
                "reading": reading,
                "previous_image_revision": previous,
                "revision_order": order,
                "previous_no_rollback_since": since,
                "limit_seconds": _LIMIT,
            }
        )
    )


def test_known_good_build_is_under_five_minutes_and_moved_forward() -> None:
    clock = _STARTED - timedelta(hours=4)
    event = _judge(_reading(), "forward", since=clock)
    assert event.health_seconds == 14.0
    assert event.under_limit is True
    assert event.rolled_back is False
    assert event.no_rollback_since == clock


def test_known_bad_build_past_five_minutes_is_not_under_limit() -> None:
    event = _judge(_reading(healthy_after=timedelta(minutes=6)), "forward")
    assert event.health_seconds == 360.0
    assert event.under_limit is False


def test_a_build_on_an_ancestor_of_the_previous_one_is_a_rollback_and_restarts_the_clock() -> (
    None
):
    event = _judge(_reading(), "back", since=_STARTED - timedelta(hours=20))
    assert event.rolled_back is True
    assert event.no_rollback_since == _STARTED


def test_the_first_build_read_starts_the_clock() -> None:
    event = _judge(_reading(), "first", previous="")
    assert event.rolled_back is False
    assert event.no_rollback_since == _STARTED


@pytest.mark.parametrize("order", ["unrelated", "unread", "same"])
def test_an_order_git_cannot_call_back_is_never_a_rollback(order: str) -> None:
    assert _judge(_reading(), order, since=_STARTED).rolled_back is False


def test_no_passing_check_read_is_unread_never_a_pass() -> None:
    event = _judge(_reading(healthy_after=None, status="unhealthy"), "forward")
    assert event.health_seconds is None
    assert event.under_limit is None


def test_a_starting_container_publishes_nothing_yet() -> None:
    assert _judge(_reading(healthy_after=None, status="starting"), "forward") is None


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env=scrub_git_location_env(os.environ),
    ).stdout.strip()


def test_revision_order_is_read_from_git_history(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@example.invalid")
    _git(tmp_path, "config", "user.name", "t")
    shas = []
    for name in ("a", "b"):
        (tmp_path / name).write_text(name)
        _git(tmp_path, "add", name)
        _git(tmp_path, "commit", "-q", "-m", name)
        shas.append(_git(tmp_path, "rev-parse", "HEAD"))
    _git(tmp_path, "checkout", "-q", "-b", "side", shas[0])
    (tmp_path / "c").write_text("c")
    _git(tmp_path, "add", "c")
    _git(tmp_path, "commit", "-q", "-m", "c")
    side = _git(tmp_path, "rev-parse", "HEAD")
    order = runtime_dev_roll_events.revision_order
    assert order(tmp_path, "", shas[1], 30.0) == "first"
    assert order(tmp_path, shas[1], shas[1], 30.0) == "same"
    assert order(tmp_path, shas[0], shas[1], 30.0) == "forward"
    assert order(tmp_path, shas[1], shas[0], 30.0) == "back"
    assert order(tmp_path, shas[1], side, 30.0) == "unrelated"
    assert order(tmp_path, shas[1], "f" * 40, 30.0) == "unread"


def test_the_contract_declares_the_topic_the_limit_and_the_operation() -> None:
    contract = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    assert contract["topics"]["dev_build_health"] == (
        "onex.evt.omnimarket.dev-build-health.v1"
    )
    assert contract["settings"]["build_health_limit_seconds"] == _LIMIT
    operations = {h["operation"] for h in contract["handler_routing"]["handlers"]}
    assert "dev_roll.read_build_health" in operations


def test_publisher_sends_the_event_on_the_contract_topic() -> None:
    topics = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))["topics"]
    sent: list[tuple[str, str, dict[str, Any]]] = []
    publisher = DevRollEventsPublisher(
        lambda topic, key, payload: sent.append((topic, key, payload)) or True, topics
    )
    event = _judge(_reading(), "forward")
    assert publisher.build_health(event) is True
    assert sent[0][0] == topics["dev_build_health"]
    assert sent[0][2]["under_limit"] is True


class _Publisher:
    def __init__(self) -> None:
        self.health: list[Any] = []

    def container_recreated(self, event: Any) -> bool:
        return True

    def build_health(self, event: Any) -> bool:
        self.health.append(event)
        return True

    def recreate_violation(self, violation: Any) -> bool:
        return True


def _docker(
    *,
    started: datetime,
    health: dict[str, Any],
    revision: str,
    ancestors: set[tuple[str, str]],
) -> Any:
    def run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        def done(out: str, code: int = 0) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(cmd, code, out, "")

        if cmd[:3] == ["docker", "ps", "-aq"]:
            return done("cid1\n")
        if cmd[:3] == ["docker", "image", "inspect"]:
            return done(revision + "\n")
        if cmd[:2] == ["docker", "inspect"]:
            raw = {
                "Name": "/omninode-runtime",
                "Id": "c" * 64,
                "RestartCount": 0,
                "Image": "sha256:" + "a" * 64,
                "Config": {"Image": "omnibase-infra-omninode-runtime"},
                "State": {
                    "StartedAt": started.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
                    "ExitCode": 0,
                    "Health": health,
                },
            }
            return done(json.dumps([raw]))
        if cmd[0] == "git":
            older, newer = cmd[-2], cmd[-1]
            return done("", 0 if (older, newer) in ancestors else 1)
        raise AssertionError(cmd)

    return run


def test_the_agent_tick_publishes_one_event_per_start_and_carries_the_clock(
    tmp_path: Path,
) -> None:
    publisher = _Publisher()
    passing = {
        "Status": "healthy",
        "Log": [
            {
                "End": (_STARTED + timedelta(seconds=14)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "ExitCode": 0,
            }
        ],
    }
    observe = runtime_dev_roll_events.observe_dev_container

    def tick(run: Any) -> None:
        observe(tmp_path, "omnibase-infra", _ROOT, publisher=publisher, run=run)

    old_start = _STARTED - timedelta(hours=4)
    tick(
        _docker(
            started=old_start,
            health={
                **passing,
                "Log": [
                    {
                        "End": (old_start + timedelta(seconds=20)).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        "ExitCode": 0,
                    }
                ],
            },
            revision=_OLD,
            ancestors=set(),
        )
    )
    tick(_docker(started=old_start, health=passing, revision=_OLD, ancestors=set()))
    assert len(publisher.health) == 1
    assert publisher.health[0].revision_order == "first"
    clock = publisher.health[0].no_rollback_since
    tick(
        _docker(
            started=_STARTED, health=passing, revision=_NEW, ancestors={(_OLD, _NEW)}
        )
    )
    assert [e.revision_order for e in publisher.health] == ["first", "forward"]
    assert publisher.health[1].no_rollback_since == clock
    assert publisher.health[1].under_limit is True
    tick(
        _docker(
            started=_STARTED, health=passing, revision=_NEW, ancestors={(_OLD, _NEW)}
        )
    )
    assert len(publisher.health) == 2
    later = _STARTED + timedelta(hours=1)
    tick(
        _docker(
            started=later,
            health={
                "Status": "healthy",
                "Log": [
                    {
                        "End": (later + timedelta(seconds=9)).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        "ExitCode": 0,
                    }
                ],
            },
            revision=_OLD,
            ancestors={(_OLD, _NEW)},
        )
    )
    assert publisher.health[2].rolled_back is True
    assert publisher.health[2].no_rollback_since == later
