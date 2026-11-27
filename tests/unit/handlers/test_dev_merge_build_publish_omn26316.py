# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The overdue dev merge build is published and requested once, on the dev lane only (OMN-26316).

A fake sender stands in for the signed build request path and the event publish: nothing
here reaches a bus.
"""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_core.utils.util_run_coro_sync import run_coro_sync
from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.handlers.handler_dev_merge_build_overdue import request_id_for
from omnibase_infra.handlers.handler_dev_merge_build_publish import (
    HandlerDevMergeBuildPublish,
)
from omnibase_infra.models.model_dev_merge_build_command import (
    ModelDevMergeBuildCommand,
)
from omnibase_infra.models.model_dev_merge_build_overdue_event import (
    ModelDevMergeBuildOverdueEvent,
)
from omnibase_infra.models.model_dev_merge_build_publish_request import (
    ModelDevMergeBuildPublishRequest,
)
from omnibase_infra.models.model_overdue_dev_merge_build_decision import (
    ModelOverdueDevMergeBuildDecision,
)
from omnibase_infra.protocols.protocol_dev_merge_build_sender import (
    ProtocolDevMergeBuildSender,
)

pytestmark = pytest.mark.unit

_MERGE = "a27c0c218555c93421803e9f6dc4dd95c28b3da5"
_HEAD = "b" * 40
_DUE = datetime(2026, 11, 23, 18, 46, 52, tzinfo=UTC)


class _Sender:
    def __init__(self, status: str = "published") -> None:
        self.status = status
        self.events: list[ModelDevMergeBuildOverdueEvent] = []
        self.commands: list[ModelDevMergeBuildCommand] = []

    def publish_event(self, event: ModelDevMergeBuildOverdueEvent) -> None:
        self.events.append(event)

    def send_command(self, command: ModelDevMergeBuildCommand) -> str:
        self.commands.append(command)
        return self.status


def _handler(sender: _Sender) -> HandlerDevMergeBuildPublish:
    container = ModelONEXContainer()
    sender_key: type[object] = ProtocolDevMergeBuildSender
    run_coro_sync(container.service_registry.register_instance(sender_key, sender))
    return HandlerDevMergeBuildPublish(container)


def _decision(
    outcome: str = "REQUEST", reason: str = "overdue_no_build"
) -> ModelOverdueDevMergeBuildDecision:
    return ModelOverdueDevMergeBuildDecision.model_validate(
        {
            "outcome": outcome,
            "reason": reason,
            "merge_sha": _MERGE,
            "due_at": _DUE,
            "request_id": request_id_for("omnibase_infra", _MERGE)
            if outcome == "REQUEST"
            else None,
        }
    )


def _request(**over: object) -> ModelDevMergeBuildPublishRequest:
    base: dict[str, object] = {
        "decision": _decision(),
        "repository": "omnibase_infra",
        "pr_number": 6191,
        "build_ref": _HEAD,
        "runtime_lane": "dev",
        "already_requested": False,
    }
    return ModelDevMergeBuildPublishRequest.model_validate({**base, **over})


def test_an_overdue_merge_publishes_the_event_and_requests_one_dev_build() -> None:
    sender = _Sender()
    result = _handler(sender).handle(_request())
    assert (result.status, result.reason) == ("requested", "published")
    (event,) = sender.events
    assert event.merge_sha == _MERGE
    assert event.due_at == _DUE
    assert event.request_id == request_id_for("omnibase_infra", _MERGE)
    (command,) = sender.commands
    assert command.correlation_id == request_id_for("omnibase_infra", _MERGE)
    assert command.runtime_lane == "dev"
    assert command.git_ref == _HEAD
    assert (command.repo, command.pr_number, command.mode) == (
        "omnibase_infra",
        6191,
        "deploy",
    )


def test_a_prod_lane_is_refused_with_a_typed_error_and_nothing_is_sent() -> None:
    sender = _Sender()
    for lane in ("prod", "other", ""):
        with pytest.raises(ProtocolConfigurationError):
            _handler(sender).handle(_request(runtime_lane=lane))
    assert sender.events == [] and sender.commands == []


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        ("WAIT", "job_in_flight"),
        ("WAIT", "failed_head"),
        ("WAIT", "within_window"),
        ("SATISFIED", "built_by_job"),
    ],
)
def test_no_build_is_requested_unless_the_decision_is_a_request(
    outcome: str, reason: str
) -> None:
    sender = _Sender()
    result = _handler(sender).handle(_request(decision=_decision(outcome, reason)))
    assert (result.status, result.reason) == ("skipped", "not_overdue")
    assert sender.events == [] and sender.commands == []


def test_a_sha_already_requested_is_never_requested_twice() -> None:
    sender = _Sender()
    result = _handler(sender).handle(_request(already_requested=True))
    assert (result.status, result.reason) == ("skipped", "already_requested")
    assert sender.events == [] and sender.commands == []


def test_a_merge_with_no_pull_request_number_is_not_requested() -> None:
    sender = _Sender()
    result = _handler(sender).handle(_request(pr_number=None))
    assert (result.status, result.reason) == ("skipped", "no_pull_request_number")
    assert sender.commands == []


@pytest.mark.parametrize("status", ["refused", "failed"])
def test_a_refused_or_failed_send_is_a_failed_result_so_the_next_tick_tries_again(
    status: str,
) -> None:
    sender = _Sender(status)
    result = _handler(sender).handle(_request())
    assert (result.status, result.reason) == ("failed", f"send_{status}")
    assert result.request_id == UUID(str(request_id_for("omnibase_infra", _MERGE)))


def test_the_tick_requests_each_overdue_sha_once_and_never_on_a_prod_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The caller reads the lane and topic from the node's settings and records a sha only once sent."""
    from types import SimpleNamespace

    from omnibase_infra.nodes.node_deploy_job_reading_effect import (
        runtime_deploy_job_reading as caller,
    )

    for relative in caller._OVERDUE_MODULES:
        caller._load(relative)
    repo = tmp_path / "clone"
    state = tmp_path / "state"
    repo.mkdir()
    state.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            env=scrub_git_location_env(os.environ),
        )

    git("init", "-q")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    git("commit", "--allow-empty", "-q", "-m", "merge (#6191)")
    sender = _Sender()
    monkeypatch.setattr(caller, "AgentDevMergeSender", lambda *_a: sender)
    monkeypatch.setattr(caller, "_TRACKED_HEAD", "HEAD")
    execution = SimpleNamespace(STATE_DIR=str(state), REPO_DIR=str(repo))
    settings = {
        "dev_merge_build_runtime_lane": "dev",
        "dev_merge_build_overdue_topic": "topic-from-settings",
        "dev_merge_build_command_topic": "command-topic-from-settings",
        "build_decision_git_diff_timeout_seconds": 10,
    }
    overdue = [_decision()]
    subjects = {_MERGE: "merge (#6191)"}
    for _ in range(2):
        caller._request_overdue_builds(execution, settings, overdue, subjects)
    assert len(sender.commands) == 1 and len(sender.events) == 1
    assert (state / caller._REQUESTED_KEYS).read_text().split() == [
        str(request_id_for("omnibase_infra", _MERGE))
    ]
    with pytest.raises(ProtocolConfigurationError):
        caller._request_overdue_builds(
            execution,
            {**settings, "dev_merge_build_runtime_lane": "prod"},
            [_decision()],
            subjects,
        )
    assert len(sender.commands) == 1
