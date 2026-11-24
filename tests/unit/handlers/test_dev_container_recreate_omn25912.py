# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Each start of the dev runtime container is published with its cause, and the reader reads it (OMN-25912).

Run with ``-k container_recreate``. The class: a dev container recreate that leaves no event
(the 2026-11-23T22:53:58Z recreate on the build of 21:14:35Z). On origin/dev nothing publishes
a container's StartedAt or RestartCount and nothing calls ``audit_and_publish`` outside tests,
so a recreate on the same build raises no alarm.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from omnibase_infra.enums.enum_dev_recreate_cause import EnumDevRecreateCause as Cause
from omnibase_infra.enums.enum_dev_recreate_violation_kind import (
    EnumDevRecreateViolationKind as Kind,
)
from omnibase_infra.handlers.handler_dev_container_recreate import (
    HandlerDevContainerRecreate,
)
from omnibase_infra.models.model_dev_container_observation import (
    ModelDevContainerObservation,
)
from omnibase_infra.models.model_dev_container_reading import ModelDevContainerReading
from omnibase_infra.models.model_dev_container_recreate_request import (
    ModelDevContainerRecreateRequest,
)
from omnibase_infra.models.model_dev_recreate_audit_request import (
    ModelDevRecreateAuditRequest,
)
from omnibase_infra.models.model_dev_recreate_policy import ModelDevRecreatePolicy
from omnibase_infra.models.model_dev_runtime_started import ModelDevRuntimeStarted
from omnibase_infra.nodes.node_dev_roll_events_effect import runtime_dev_roll_events
from omnibase_infra.nodes.node_dev_roll_events_effect.dev_recreate_audit import (
    DevRecreateAuditReader,
)
from omnibase_infra.nodes.node_dev_roll_events_effect.dev_roll_events import (
    DevRollEventsPublisher,
    audit_settings,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
NODE = REPO_ROOT / "src" / "omnibase_infra" / "nodes" / "node_dev_roll_events_effect"
CONTRACT = NODE / "contract.yaml"
PROJECT = "omnibase-infra-dev"
SAME_BUILD = "sha256:" + "a" * 64
NEW_BUILD = "sha256:" + "b" * 64


def _contract() -> dict[str, Any]:
    declared: dict[str, Any] = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    return declared


def _reading(
    *,
    container_digest: str,
    started_at: datetime,
    restart_count: int = 0,
    image_digest: str = SAME_BUILD,
) -> ModelDevContainerReading:
    return ModelDevContainerReading(
        container="omninode-runtime",
        container_digest=container_digest,
        started_at=started_at,
        restart_count=restart_count,
        exit_code=0,
        image_digest=image_digest,
        image_ref="omninode-runtime:dev",
    )


def _tolerance() -> int:
    return audit_settings(CONTRACT)[1]


def _handle(
    reading: ModelDevContainerReading,
    previous: ModelDevContainerReading | None,
    **kwargs: Any,
) -> Any:
    return (
        HandlerDevContainerRecreate()
        .handle(
            ModelDevContainerRecreateRequest(
                reading=reading,
                previous=previous,
                start_match_tolerance_seconds=_tolerance(),
                **kwargs,
            )
        )
        .event
    )


def _policy() -> ModelDevRecreatePolicy:
    env: dict[str, str] = {}
    for line in (REPO_ROOT / "docker" / "runtime-policy.env").read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.startswith("#"):
            env[key] = value
    return ModelDevRecreatePolicy.from_env(env)


def test_container_recreate_on_the_same_build_is_published_with_its_cause_and_flagged() -> (
    None
):
    """The proof case: a recreate on the previous build, no job event, no restart policy."""
    previous = _reading(
        container_digest="c1", started_at=datetime(2026, 11, 23, 21, 14, 35, tzinfo=UTC)
    )
    now = _reading(
        container_digest="c2",
        started_at=datetime(2026, 11, 23, 22, 53, 58, tzinfo=UTC),
    )
    event = _handle(now, previous)

    assert event is not None
    assert event.cause is Cause.UNKNOWN
    assert event.same_build is True
    assert (event.image_digest, event.image_ref) == (SAME_BUILD, "omninode-runtime:dev")
    assert event.started_at == now.started_at

    sent: list[tuple[str, str, dict[str, Any]]] = []
    topics = _contract()["topics"]
    publisher = DevRollEventsPublisher(
        lambda topic, key, payload: sent.append((topic, key, payload)) or True, topics
    )
    assert publisher.container_recreated(event) is True
    (topic, _, payload) = sent[0]
    assert topic == topics["dev_container_recreated"]
    assert payload["cause"] == "unknown"
    assert payload["restart_count"] == 0
    assert payload["same_build"] is True

    lookback, tolerance = audit_settings(CONTRACT)
    observation = ModelDevContainerObservation.model_validate(
        {
            "container": payload["container"],
            "started_at": payload["started_at"],
            "restart_count": payload["restart_count"],
            "exit_code": payload["exit_code"],
            "cause": payload["cause"],
            "same_build": payload["same_build"],
        }
    )
    verdict = DevRecreateAuditReader().handle(
        ModelDevRecreateAuditRequest(
            policy=_policy(),
            announcement_lookback_seconds=lookback,
            start_match_tolerance_seconds=tolerance,
            container_observations=(observation,),
        )
    )
    flagged = [v for v in verdict.violations if v.kind is Kind.START_WITHOUT_JOB_RECORD]
    assert len(flagged) == 1
    assert "cause unknown, on the same build as the previous start" in flagged[0].detail


def test_container_recreate_first_reading_and_an_unchanged_container_name_no_start() -> (
    None
):
    started = datetime(2026, 11, 23, 21, 14, 35, tzinfo=UTC)
    reading = _reading(container_digest="c1", started_at=started)
    assert _handle(reading, None) is None
    assert _handle(reading, reading) is None


def test_container_recreate_cause_is_the_job_the_restart_policy_or_unknown() -> None:
    started = datetime(2026, 11, 23, 22, 53, 58, tzinfo=UTC)
    previous = _reading(
        container_digest="c1", started_at=started - timedelta(hours=1), restart_count=2
    )

    # The same container started again with a higher RestartCount and no job event.
    policy = _handle(
        _reading(container_digest="c1", started_at=started, restart_count=3), previous
    )
    assert policy.cause is Cause.RESTART_POLICY
    assert (policy.job_id, policy.requested_by) == ("", "")

    # The same container started again and RestartCount did not move: no proof, no guess.
    hand = _handle(
        _reading(container_digest="c1", started_at=started, restart_count=2), previous
    )
    assert hand.cause is Cause.UNKNOWN

    # A new image: not the same build.
    changed = _handle(
        _reading(container_digest="c2", started_at=started, image_digest=NEW_BUILD),
        previous,
    )
    assert changed.same_build is False
    assert changed.previous_image_digest == SAME_BUILD

    # A start event of a deploy job within the contract tolerance names the job.
    job = _handle(
        _reading(container_digest="c2", started_at=started),
        previous,
        starts=(
            ModelDevRuntimeStarted(
                event_id=uuid4(),
                job_id="job-7",
                runtime_lane="dev",
                started_at=started + timedelta(seconds=_tolerance() - 1),
                cause="roll_recreate",
            ),
        ),
    )
    assert job.cause is Cause.DEPLOY_AGENT_JOB
    assert job.job_id == "job-7"
    # No manifest names the issuer, so the requester stays empty rather than guessed.
    assert job.requested_by == ""


def _docker(
    container_digest: str, started_at: datetime, *, image: str, restarts: int
) -> Any:
    inspected = [
        {
            "Id": container_digest,
            "Name": "/omninode-runtime",
            "Image": image,
            "RestartCount": restarts,
            "Config": {"Image": "omninode-runtime:dev"},
            "State": {
                "StartedAt": started_at.strftime("%Y-%m-%dT%H:%M:%S.123456789Z"),
                "ExitCode": 0,
            },
        }
    ]

    def run(argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        stdout = container_digest if argv[1] == "ps" else json.dumps(inspected)
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    return run


def _tick(
    tmp_path: Path,
    sent: list[tuple[str, str, dict[str, Any]]],
    run: Any,
) -> Any:
    publisher = runtime_dev_roll_events.build_events_at(
        REPO_ROOT,
        tmp_path,
        send=lambda topic, key, payload: sent.append((topic, key, payload)) or True,
    )
    return (
        runtime_dev_roll_events.observe_dev_container(
            tmp_path, PROJECT, REPO_ROOT, publisher=publisher, run=run
        ),
        publisher,
    )


def test_container_recreate_tick_publishes_the_event_and_calls_the_reader(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    topics = _contract()["topics"]
    sent: list[tuple[str, str, dict[str, Any]]] = []

    # Tick 1 reads the container once: the baseline, no start named.
    audited, _ = _tick(
        tmp_path,
        sent,
        _docker("c1", now - timedelta(hours=2), image=SAME_BUILD, restarts=0),
    )
    assert [t for t, _, _ in sent if t == topics["dev_container_recreated"]] == []

    # Tick 2: a new container on the same image started 5 minutes ago, no deploy job.
    audited, _ = _tick(
        tmp_path,
        sent,
        _docker("c2", now - timedelta(minutes=5), image=SAME_BUILD, restarts=0),
    )
    events = [p for t, _, p in sent if t == topics["dev_container_recreated"]]
    assert len(events) == 1
    assert events[0]["cause"] == "unknown"
    assert events[0]["same_build"] is True
    assert events[0]["container_digest"] == "c2"
    violations = [p for t, _, p in sent if t == topics["dev_recreate_violation"]]
    assert [v["kind"] for v in violations] == ["start_without_job_record"]
    assert "cause unknown, on the same build" in violations[0]["detail"]
    assert audited is not None

    # Tick 3 on the same container: nothing new is published.
    count = len(sent)
    _tick(
        tmp_path,
        sent,
        _docker("c2", now - timedelta(minutes=5), image=SAME_BUILD, restarts=0),
    )
    assert len(sent) == count


def test_container_recreate_tick_names_the_deploy_agent_job_that_made_the_start(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    topics = _contract()["topics"]
    sent: list[tuple[str, str, dict[str, Any]]] = []
    _tick(
        tmp_path,
        sent,
        _docker("c1", now - timedelta(hours=2), image=SAME_BUILD, restarts=0),
    )
    started = now - timedelta(minutes=3)
    _, publisher = _tick(
        tmp_path,
        sent,
        _docker("c1", now - timedelta(hours=2), image=SAME_BUILD, restarts=0),
    )
    # The deploy agent's roll: its manifest and start event, journaled as published.
    publisher.manifest(
        {
            "correlation_id": "job-9",
            "requested_by": "agent/idle-converge",
            "git_ref": "abc123",
            "scope": "runtime",
        }
    )
    publisher.started("job-9", started, "roll_recreate")
    _tick(
        tmp_path,
        sent,
        _docker("c3", started, image=NEW_BUILD, restarts=0),
    )
    events = [p for t, _, p in sent if t == topics["dev_container_recreated"]]
    assert [(e["cause"], e["job_id"], e["requested_by"]) for e in events] == [
        ("deploy_agent_job", "job-9", "agent/idle-converge")
    ]
    assert events[0]["same_build"] is False
    assert not [
        p
        for t, _, p in sent
        if t == topics["dev_recreate_violation"]
        and p["kind"] == "start_without_job_record"
    ]


def test_container_recreate_reader_is_called_on_a_declared_trigger() -> None:
    """The caller of ``audit_and_publish`` outside tests is the agent tick the contract declares."""
    settings = _contract()["settings"]
    assert settings["audit_trigger"] == "agent_tick"
    glue = (NODE / "runtime_dev_roll_events.py").read_text(encoding="utf-8")
    assert "audit_and_publish(" in glue
    reading = (
        REPO_ROOT
        / "src/omnibase_infra/nodes/node_deploy_job_reading_effect"
        / "runtime_deploy_job_reading.py"
    ).read_text(encoding="utf-8")
    # The agent's idle tick calls check_pin for the dev lane; check_pin fans out to the observer.
    assert "check_dev_container(force=force)" in reading
    assert "roll.observe_if_due(" in reading
    agent = (
        REPO_ROOT / "scripts" / "deploy-agent" / "deploy_agent" / "agent.py"
    ).read_text(encoding="utf-8")
    assert 'await self._offload(self._job_reading, "check_pin")' in agent
    # No topic or interval is spelled in the glue: the contract declares them.
    assert "onex.evt." not in glue.split("# --- OMN-25912: the container observer")[1]


def test_container_recreate_observer_runs_once_per_declared_interval(
    monkeypatch: Any, tmp_path: Path
) -> None:
    interval = _contract()["settings"]["container_observation_interval_seconds"]
    ran: list[Path] = []
    monkeypatch.setattr(
        runtime_dev_roll_events,
        "observe_dev_container",
        lambda state_dir, project, repo_root: ran.append(state_dir),
    )
    monkeypatch.setattr(runtime_dev_roll_events, "_observed_last", [])
    clock = [1000.0]

    def tick() -> bool:
        return runtime_dev_roll_events.observe_if_due(
            tmp_path, PROJECT, monotonic=lambda: clock[0]
        )

    assert tick() is True
    clock[0] += interval - 1
    assert tick() is False
    clock[0] += 1
    assert tick() is True
    assert len(ran) == 2


def test_container_recreate_a_failing_observation_never_raises_into_the_tick(
    monkeypatch: Any, tmp_path: Path
) -> None:
    def boom(*_: Any) -> None:
        raise RuntimeError("docker unreadable")

    monkeypatch.setattr(runtime_dev_roll_events, "observe_dev_container", boom)
    monkeypatch.setattr(runtime_dev_roll_events, "_observed_last", [])
    assert runtime_dev_roll_events.observe_if_due(tmp_path, PROJECT) is False
