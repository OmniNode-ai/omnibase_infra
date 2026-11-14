# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev roll events the deploy agent publishes, and their seams (OMN-24013).

Five whys plan roots R1 and R2. The start event's shape is pinned by omnimarket node
node_dev_roll_verify_compute (its test_seam_start_event_contract.py spells the same five
fields and three causes); the incident and manifest shapes are pinned by omnimarket node
node_dev_roll_failure_capture_effect (its test_seam_agent_events.py). The other repository
cannot be imported here and a skipped cross-repository check is refused, so each side
spells the wire shape; when one side changes, the spelled copy on the other side changes
in the change that follows it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_dev_roll_events_effect import runtime_dev_roll_events
from omnibase_infra.nodes.node_dev_roll_events_effect.dev_roll_events import (
    DevRollEventsPublisher,
    EnumDevStartCause,
    ModelDevRollIncident,
    ModelDevRollManifest,
    ModelDevRuntimeStarted,
    docker_started_at,
)

pytestmark = pytest.mark.unit

# What node_dev_roll_verify_compute reads (its seam test spells the same set).
START_EVENT_FIELDS = {"event_id", "job_id", "runtime_lane", "started_at", "cause"}
START_CAUSES = {"roll_recreate", "branch_install_reapply", "other"}
# What node_dev_roll_failure_capture_effect reads (its seam test spells the same sets).
INCIDENT_FIELDS = {
    "event_id",
    "job_id",
    "issuer",
    "commit",
    "error",
    "project",
    "at",
    "restored_revision",
    "recovery_verified",
    "recovery_skipped_healthy",
    "recovery_error",
    "failed_phase",
    "runtime_kept_revision",
    "health_probes",
    "log_captures",
}
LOG_CAPTURE_FIELDS = {
    "service",
    "container",
    "bytes_total",
    "bytes_kept",
    "complete",
    "redacted",
    "sha256",
    "artifact_path",
    "error",
    "log",
}
MANIFEST_FIELDS = {
    "event_id",
    "job_id",
    "issuer",
    "git_ref",
    "scope",
    "requested_at",
    "accepted_at",
    "open_pr",
    "rollback",
}
# The topics by role, spelled as the node's contract declares them; the consumers in
# omnimarket (node_dev_roll_verify_compute, node_dev_roll_failure_capture_effect)
# subscribe to these names.
TOPICS = {
    "dev_runtime_started": "onex.evt.omnimarket.dev-runtime-started.v1",
    "dev_roll_manifest": "onex.evt.omnimarket.dev-roll-manifest.v1",
    "dev_roll_incident": "onex.evt.omnimarket.dev-roll-incident.v1",
}
TOPIC_DEV_RUNTIME_STARTED = TOPICS["dev_runtime_started"]
TOPIC_DEV_ROLL_MANIFEST = TOPICS["dev_roll_manifest"]
TOPIC_DEV_ROLL_INCIDENT = TOPICS["dev_roll_incident"]
COMMAND: dict[str, Any] = {
    "correlation_id": "6f1c0e0a-0000-4000-8000-000000000135",
    "requested_by": "gha/omnimarket/pr-5942",
    "git_ref": "a" * 40,
    "scope": "full",
    "requested_at": "2026-11-10T04:06:00Z",
    "open_pr": "",
    "rollback": None,
}


def _guard_incident(**extra: Any) -> dict[str, Any]:
    return {
        "job": "job-1",
        "commit": "c" * 40,
        "error": "unhealthy",
        "status": "failed",
        "project": "omnibase-infra",
        "at": "2026-11-10T04:15:42Z",
        "restored_revision": "a" * 40,
        "recovery_verified": True,
        "recovery_skipped_healthy": False,
        "ledger_pending": True,
        "exception_pending": True,
        "health_probes": [{"service": "omninode-runtime"}],
        "log_captures": [
            {
                "service": "omninode-runtime",
                "container": "abc123",
                "bytes_total": 10,
                "bytes_kept": 10,
                "complete": True,
                "redacted": False,
                "sha256": "a" * 64,
                "artifact_path": "/var/lib/deploy/incident-job-1-omninode-runtime.log",
                "error": None,
                "log": "boot failed",
            }
        ],
        **extra,
    }


class _Bus:
    def __init__(self, *, accepts: bool = True, raises: bool = False) -> None:
        self.sent: list[tuple[str, str, dict[str, Any]]] = []
        self._accepts = accepts
        self._raises = raises

    def __call__(self, topic: str, key: str, payload: dict[str, Any]) -> bool:
        if self._raises:
            raise OSError("broker down")
        self.sent.append((topic, key, payload))
        return self._accepts


def test_the_start_event_has_exactly_the_five_fields_the_start_count_node_reads() -> (
    None
):
    assert set(ModelDevRuntimeStarted.model_fields) == START_EVENT_FIELDS
    assert {cause.value for cause in EnumDevStartCause} == START_CAUSES


def test_incident_and_manifest_shapes_are_the_ones_the_capture_node_spells() -> None:
    assert set(ModelDevRollIncident.model_fields) == INCIDENT_FIELDS
    assert set(ModelDevRollManifest.model_fields) == MANIFEST_FIELDS
    log_capture = ModelDevRollIncident.model_fields["log_captures"].annotation
    assert log_capture is not None


def test_a_log_capture_carries_exactly_the_spelled_fields() -> None:
    incident = ModelDevRollIncident.from_guard_incident(
        _guard_incident(), event_id=uuid4(), issuer="lane-x"
    )
    assert set(type(incident.log_captures[0]).model_fields) == LOG_CAPTURE_FIELDS


def test_a_naive_start_time_a_foreign_lane_and_an_extra_field_are_refused() -> None:
    good: dict[str, Any] = {
        "event_id": uuid4(),
        "job_id": "j1",
        "runtime_lane": "dev",
        "started_at": datetime(2026, 11, 10, 4, 15, tzinfo=UTC),
        "cause": EnumDevStartCause.ROLL_RECREATE,
    }
    ModelDevRuntimeStarted(**good)
    for bad in (
        {**good, "started_at": datetime(2026, 11, 10, 4, 15)},
        {**good, "runtime_lane": "elsewhere"},
        {**good, "extra": 1},
    ):
        with pytest.raises(ValidationError):
            ModelDevRuntimeStarted(**bad)


def test_the_manifest_names_the_command_and_its_issuer() -> None:
    bus = _Bus()
    DevRollEventsPublisher(bus, TOPICS).manifest(COMMAND)

    ((topic, key, payload),) = bus.sent
    assert topic == TOPIC_DEV_ROLL_MANIFEST
    assert key == f"dev-roll-manifest/{COMMAND['correlation_id']}"
    assert payload["job_id"] == COMMAND["correlation_id"]
    assert payload["issuer"] == "gha/omnimarket/pr-5942"
    assert payload["rollback"] is False
    ModelDevRollManifest.model_validate(payload)


def test_an_incident_carries_the_issuer_and_drops_the_guards_pending_keys() -> None:
    bus = _Bus()
    DevRollEventsPublisher(bus, TOPICS).incident(_guard_incident(), issuer="lane-x")

    topic, _key, payload = bus.sent[0]
    assert topic == TOPIC_DEV_ROLL_INCIDENT
    assert payload["issuer"] == "lane-x"
    assert "ledger_pending" not in payload
    assert payload["log_captures"][0]["log"] == "boot failed"
    ModelDevRollIncident.model_validate(payload)


def test_a_verified_recovery_recreate_is_one_more_start_of_cause_other() -> None:
    bus = _Bus()
    DevRollEventsPublisher(bus, TOPICS).incident(_guard_incident(), issuer="lane-x")

    assert [topic for topic, _k, _p in bus.sent] == [
        TOPIC_DEV_ROLL_INCIDENT,
        TOPIC_DEV_RUNTIME_STARTED,
    ]
    assert bus.sent[1][2]["cause"] == "other"
    assert bus.sent[1][2]["job_id"] == "job-1"


@pytest.mark.parametrize(
    "extra",
    [
        {"recovery_skipped_healthy": True},
        {"runtime_kept_revision": "b" * 40},
        {"recovery_verified": False},
    ],
)
def test_a_recovery_that_recreated_nothing_is_not_a_second_start(
    extra: dict[str, Any],
) -> None:
    bus = _Bus()
    DevRollEventsPublisher(bus, TOPICS).incident(
        _guard_incident(**extra), issuer="lane-x"
    )

    assert [topic for topic, _k, _p in bus.sent] == [TOPIC_DEV_ROLL_INCIDENT]


@pytest.mark.parametrize("bus", [_Bus(accepts=False), _Bus(raises=True)])
def test_a_bus_that_refuses_or_raises_never_propagates(bus: _Bus) -> None:
    events = DevRollEventsPublisher(bus, TOPICS)
    events.manifest(COMMAND)
    events.started("j1", datetime(2026, 11, 10, tzinfo=UTC), EnumDevStartCause.OTHER)
    events.incident(_guard_incident(), issuer="lane-x")


def test_an_incident_with_no_job_sends_nothing_and_does_not_raise() -> None:
    bus = _Bus()
    DevRollEventsPublisher(bus, TOPICS).incident({"container": "c1"}, issuer="lane-x")

    assert bus.sent == []


def test_a_manifest_with_no_issuer_sends_nothing() -> None:
    bus = _Bus()
    DevRollEventsPublisher(bus, TOPICS).manifest({**COMMAND, "requested_by": ""})

    assert bus.sent == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "2026-11-10T04:07:46.123456789Z",
            datetime(2026, 11, 10, 4, 7, 46, 123456, tzinfo=UTC),
        ),
        ("2026-11-10T04:07:46Z", datetime(2026, 11, 10, 4, 7, 46, tzinfo=UTC)),
        (
            "2026-11-10T04:07:46.5+00:00",
            datetime(2026, 11, 10, 4, 7, 46, 500000, tzinfo=UTC),
        ),
    ],
)
def test_docker_start_times_are_cut_to_six_digits_not_rounded(
    raw: str, expected: datetime
) -> None:
    fallback = datetime(2000, 1, 1, tzinfo=UTC)

    assert docker_started_at(raw, fallback=fallback) == expected


@pytest.mark.parametrize("raw", [None, "", "not a time"])
def test_an_unreadable_docker_start_time_is_the_fallback(raw: str | None) -> None:
    fallback = datetime(2000, 1, 1, tzinfo=UTC)

    assert docker_started_at(raw, fallback=fallback) == fallback


def test_preflight_failure_sends_manifest_then_incident_with_failed_phase() -> None:
    bus = _Bus()
    DevRollEventsPublisher(bus, TOPICS).preflight_failure(
        COMMAND,
        error="Git remote unreachable",
        phase="preflight",
        project="omnibase-infra-dev",
    )

    assert [topic for topic, _k, _p in bus.sent] == [
        TOPIC_DEV_ROLL_MANIFEST,
        TOPIC_DEV_ROLL_INCIDENT,
    ]
    incident_payload = bus.sent[1][2]
    assert incident_payload["issuer"] == "gha/omnimarket/pr-5942"
    assert incident_payload["failed_phase"] == "preflight"
    assert incident_payload["error"] == "Git remote unreachable"
    assert incident_payload["recovery_verified"] is False
    assert set(incident_payload) == INCIDENT_FIELDS
    ModelDevRollIncident.model_validate(incident_payload)


def test_the_agents_sender_creates_the_topic_then_produces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deploy_agent import kafka_config, publisher

    from omnibase_infra.topics.topic_namespace import apply_topic_namespace

    calls: list[tuple[str, str]] = []
    config = object()
    monkeypatch.setattr(
        kafka_config, "load_deploy_agent_kafka_config_from_env", lambda: config
    )
    monkeypatch.setattr(
        publisher, "_ensure_topic", lambda topic, cfg: calls.append(("ensure", topic))
    )
    monkeypatch.setattr(
        publisher,
        "_produce",
        lambda topic, key, payload, cfg: calls.append(("produce", topic)) or True,
    )

    assert (
        runtime_dev_roll_events.agent_send()(TOPIC_DEV_RUNTIME_STARTED, "k", {"a": 1})
        is True
    )
    assert calls == [
        ("ensure", apply_topic_namespace(TOPIC_DEV_RUNTIME_STARTED)),
        ("produce", TOPIC_DEV_RUNTIME_STARTED),
    ]


def test_an_event_id_is_a_uuid() -> None:
    bus = _Bus()
    DevRollEventsPublisher(bus, TOPICS).manifest(COMMAND)

    assert UUID(bus.sent[0][2]["event_id"])


def test_the_contract_declares_the_topics_and_cap_the_composition_root_reads() -> None:
    from deploy_agent import executor

    declared = runtime_dev_roll_events.declaration(executor)

    assert declared["topics"] == TOPICS
    assert runtime_dev_roll_events.log_byte_cap(executor) == 262_144


def test_the_composition_root_builds_a_publisher_on_the_declared_topics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deploy_agent import executor

    bus = _Bus()
    monkeypatch.setattr(runtime_dev_roll_events, "agent_send", lambda: bus)
    runtime_dev_roll_events.build_events(executor).manifest(COMMAND)

    assert bus.sent[0][0] == TOPIC_DEV_ROLL_MANIFEST


def test_a_contract_with_no_log_cap_refuses(tmp_path: Any) -> None:
    import types

    root = tmp_path / "scripts" / "deploy-agent" / "deploy_agent"
    node = tmp_path / "src/omnibase_infra/nodes/node_dev_roll_events_effect"
    root.mkdir(parents=True)
    node.mkdir(parents=True)
    (node / "contract.yaml").write_text("topics: {}\nsettings: {}\n", encoding="utf-8")
    execution = types.ModuleType("fake_execution")
    execution.__file__ = str(root / "executor.py")

    with pytest.raises(RuntimeError, match="log_byte_cap"):
        runtime_dev_roll_events.log_byte_cap(execution)
