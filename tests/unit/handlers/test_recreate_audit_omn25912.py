# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The reader of the one-recreate-an-hour rule names every start that broke it (OMN-25912).

Run with ``-k recreate_audit``. The fixture is the 24 hours from 2026-11-22T20:04Z of the
dev lane on h201: the deploy agent's job records (16 recreates) and the start, manifest and
announcement events of the same window. The window gate covers the idle converge and the
idle restore; a lane's rollback command and the container's own restart policy bypass it,
and nothing read the events they left. On origin/dev this module's handler does not exist,
so no violation event is emitted for any of the 13 cases below.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import yaml

from omnibase_infra.enums.enum_dev_recreate_violation_kind import (
    EnumDevRecreateViolationKind as Kind,
)
from omnibase_infra.models.model_dev_container_observation import (
    ModelDevContainerObservation,
)
from omnibase_infra.models.model_dev_recreate_announcement import (
    ModelDevRecreateAnnouncement,
)
from omnibase_infra.models.model_dev_recreate_audit_request import (
    ModelDevRecreateAuditRequest,
)
from omnibase_infra.models.model_dev_recreate_policy import ModelDevRecreatePolicy
from omnibase_infra.models.model_dev_recreate_violation import (
    ModelDevRecreateViolation,
)
from omnibase_infra.models.model_dev_roll_manifest import ModelDevRollManifest
from omnibase_infra.models.model_dev_runtime_started import ModelDevRuntimeStarted
from omnibase_infra.nodes.node_dev_roll_events_effect.dev_recreate_audit import (
    DevRecreateAuditReader,
    audit_and_publish,
)
from omnibase_infra.nodes.node_dev_roll_events_effect.dev_roll_events import (
    DevRollEventsPublisher,
    audit_settings,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = Path(__file__).parent / "fixtures" / "recreate_audit_window_24h_omn25912.json"
POLICY_ENV = REPO_ROOT / "docker" / "runtime-policy.env"
NODE = REPO_ROOT / "src" / "omnibase_infra" / "nodes" / "node_dev_roll_events_effect"
AUDIT_SOURCE = NODE / "dev_recreate_audit.py"
PINNED = ("operator/pinned-fix/", "agent/merge-restore")


def _declared_policy() -> ModelDevRecreatePolicy:
    env: dict[str, str] = {}
    for line in POLICY_ENV.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.startswith("#"):
            env[key] = value
    return ModelDevRecreatePolicy.from_env(env)


def _at(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)


def _fixture() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return loaded


def _request(
    *,
    starts: Any = (),
    manifests: Any = (),
    announcements: Any = (),
    observations: Any = (),
) -> ModelDevRecreateAuditRequest:
    lookback, tolerance = audit_settings(NODE / "contract.yaml")
    return ModelDevRecreateAuditRequest(
        policy=_declared_policy(),
        announcement_lookback_seconds=lookback,
        start_match_tolerance_seconds=tolerance,
        starts=tuple(ModelDevRuntimeStarted.model_validate(s) for s in starts),
        manifests=tuple(ModelDevRollManifest.model_validate(m) for m in manifests),
        announcements=tuple(
            ModelDevRecreateAnnouncement.model_validate(a) for a in announcements
        ),
        container_observations=tuple(
            ModelDevContainerObservation.model_validate(o) for o in observations
        ),
    )


def _window_request(*, with_observations: bool = True) -> ModelDevRecreateAuditRequest:
    fixture = _fixture()
    return _request(
        starts=fixture["starts"],
        manifests=fixture["manifests"],
        announcements=fixture["announcements"],
        observations=fixture["container_observations"]["rows"]
        if with_observations
        else (),
    )


def _by_kind(violations: tuple[ModelDevRecreateViolation, ...]) -> Counter[Kind]:
    return Counter(v.kind for v in violations)


def _hand_count(fixture: dict[str, Any]) -> dict[str, Any]:
    """The counts read straight off the job records and announcements, no handler."""
    jobs = fixture["job_records"]
    per_hour = Counter(_at(j["accepted_at"]).strftime("%Y-%m-%dT%H") for j in jobs)
    immediate = [a for a in fixture["announcements"] if a["immediate"]]
    off_the_hour = [
        j["job_id"][:8]
        for j in jobs
        if _at(j["accepted_at"]).minute * 60 + _at(j["accepted_at"]).second > 300
        and not j["requested_by"].startswith("agent/idle-restore")
    ]
    unannounced = [
        j["job_id"][:8]
        for j in jobs
        if not any(
            _at(j["accepted_at"]) - timedelta(seconds=1800)
            <= _at(a["announced_at"])
            <= _at(j["accepted_at"])
            for a in immediate
        )
    ]
    return {
        "recreates": len(jobs),
        "two_hours": sorted(h for h, n in per_hour.items() if n > 1),
        "off_the_hour_not_agent": off_the_hour,
        "unannounced": unannounced,
    }


def test_recreate_audit_24h_window_emits_one_violation_per_case() -> None:
    fixture = _fixture()
    expected = fixture["expected"]
    result = DevRecreateAuditReader().handle(_window_request())
    counts = _by_kind(result.violations)

    assert result.recreates_read == expected["recreates"] == 16
    assert counts[Kind.TWO_STARTS_IN_WINDOW] == expected["two_starts_in_window"] == 3
    assert counts[Kind.OFF_WINDOW_START] == expected["off_window_start"] == 4
    assert counts[Kind.UNANNOUNCED_START] == expected["unannounced_start"] == 4
    assert counts[Kind.START_WITHOUT_JOB_RECORD] == 2
    assert len(result.violations) == expected["total"] == 13
    assert len({v.event_id for v in result.violations}) == 13


def test_recreate_audit_count_equals_the_hand_count_from_the_job_records() -> None:
    fixture = _fixture()
    hand = _hand_count(fixture)
    result = DevRecreateAuditReader().handle(_window_request())

    two = sorted(
        v.window_start.strftime("%Y-%m-%dT%H")
        for v in result.violations
        if v.kind is Kind.TWO_STARTS_IN_WINDOW
    )
    assert (
        two == hand["two_hours"] == ["2026-11-23T12", "2026-11-23T15", "2026-11-23T17"]
    )
    off = sorted(
        v.rows[0].job_id[:8]
        for v in result.violations
        if v.kind is Kind.OFF_WINDOW_START
    )
    assert off == sorted(hand["off_the_hour_not_agent"])
    unannounced = sorted(
        v.rows[0].job_id[:8]
        for v in result.violations
        if v.kind is Kind.UNANNOUNCED_START
    )
    assert unannounced == sorted(hand["unannounced"])
    assert unannounced == sorted(fixture["expected"]["unannounced_jobs"])


def test_recreate_audit_each_violation_carries_the_rows_that_show_it() -> None:
    result = DevRecreateAuditReader().handle(_window_request())
    two = [v for v in result.violations if v.kind is Kind.TWO_STARTS_IN_WINDOW]
    twelve = next(v for v in two if v.window_start == _at("2026-11-23T12:00:00Z"))
    assert [(r.requested_by, r.at) for r in twelve.rows] == [
        ("agent/merge-restore", _at("2026-11-23T12:19:38.822271Z")),
        ("lane/gwrun-1255", _at("2026-11-23T12:52:04.921476Z")),
    ]
    off = {
        v.rows[0].requested_by
        for v in result.violations
        if v.kind is Kind.OFF_WINDOW_START
    }
    assert off == {
        "lane/devunstick-2150",
        "lane/rollgate-0420",
        "lane/devsecret-1050",
        "lane/gwrun-1255",
    }
    assert not any(r.startswith(PINNED) for r in off)


def test_recreate_audit_a_hold_announcement_does_not_announce_the_start() -> None:
    """92ab0ad6 named 05:00 as the next recreate 141 s before the 04:46:07Z rollback."""
    result = DevRecreateAuditReader().handle(_window_request())
    rollgate = next(
        v
        for v in result.violations
        if v.kind is Kind.UNANNOUNCED_START and v.rows[0].job_id.startswith("0ca618c3")
    )
    assert rollgate.rows[0].announced is False
    hold = next(
        a for a in _fixture()["announcements"] if a["event_id"].startswith("92ab0ad6")
    )
    assert hold["immediate"] is False


def test_recreate_audit_a_start_with_no_job_record_carries_the_restart_count() -> None:
    result = DevRecreateAuditReader().handle(_window_request())
    found = sorted(
        (v.at, v.restart_count)
        for v in result.violations
        if v.kind is Kind.START_WITHOUT_JOB_RECORD
    )
    assert found == [(_at("2026-11-23T19:36:13Z"), 2), (_at("2026-11-23T20:08:48Z"), 3)]
    assert all(
        v.rows == ()
        for v in result.violations
        if v.kind is Kind.START_WITHOUT_JOB_RECORD
    )


def test_recreate_audit_without_container_readings_no_restart_is_seen() -> None:
    result = DevRecreateAuditReader().handle(_window_request(with_observations=False))
    assert Kind.START_WITHOUT_JOB_RECORD not in _by_kind(result.violations)
    assert len(result.violations) == 11


def test_recreate_audit_a_clean_window_emits_nothing() -> None:
    """Positive control: one announced idle converge a window, on the hour, is quiet."""
    starts, manifests, announcements, observations = [], [], [], []
    for hour in range(6):
        at = datetime(2026, 11, 24, hour, 0, 20, tzinfo=UTC)
        job = f"job-{hour}"
        starts.append(
            {
                "event_id": str(uuid4()),
                "job_id": job,
                "runtime_lane": "dev",
                "started_at": (at + timedelta(minutes=9)).isoformat(),
                "cause": "roll_recreate",
            }
        )
        manifests.append(
            {
                "event_id": str(uuid4()),
                "job_id": job,
                "issuer": "agent/idle-converge",
                "git_ref": "origin/dev",
                "scope": "runtime",
                "requested_at": None,
                "accepted_at": at.isoformat(),
                "open_pr": "",
                "rollback": False,
            }
        )
        announcements.append(
            {
                "event_id": str(uuid4()),
                "runtime_lane": "dev",
                "requested_by": "agent/idle-converge",
                "announced_at": (at - timedelta(seconds=2)).isoformat(),
                "next_recreate_at": at.isoformat(),
                "reason": "window_open",
                "immediate": True,
            }
        )
        observations.append(
            {
                "container": "omninode-runtime",
                "started_at": (at + timedelta(minutes=9, seconds=30)).isoformat(),
                "restart_count": 0,
                "exit_code": None,
            }
        )
    result = DevRecreateAuditReader().handle(
        _request(
            starts=starts,
            manifests=manifests,
            announcements=announcements,
            observations=observations,
        )
    )
    assert result.recreates_read == 6
    assert result.violations == ()


def test_recreate_audit_reading_the_window_twice_names_each_case_once() -> None:
    first = DevRecreateAuditReader().handle(_window_request())
    second = DevRecreateAuditReader().handle(_window_request())
    assert [v.event_id for v in first.violations] == [
        v.event_id for v in second.violations
    ]


class _Bus:
    def __init__(self, *, accepts: bool = True) -> None:
        self.sent: list[tuple[str, str, dict[str, Any]]] = []
        self._accepts = accepts

    def __call__(self, topic: str, key: str, payload: dict[str, Any]) -> bool:
        self.sent.append((topic, key, payload))
        return self._accepts


def _topics() -> dict[str, str]:
    contract = yaml.safe_load((NODE / "contract.yaml").read_text(encoding="utf-8"))
    topics: dict[str, str] = contract["topics"]
    return topics


def test_recreate_audit_publishes_one_event_per_violation_on_the_declared_topic() -> (
    None
):
    bus = _Bus()
    result = audit_and_publish(
        DevRecreateAuditReader(),
        _window_request(),
        DevRollEventsPublisher(bus, _topics()),
    )

    declared = _topics()["dev_recreate_violation"]
    assert len(result.violations) == len(bus.sent) == 13
    assert {topic for topic, _, _ in bus.sent} == {declared}
    assert len({key for _, key, _ in bus.sent}) == 13
    for _, _, payload in bus.sent:
        ModelDevRecreateViolation.model_validate(payload)
    assert _by_kind(
        tuple(ModelDevRecreateViolation.model_validate(p) for _, _, p in bus.sent)
    ) == Counter(
        {
            Kind.TWO_STARTS_IN_WINDOW: 3,
            Kind.OFF_WINDOW_START: 4,
            Kind.UNANNOUNCED_START: 4,
            Kind.START_WITHOUT_JOB_RECORD: 2,
        }
    )


def test_recreate_audit_a_refusing_bus_is_logged_and_never_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bus = _Bus(accepts=False)
    with caplog.at_level(logging.ERROR):
        result = audit_and_publish(
            DevRecreateAuditReader(),
            _window_request(),
            DevRollEventsPublisher(bus, _topics()),
        )
    assert len(result.violations) == 13
    assert caplog.text.count("friction_type=dev_event_publish_failed") == 13


def test_recreate_audit_topic_and_bounds_are_declared_in_the_contract_not_the_code() -> (
    None
):
    contract = yaml.safe_load((NODE / "contract.yaml").read_text(encoding="utf-8"))
    assert (
        contract["topics"]["dev_recreate_violation"]
        == "onex.evt.omnimarket.dev-recreate-violation.v1"
    )
    assert {
        "dev_runtime_started",
        "dev_roll_manifest",
        "dev_recreate_announced",
    } <= set(contract["topics"])
    assert audit_settings(NODE / "contract.yaml") == (1800, 120)
    operations = [h["operation"] for h in contract["handler_routing"]["handlers"]]
    assert "dev_roll.audit_recreates" in operations
    source = AUDIT_SOURCE.read_text(encoding="utf-8")
    assert "onex.evt." not in source
    assert "3600" not in source
    assert "agent/" not in source
    assert "operator/pinned-fix" not in source
