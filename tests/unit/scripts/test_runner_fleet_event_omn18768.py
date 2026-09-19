# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18768 AC1 — the runner monitor's fleet-observation bus event builder.

The runner monitor posted its findings to a chat channel and emitted NO bus
event (OMN-16943), so nothing downstream — alert triage, the dashboard, a
projection — could see the fleet at all. These tests pin the builder that
closes that: one typed event per observation cycle carrying every runner's
name, label class, host, status and, when busy, the job it is running.

The properties pinned here are the ones a downstream projection cannot
reconstruct if the emitter gets them wrong:

  * an OFFLINE runner is REPORTED, never omitted (AC4). Omitting it is how a
    fleet outage reads as a smaller healthy fleet.
  * one event per CYCLE, not per runner — a ~69-runner fleet is one message.
  * the label CLASS is carried, because which class is down is the whole
    operational question (a `omnibase-prod-deploy` outage is not a
    `omnibase-ci` outage).
  * `current_job_id` is NULL when unresolved and is never invented — an absent
    job id and "idle" are different facts.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

# The builder lives in scripts/ and the monitor resolves it at ../../scripts/
# from its own deployed location. deploy-runners.sh rsyncs it there; a builder
# present in the repo and absent from that sync set is a fleet that is silently
# unobservable in production while every test here passes, which is why the
# sync set is itself asserted in the behavioural tests.
_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
_spec = importlib.util.spec_from_file_location(
    "runner_fleet_event", _SCRIPTS / "runner_fleet_event.py"
)
assert _spec and _spec.loader
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

NOW = datetime(2026, 9, 18, 23, 30, 0, tzinfo=UTC)
TOPIC = "onex.evt.infra.runner-fleet.v1"
HOST = "omni-201"
GROUP = "omnibase-ci"


def _runner(
    name: str,
    status: str = "online",
    busy: bool = False,
    labels: list[str] | None = None,
    runner_id: int = 1,
) -> dict[str, object]:
    return {
        "id": runner_id,
        "name": name,
        "status": status,
        "busy": busy,
        "labels": [{"name": label} for label in (labels or ["self-hosted", GROUP])],
    }


def _build(runners: list[dict[str, object]], **kwargs: object) -> dict[str, object]:
    params: dict[str, object] = {
        "runners_payload": {"total_count": len(runners), "runners": runners},
        "host": HOST,
        "name_prefix": "omninode-runner",
        "runner_group": GROUP,
        "topic": TOPIC,
        "now": NOW,
    }
    params.update(kwargs)
    return mod.build_event(**params)  # type: ignore[arg-type]


@pytest.mark.unit
class TestRunnerFleetEventShape:
    def test_one_event_carries_every_runner_in_the_cycle(self) -> None:
        """AC1 — a ~69-runner fleet is ONE message, not 69."""
        event = _build(
            [_runner(f"omninode-runner-{i}", runner_id=i) for i in range(1, 70)]
        )
        assert event["event_type"] == "runner-fleet-observation"
        assert event["topic"] == TOPIC
        assert len(event["runners"]) == 69
        assert event["runner_count"] == 69

    def test_every_runner_carries_name_class_host_status_and_observed_at(self) -> None:
        """AC1 falsifier — the message carries per-runner status and labels."""
        event = _build([_runner("omninode-runner-1", runner_id=7)])
        row = event["runners"][0]
        assert row["runner_name"] == "omninode-runner-1"
        assert row["label_class"] == GROUP
        assert row["labels"] == ["self-hosted", GROUP]
        assert row["host"] == HOST  # no host-<id> label on this fixture
        assert row["observing_host"] == HOST
        assert row["status"] == "online"
        assert row["observed_at"] == NOW.isoformat()
        assert row["runner_id"] == 7

    def test_the_payload_is_json_serializable(self) -> None:
        event = _build([_runner("omninode-runner-1")])
        assert json.loads(json.dumps(event))["runner_count"] == 1


@pytest.mark.unit
class TestOfflineRunnersAreReportedNotOmitted:
    def test_an_offline_runner_appears_with_status_offline(self) -> None:
        """AC4 — an offline runner is reported offline rather than omitted."""
        event = _build(
            [
                _runner("omninode-runner-1", status="online", runner_id=1),
                _runner("omninode-runner-2", status="offline", runner_id=2),
            ]
        )
        by_name = {r["runner_name"]: r for r in event["runners"]}
        assert by_name["omninode-runner-2"]["status"] == "offline"
        assert event["runner_count"] == 2
        assert event["offline_count"] == 1
        assert event["online_count"] == 1

    def test_a_fleet_that_is_entirely_offline_is_a_full_row_set_not_an_empty_one(
        self,
    ) -> None:
        """The failure this guards: an outage rendering as a smaller healthy fleet."""
        event = _build(
            [
                _runner(f"omninode-runner-{i}", status="offline", runner_id=i)
                for i in range(1, 6)
            ]
        )
        assert event["runner_count"] == 5
        assert event["offline_count"] == 5
        assert event["online_count"] == 0
        assert all(r["status"] == "offline" for r in event["runners"])


@pytest.mark.unit
class TestBusyAndJobIdentity:
    def test_a_busy_runner_reports_status_busy_not_online(self) -> None:
        """`busy` is a distinct operational state; folding it into `online`
        makes "the fleet is saturated" indistinguishable from "the fleet is
        idle", which is the question a capacity panel exists to answer."""
        event = _build([_runner("omninode-runner-1", status="online", busy=True)])
        assert event["runners"][0]["status"] == "busy"
        assert event["busy_count"] == 1
        assert event["online_count"] == 1  # busy runners are still online

    def test_a_busy_runner_carries_its_job_id_when_resolved(self) -> None:
        event = _build(
            [_runner("omninode-runner-1", busy=True)],
            job_by_runner={"omninode-runner-1": 4242},
        )
        assert event["runners"][0]["current_job_id"] == "4242"

    def test_an_unresolved_job_id_is_null_and_never_invented(self) -> None:
        """An absent job id and "idle" are different facts. The org runners API
        does not carry a job id, so an unresolved busy runner reports NULL."""
        event = _build([_runner("omninode-runner-1", busy=True)])
        assert event["runners"][0]["current_job_id"] is None

    def test_an_idle_runner_never_carries_a_job_id(self) -> None:
        event = _build(
            [_runner("omninode-runner-1", busy=False)],
            job_by_runner={"omninode-runner-1": 4242},
        )
        assert event["runners"][0]["current_job_id"] is None


@pytest.mark.unit
class TestLabelClassRollup:
    def test_the_class_rollup_counts_each_class_separately(self) -> None:
        """Which class is down is the operational question; a fleet-wide
        healthy count hides a total outage of a single-runner class."""
        event = _build(
            [
                _runner(
                    "omninode-runner-1",
                    labels=["self-hosted", "omnibase-ci"],
                    runner_id=1,
                ),
                _runner(
                    "omninode-runner-2",
                    labels=["self-hosted", "omnibase-ci"],
                    runner_id=2,
                ),
                _runner(
                    "omninode-runner-3",
                    status="offline",
                    labels=["self-hosted", "omnibase-prod-deploy"],
                    runner_id=3,
                ),
            ]
        )
        rollup = {entry["label_class"]: entry for entry in event["class_rollup"]}
        assert rollup["omnibase-ci"]["online"] == 2
        assert rollup["omnibase-ci"]["offline"] == 0
        assert rollup["omnibase-prod-deploy"]["online"] == 0
        assert rollup["omnibase-prod-deploy"]["offline"] == 1
        assert rollup["omnibase-prod-deploy"]["total"] == 1

    def test_the_class_is_the_first_known_fleet_class_not_self_hosted(self) -> None:
        """`self-hosted` is on every runner and classifies nothing."""
        event = _build(
            [
                _runner(
                    "omninode-runner-1",
                    labels=["self-hosted", "Linux", "X64", "omnibase-verify"],
                )
            ]
        )
        assert event["runners"][0]["label_class"] == "omnibase-verify"

    def test_a_runner_with_no_known_class_label_is_classed_unknown_not_dropped(
        self,
    ) -> None:
        event = _build([_runner("omninode-runner-1", labels=["self-hosted", "Linux"])])
        assert event["runners"][0]["label_class"] == "unclassified"
        assert event["runner_count"] == 1


@pytest.mark.unit
class TestFleetScoping:
    def test_a_runner_outside_the_name_prefix_is_excluded(self) -> None:
        """The monitor watches one named fleet; a foreign org runner is not
        this fleet's liveness and must not dilute its counts."""
        event = _build(
            [
                _runner("omninode-runner-1", runner_id=1),
                _runner("some-other-fleet-9", runner_id=9),
            ]
        )
        assert event["runner_count"] == 1
        assert event["runners"][0]["runner_name"] == "omninode-runner-1"

    def test_runners_are_emitted_in_a_deterministic_order(self) -> None:
        """Replay determinism: the same observation builds the same event."""
        runners = [_runner(f"omninode-runner-{i}", runner_id=i) for i in (3, 1, 2)]
        first = _build(runners)
        second = _build(list(reversed(runners)))
        assert first == second


@pytest.mark.unit
class TestRefusals:
    def test_an_unparseable_runners_payload_is_refused_loudly(self) -> None:
        """A monitor that cannot see the fleet must fail loudly. Emitting an
        empty fleet would read downstream as "every runner is gone"."""
        with pytest.raises(ValueError, match="runners"):
            mod.build_event(
                runners_payload={"total_count": 0},
                host=HOST,
                name_prefix="omninode-runner",
                runner_group=GROUP,
                topic=TOPIC,
                now=NOW,
            )

    def test_a_missing_host_attribution_is_refused(self) -> None:
        with pytest.raises(ValueError, match="host"):
            _build([_runner("omninode-runner-1")], host="")

    def test_a_runner_with_no_name_is_refused_rather_than_silently_skipped(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="name"):
            _build([{"id": 1, "status": "online", "busy": False, "labels": []}])


@pytest.mark.unit
class TestHostAttribution:
    """Measured against the live org pool, 2026-09-18.

    Runners carry their own host as a `host-<id>` label. On that date the one
    offline runner in the whole pool was `omninode-air-runner-1`, labelled
    `host-105` — a machine the observing host (.201) has no visibility into.
    Attributing it to the observer would point an operator at the wrong box.
    """

    def test_a_runner_with_a_host_label_reports_its_own_host(self) -> None:
        event = _build(
            [
                _runner(
                    "omninode-runner-1",
                    labels=[
                        "self-hosted",
                        "Linux",
                        "ARM64",
                        "omnibase-verify",
                        "host-105",
                    ],
                )
            ]
        )
        row = event["runners"][0]
        assert row["host"] == "host-105"
        assert row["observing_host"] == HOST

    def test_a_runner_with_no_host_label_falls_back_to_the_observing_host(self) -> None:
        event = _build([_runner("omninode-runner-1", labels=["self-hosted", GROUP])])
        row = event["runners"][0]
        assert row["host"] == HOST
        assert row["observing_host"] == HOST

    def test_a_bare_host_prefix_is_not_a_host(self) -> None:
        event = _build([_runner("omninode-runner-1", labels=["self-hosted", "host-"])])
        assert event["runners"][0]["host"] == HOST


@pytest.mark.unit
class TestFleetPrefixCoversTheWholePool:
    """The fleet prefix is BROADER than the monitor's detection prefix, and
    this is the test that says why.

    Measured live 2026-09-18: the org pool held 69 runners, 68 online and
    exactly one offline — `omninode-air-runner-1`. The detection loop's
    `RUNNER_NAME_PREFIX` is `omninode-runner`, which that name does not start
    with, so scoping the fleet observation to the detection prefix would have
    dropped the ONLY runner that was down. AC4 says an offline runner is
    reported rather than omitted; that is not satisfiable on the narrow prefix.
    """

    LIVE_POOL_NAMES = (
        "omninode-runner-1",
        "omninode-air-runner-1",
        "omninode-mini-runner-1",
        "omninode-verify-runner-1",
        "omninode-customer-plane-runner-1",
        "omninode-deploy-runner",
        "omninode-prod-deploy-runner-1",
    )

    def test_the_detection_prefix_would_drop_the_offline_runner(self) -> None:
        """The defect, stated as a test so it cannot come back by a rename."""
        event = _build(
            [
                _runner("omninode-runner-1", runner_id=1),
                _runner("omninode-air-runner-1", status="offline", runner_id=2),
            ],
            name_prefix="omninode-runner",
        )
        assert event["offline_count"] == 0
        assert event["runner_count"] == 1

    def test_the_fleet_prefix_keeps_every_class_in_the_live_pool(self) -> None:
        event = _build(
            [
                _runner(name, runner_id=i, labels=["self-hosted", "omnibase-verify"])
                for i, name in enumerate(self.LIVE_POOL_NAMES, start=1)
            ],
            name_prefix="omninode-",
        )
        assert event["runner_count"] == len(self.LIVE_POOL_NAMES)
        assert {r["runner_name"] for r in event["runners"]} == set(self.LIVE_POOL_NAMES)
