# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18769 — the three lab lane-health emitters, at the seam each promises.

These tests pin the WIRE SHAPE. They deliberately do not attempt to prove a
message reached a broker: that is the live readback, quoted in the PR body, and
a test that mocked a producer would prove neither half. What a test can prove,
and what a live readback cannot, is that the document a consumer folds is the
document this repository produces.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CENSUS_EVENT = REPO_ROOT / "scripts" / "lane_census_event.py"
CENSUS_CHECK = REPO_ROOT / "scripts" / "lane-census-check.sh"
RECEIPT = REPO_ROOT / "scripts" / "ci" / "lab_pass_receipt.py"

pytestmark = pytest.mark.unit


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lane_census_event = _load(CENSUS_EVENT, "omn18769_lane_census_event")
lab_pass_receipt = _load(RECEIPT, "omn18769_lab_pass_receipt")


def _plan(*, drift: int) -> dict[str, Any]:
    return {
        "has_drift": drift > 0,
        "lanes_checked": ["dev", "stability-test"],
        "findings": [
            {
                "lane": "dev",
                "kind": "unexpected_container",
                "container": f"stray-{i}",
                "detail": "running but not declared",
                "severity": "warning",
            }
            for i in range(drift)
        ],
    }


# --------------------------------------------------------------------------
# AC1 — the census fact reaches the bus on EVERY run
# --------------------------------------------------------------------------


def test_ac1_a_clean_run_still_produces_an_observed_event() -> None:
    """The case the drift topic structurally cannot carry.

    Without this document a reducer cannot distinguish "the fleet matches its
    manifest" from "the census stopped running two days ago".
    """
    event = lane_census_event.build_observed_event(
        host="192.168.86.201", plan=_plan(drift=0)
    )

    assert event["event_type"] == "lane-census-observed"
    assert event["topic"] == "onex.evt.omnibase-infra.lane-census-observed.v1"
    assert event["drift_count"] == 0
    assert event["findings"] == []
    assert event["lanes_checked"] == ["dev", "stability-test"]


def test_ac1_the_observed_event_carries_the_same_drift_count_as_the_plan() -> None:
    event = lane_census_event.build_observed_event(host="h", plan=_plan(drift=5))

    assert event["drift_count"] == 5
    assert len(event["findings"]) == 5


def test_ac1_the_observed_event_is_not_an_alert_and_carries_no_ticket_fields() -> None:
    """The drift topic is the alert authority; this one must not impersonate it.

    A consumer that pattern-matches on ``alert_key`` to open a ticket would
    otherwise file one six times a day about a healthy fleet.
    """
    event = lane_census_event.build_observed_event(host="h", plan=_plan(drift=3))

    assert "alert_key" not in event
    assert "ticket_title" not in event
    assert "ticket_body" not in event


def test_ac1_the_drift_event_is_unchanged_by_this_ticket() -> None:
    """The pre-existing alert path keeps its exact shape and topic."""
    event = lane_census_event.build_event(host="h", plan=_plan(drift=2))

    assert event["event_type"] == "lane-census-drift"
    assert event["topic"] == "onex.evt.infra.lane-census-drift.v1"
    assert event["alert_key"].startswith("lane-census-drift:h:")


def test_ac1_the_observed_event_names_when_the_census_looked() -> None:
    """``observed_at``, not ``emitted_at``.

    The consumer keys this fact's freshness on it. ``emitted_at`` would invite
    a republisher to restamp it, making a two-day-old observation read current.
    """
    now = datetime(2026, 9, 18, 23, 0, tzinfo=UTC)
    event = lane_census_event.build_observed_event(
        host="h", plan=_plan(drift=0), now=now
    )

    assert event["observed_at"] == now.isoformat()
    assert "emitted_at" not in event


def test_ac1_the_cli_default_is_still_the_drift_event() -> None:
    """--observed is additive; every existing caller is byte-for-byte unchanged."""
    plan = json.dumps(_plan(drift=1))
    env = {"LANE_CENSUS_HOST": "h", "PATH": "/usr/bin:/bin"}

    default = subprocess.run(
        [sys.executable, str(CENSUS_EVENT)],
        input=plan,
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    observed = subprocess.run(
        [sys.executable, str(CENSUS_EVENT), "--observed"],
        input=plan,
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    assert json.loads(default.stdout)["event_type"] == "lane-census-drift"
    assert json.loads(observed.stdout)["event_type"] == "lane-census-observed"


def test_ac1_the_observed_document_is_written_before_the_no_drift_exit() -> None:
    """A write placed after the early exit would never fire on a clean run.

    Asserted on source ORDER rather than by running the script, which needs a
    lab host: the ordering is the whole mechanism, and getting it wrong looks
    identical to working until the fleet is healthy.
    """
    body = CENSUS_CHECK.read_text()
    write_at = body.index('log "census-observed document written: $OBSERVED_OUT"')
    exit_at = body.index('log "No lane drift. Desired == actual."')

    assert write_at < exit_at


def test_ac1_the_census_script_does_not_publish_and_says_why() -> None:
    """The publish moved OUT of this script, and that is the mechanism.

    `rpk` is not on the .201 host PATH -- it lives inside the broker container,
    and the live census log has read `rpk not found` on every run since the
    drift event was added -- and the hourly drop-in sets no broker address. A
    branch that can only warn is not a mechanism. This pins that the script
    writes a document and opens no transport of its own, so a later change
    cannot quietly reintroduce the unreachable branch.
    """
    body = CENSUS_CHECK.read_text()

    assert 'rpk topic produce "$OBSERVED_TOPIC"' not in body
    assert "localhost:9092" not in body
    assert "publish_lab_fact_event.py" in body, (
        "the script must name where publication actually happens"
    )


def test_ac1_the_refresh_workflow_publishes_the_observed_document() -> None:
    """The caller that HAS the transport is the one that publishes it."""
    workflow = (REPO_ROOT / ".github/workflows/lane-census-refresh.yml").read_text()

    assert "--observed-out" in workflow
    assert "scripts/ci/publish_lab_fact_event.py" in workflow
    assert "config/ci_bus_lanes.yaml" in workflow
    assert "rpk topic produce" not in workflow


def test_ac1_the_refresh_publish_never_fails_the_census_collection() -> None:
    """A broker that is down is not a lane that drifted.

    The census file is the durable record; the publish is renderability. If the
    publish could fail the job, an unreachable broker would stop the committed
    census from ever being refreshed -- trading the surface this ticket is
    making readable for the one that already worked.
    """
    workflow = (REPO_ROOT / ".github/workflows/lane-census-refresh.yml").read_text()
    publish_at = workflow.index("Publish the census-observed fact to the bus")
    tail = workflow[publish_at : publish_at + 600]

    assert "continue-on-error: true" in tail
    assert "if: always()" in tail


# --------------------------------------------------------------------------
# AC2 — the runtime names its own lane
# --------------------------------------------------------------------------


def test_ac2_the_runtime_resolves_its_declared_lane() -> None:
    from omnibase_infra.runtime.health.runtime_lane_identity import resolve_runtime_lane

    assert resolve_runtime_lane({"ONEX_RUNTIME_LANE": "compose-dev"}) == "compose-dev"
    assert resolve_runtime_lane({"ONEX_RUNTIME_LANE": " Onex-Lab "}) == "onex-lab"


def test_ac2_an_undeclared_lane_is_none_not_a_guess() -> None:
    from omnibase_infra.runtime.health.runtime_lane_identity import resolve_runtime_lane

    assert resolve_runtime_lane({}) is None
    assert resolve_runtime_lane({"ONEX_RUNTIME_LANE": ""}) is None


def test_ac2_an_unknown_lane_name_is_refused_rather_than_passed_through() -> None:
    """A typo would otherwise mint a phantom lane indistinguishable from a real one."""
    from omnibase_infra.runtime.health.runtime_lane_identity import resolve_runtime_lane

    assert resolve_runtime_lane({"ONEX_RUNTIME_LANE": "compose-dv"}) is None
    assert resolve_runtime_lane({"ONEX_RUNTIME_LANE": "prod"}) is None


def test_ac6_a_runtime_cannot_claim_a_lane_outside_the_lab() -> None:
    """stability-test, judge and the collaborator lane are read-only surfaces."""
    from omnibase_infra.runtime.health.runtime_lane_identity import (
        KNOWN_LANES,
        resolve_runtime_lane,
    )

    assert {"compose-dev", "onex-lab", "onex-lab-k3s"} == KNOWN_LANES
    for lane in ("stability-test", "judge", "lakshman"):
        assert resolve_runtime_lane({"ONEX_RUNTIME_LANE": lane}) is None


def test_ac2_the_health_event_carries_the_lane_and_defaults_to_none() -> None:
    """Nullable on purpose: a required field would stop every deployed runtime.

    An already-deployed runtime that does not set the variable must keep
    emitting health events -- turning an observability improvement into an
    outage is not an acceptable price for a non-null column.
    """
    from uuid import uuid4

    from omnibase_infra.models.health.model_runtime_health_check_event import (
        ModelRuntimeHealthCheckEvent,
    )

    without = ModelRuntimeHealthCheckEvent(
        correlation_id=uuid4(), timestamp=datetime.now(UTC), status="HEALTHY"
    )
    assert without.lane is None

    with_lane = ModelRuntimeHealthCheckEvent(
        correlation_id=uuid4(),
        timestamp=datetime.now(UTC),
        status="DEGRADED",
        lane="compose-dev",
    )
    assert with_lane.lane == "compose-dev"


# --------------------------------------------------------------------------
# AC3 — the lab-pass verdict reaches the bus, FAIL included
# --------------------------------------------------------------------------


def _receipt(*, ok: bool) -> Any:
    started = datetime(2026, 9, 18, 22, 0, tzinfo=UTC)
    checks = [
        lab_pass_receipt.ModelLabPassCheck(
            name="ready_main", ok=True, evidence="HTTP_200"
        ),
        lab_pass_receipt.ModelLabPassCheck(
            name="ready_effects", ok=ok, evidence="HTTP_200" if ok else "HTTP_503"
        ),
    ]
    return lab_pass_receipt.build_receipt(
        sha="a" * 40,
        lane=lab_pass_receipt.EnumLabLane.COMPOSE_DEV,
        started_at=started,
        finished_at=started,
        checks=checks,
        agent_command_id=None,
    )


def test_ac3_a_fail_receipt_produces_a_bus_event_naming_the_failing_checks() -> None:
    event = lab_pass_receipt.build_bus_event(_receipt(ok=False))

    assert event["topic"] == "onex.evt.omnibase-infra.lab-pass-receipt.v1"
    assert event["result"] == "FAIL"
    assert event["failing_checks"] == ["ready_effects"]
    assert event["lane"] == "compose-dev"
    assert event["sha"] == "a" * 40


def test_ac3_a_pass_receipt_produces_an_event_with_no_failing_checks() -> None:
    event = lab_pass_receipt.build_bus_event(_receipt(ok=True))

    assert event["result"] == "PASS"
    assert event["failing_checks"] == []


def test_ac3_the_event_carries_the_receipts_verdict_and_never_re_derives_it() -> None:
    """The receipt model derives the verdict from the checks so an emitter
    cannot record green over a failed readback. Re-deriving here would put that
    seam back."""
    receipt = _receipt(ok=False)
    event = lab_pass_receipt.build_bus_event(receipt)

    assert event["result"] == receipt.result.value
    assert [c["name"] for c in event["checks"]] == [c.name for c in receipt.checks]


def test_ac3_the_event_points_back_at_the_artifact_that_is_the_evidence() -> None:
    """The event makes the verdict renderable; the artifact stays the evidence."""
    receipt = _receipt(ok=True)
    event = lab_pass_receipt.build_bus_event(receipt)

    assert event["artifact_name"] == lab_pass_receipt.artifact_name(
        receipt.lane, receipt.sha
    )


def test_ac3_emit_writes_the_event_document_beside_the_artifact(tmp_path: Path) -> None:
    out = tmp_path / "receipt.json"
    event_out = tmp_path / "event.json"
    started = "2026-09-18T22:00:00+00:00"

    completed = subprocess.run(
        [
            sys.executable,
            str(RECEIPT),
            "emit",
            "--sha",
            "b" * 40,
            "--lane",
            "compose-dev",
            "--started-at",
            started,
            "--finished-at",
            started,
            "--check",
            "ready_main:ok:HTTP_200",
            "--agent-command-id",
            "",
            "--out",
            str(out),
            "--event-out",
            str(event_out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert out.exists()
    event = json.loads(event_out.read_text())
    assert event["result"] == "PASS"
    assert event["lane"] == "compose-dev"


def test_ac3_emit_writes_the_event_for_a_failing_lab_pass_too(tmp_path: Path) -> None:
    """The emit step goes red on a FAIL, and the event is written anyway.

    A record that exists only on success cannot distinguish a failed lab pass
    from one nobody ran.
    """
    out = tmp_path / "receipt.json"
    event_out = tmp_path / "event.json"
    started = "2026-09-18T22:00:00+00:00"

    completed = subprocess.run(
        [
            sys.executable,
            str(RECEIPT),
            "emit",
            "--sha",
            "c" * 40,
            "--lane",
            "compose-dev",
            "--started-at",
            started,
            "--finished-at",
            started,
            "--check",
            "ready_main:fail:HTTP_503",
            "--agent-command-id",
            "",
            "--out",
            str(out),
            "--event-out",
            str(event_out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1, "a FAIL receipt must still fail the step"
    event = json.loads(event_out.read_text())
    assert event["result"] == "FAIL"
    assert event["failing_checks"] == ["ready_main"]


def test_ac3_this_script_never_opens_a_broker_connection_itself() -> None:
    """Publishing is the job's business, not this script's.

    The emitting jobs differ in whether they can reach a broker at all, and a
    publish failure must never fail a lab pass that genuinely ran.
    """
    body = RECEIPT.read_text()

    assert "rpk topic produce" not in body
    assert "KafkaProducer" not in body
    assert "--event-out" in body


# --------------------------------------------------------------------------
# AC3 (transport) — the event document is published on the LANE-DECLARED
# transport, and a broker that cannot be reached never fails the lab pass
# --------------------------------------------------------------------------
#
# WHY THIS SECTION EXISTS. Writing the event document is not publishing it.
# The first revision of this change published it with a bare
# `rpk topic produce --brokers "$KAFKA_BOOTSTRAP_SERVERS"` step, and both
# halves of that were wrong on the job it ran in: nothing in
# runtime-rebuild-trigger.yml sets KAFKA_BOOTSTRAP_SERVERS, so the step could
# only ever take its own "unset" branch, and the .201 dev-lane Redpanda
# EXTERNAL listener has required SASL/SCRAM-SHA-256 since OMN-18012 Phase B,
# which a bare `--brokers` invocation does not speak. A step that can only
# warn is not a mechanism; it is a comment that runs.

PUBLISHER = REPO_ROOT / "scripts" / "ci" / "publish_lab_fact_event.py"
REBUILD_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "runtime-rebuild-trigger.yml"


def test_ac3_the_publisher_exists_as_its_own_script() -> None:
    """The broker concern lives beside the emitter, not inside it.

    ``lab_pass_receipt.py`` stays a pure document builder -- the test above
    pins that -- so the transport gets its own file rather than being folded
    into the receipt model's module.
    """
    assert PUBLISHER.is_file()


def test_ac3_the_publisher_never_infers_a_transport() -> None:
    """OMN-18012: credential PRESENCE is not a statement about transport.

    The protocol and mechanism come from the checked-in lane overlay and from
    nowhere else. A publisher that picked SASL_SSL because credentials happened
    to be in the environment is the exact 2026-09-07 outage.
    """
    body = PUBLISHER.read_text()
    assert "resolve_ci_bus_security" in body
    assert "build_kafka_producer_config" in body
    # Read the CODE, not the prose: the docstring names SASL_SSL to record the
    # outage this design exists to avoid, which a naive substring check would
    # flag. What must be absent is any statement that SETS a transport here.
    code = "\n".join(
        line for line in body.splitlines() if not line.lstrip().startswith("#")
    )
    code = code.split('"""', 2)[-1]
    assert "security.protocol" not in code
    assert "sasl.mechanisms" not in code
    assert "SASL_SSL" not in code


def test_ac3_the_publisher_never_defaults_a_broker_address() -> None:
    """Rule 8: no localhost fallback, no `or "localhost:19092"`."""
    body = PUBLISHER.read_text()
    assert "localhost:19092" not in body
    assert "127.0.0.1" not in body


def test_ac3_an_unreachable_broker_reports_zero_and_never_fails_the_lab_pass() -> None:
    """A broker that is down is not a lab pass that failed.

    The publisher reports a FACT about a run that has already happened. Its
    exit code is 0 on every publish-side failure, and the failure is loud in
    the log. The artifact -- the authority the delivery gate reads -- is
    unaffected either way.
    """
    publisher = _load(PUBLISHER, "omn18769_publish_lab_fact_event")
    assert publisher.EXIT_OK == 0
    assert publisher.main(["--event", "/nonexistent/event.json"]) == 0


def test_ac3_the_workflow_calls_the_publisher_and_not_raw_rpk() -> None:
    """The step that publishes is the one that can actually publish."""
    body = REBUILD_WORKFLOW.read_text()
    assert "scripts/ci/publish_lab_fact_event.py" in body
    assert "rpk topic produce onex.evt.omnibase-infra.lab-pass-receipt.v1" not in body


def test_ac3_the_publishing_step_is_given_the_sasl_credentials() -> None:
    """A SASL lane cannot be published to without them.

    The publisher refuses to downgrade the declared transport, so omitting the
    credentials here would turn every publish into a loud no-op -- which is
    precisely the failure this section was written to remove.
    """
    body = REBUILD_WORKFLOW.read_text()
    publish_step = body.split("Publish the compose-dev lab-pass verdict to the bus", 1)
    assert len(publish_step) == 2, "the publishing step must still exist"
    step = publish_step[1].split("- name:", 1)[0]
    assert "KAFKA_SASL_USERNAME" in step
    assert "KAFKA_SASL_PASSWORD" in step
    assert "--bus-lane" in step
    assert "--bus-overlay" in step
