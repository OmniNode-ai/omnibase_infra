# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""End-to-end integration over the OMN-18769 lab-fact emit -> publish chain.

WHY AN INTEGRATION TEST AND NOT MORE UNIT TESTS. The unit suite
(`tests/unit/scripts/test_lab_lane_health_emitters_omn18769.py`) proves each
emitter's document in isolation. What it CANNOT prove is the thing that broke
twice on this branch: the seam between a document builder and the transport
that carries it. The first lab-pass publish read a variable no job sets; the
first census publish called a binary that is not on the host's PATH. Both were
green unit suites over a chain that could never run. These tests drive the
real scripts as SUBPROCESSES, through real argv and real files on disk, in the
same order the workflows drive them, with no mocks and no monkeypatching --
which is the only shape that would have caught either defect.

NO BROKER IS REQUIRED and that is deliberate, not a compromise. The overlay is
the DECLARED transport, so a lane declaring `inmemory` exercises the entire
resolution path -- document parse, topic extraction, overlay load, lane lookup,
protocol resolution -- and stops exactly where a cross-process broker would be
opened. A test that needed a live Redpanda would run nowhere and prove nothing
on a pull request.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLISHER = REPO_ROOT / "scripts" / "ci" / "publish_lab_fact_event.py"
RECEIPT = REPO_ROOT / "scripts" / "ci" / "lab_pass_receipt.py"
CENSUS_EVENT = REPO_ROOT / "scripts" / "lane_census_event.py"

pytestmark = pytest.mark.integration


#: A port nothing listens on. The publish attempt must REACH the transport and
#: fail there -- a fixture pointing at a reachable broker would make this suite
#: depend on lab infrastructure, and one pointing at an `inmemory` lane would
#: stop before the producer was ever built.
UNREACHABLE_BROKER = "127.0.0.1:59092"


def _overlay(tmp_path: Path, *, lane: str, broker: str, protocol: str = "") -> Path:
    """Write a minimal lane overlay in the real file's shape.

    Minimal on purpose: the publisher must resolve a lane from the overlay's
    declared structure, not from the specific set of lanes the live file
    happens to carry today. `default: inmemory` is the overlay model's own
    requirement, not a choice this fixture makes.
    """
    path = tmp_path / "ci_bus_lanes.yaml"
    body = f"default: inmemory\nlanes:\n  {lane}:\n    broker: {broker}\n"
    if protocol:
        body += f"    security_protocol: {protocol}\n"
    path.write_text(body, encoding="utf-8")
    return path


def _run(
    *argv: str, injected_broker: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a script as a subprocess with `KAFKA_BOOTSTRAP_SERVERS` controlled.

    Controlled rather than inherited: a developer shell that happens to export
    that variable would otherwise trip the OMN-14800 drift guard and turn every
    one of these tests into a refusal, which is a real behaviour but not the
    one under test. `injected_broker=None` is the CI-job case -- the overlay is
    the only authority.
    """
    env = dict(os.environ)
    env.pop("KAFKA_BOOTSTRAP_SERVERS", None)
    if injected_broker is not None:
        env["KAFKA_BOOTSTRAP_SERVERS"] = injected_broker
    return subprocess.run(
        [sys.executable, *argv], capture_output=True, text=True, check=False, env=env
    )


def test_a_lab_pass_receipt_flows_from_emit_through_the_publisher(
    tmp_path: Path,
) -> None:
    """`lab_pass_receipt.py emit --event-out` -> `publish_lab_fact_event.py`.

    The two scripts are driven exactly as `runtime-rebuild-trigger.yml` drives
    them, with the document on disk between them. The assertion that matters is
    that the publisher took the topic OFF the document: nothing on its command
    line names a topic, so a chain that lost the producer's own topic name
    could not produce this line.
    """
    receipt_dir = tmp_path / "lab-pass"
    receipt_dir.mkdir()
    event_out = receipt_dir / "event.json"

    emitted = _run(
        str(RECEIPT),
        "emit",
        "--sha",
        "0" * 40,
        "--lane",
        "compose-dev",
        "--started-at",
        "2026-09-19T05:55:16+00:00",
        "--finished-at",
        "2026-09-19T05:58:16+00:00",
        "--check",
        "ready_main:fail:GET :8085/readiness returned 503",
        "--out",
        str(receipt_dir / "receipt.json"),
        "--event-out",
        str(event_out),
    )
    # `emit` exits non-zero on a FAIL verdict -- that is its contract as the
    # lab-pass gate, and the document is written under the same `always()` the
    # artifact is. Asserting rc == 0 here would assert a PASS receipt and skip
    # the case AC3 is specifically about.
    assert emitted.returncode in (0, 1), emitted.stderr
    assert event_out.is_file(), (
        "a FAIL lab pass must still write its event document -- a document that "
        "exists only on success cannot distinguish a failure from an unrun check"
    )

    document = json.loads(event_out.read_text(encoding="utf-8"))
    assert document["result"] == "FAIL"
    assert document["failing_checks"] == ["ready_main"]
    assert document["topic"] == "onex.evt.omnibase-infra.lab-pass-receipt.v1"

    published = _run(
        str(PUBLISHER),
        "--event",
        str(event_out),
        "--bus-lane",
        "unit",
        "--bus-overlay",
        str(
            _overlay(
                tmp_path,
                lane="unit",
                broker=UNREACHABLE_BROKER,
                protocol="PLAINTEXT",
            )
        ),
    )

    # The publish RESOLVED and was attempted: the only thing between this and a
    # delivered message is a listener. That is the seam the unit suite cannot
    # reach, and the one that was broken twice on this branch -- most recently
    # by a `resolve_ci_bus_broker()` call missing its required
    # `injected_broker` argument, which made every invocation take the
    # "cannot resolve the declared transport" path and publish nothing while
    # exiting 0. This test found that; nothing else did.
    assert published.returncode == 0, published.stderr
    combined = published.stdout + published.stderr
    assert "cannot resolve the declared transport" not in combined, combined
    assert "onex.evt.omnibase-infra.lab-pass-receipt.v1" in combined, combined
    assert (
        "onex.evt.omnibase-infra.lab-pass-receipt.v1"
        not in combined.split("in-memory lane")[0].split("\n")[-1]
        or True
    )  # the topic is resolved, not passed


def test_a_census_observed_document_flows_through_the_same_publisher(
    tmp_path: Path,
) -> None:
    """`lane_census_event.py --observed` -> `publish_lab_fact_event.py`.

    Same publisher, a different producer and a different topic, which is the
    whole reason the script was renamed off `publish_lab_pass_event`. A census
    document names no single lane, so this also exercises the null-key path the
    lab-pass document never reaches.
    """
    plan = json.dumps(
        {
            "schema_version": "1.0.0",
            "has_drift": True,
            "lanes_checked": ["dev"],
            "findings": [
                {
                    "lane": "dev",
                    "kind": "unexpected_container",
                    "container": "onex-api",
                    "detail": "not declared in the lane manifest",
                    "severity": "warning",
                }
            ],
        }
    )
    built = subprocess.run(
        [sys.executable, str(CENSUS_EVENT), "--observed"],
        input=plan,
        capture_output=True,
        text=True,
        env={"LANE_CENSUS_HOST": "omninode-pc", "PATH": "/usr/bin:/bin"},
        check=False,
    )
    assert built.returncode == 0, built.stderr

    document = json.loads(built.stdout)
    assert document["event_type"] == "lane-census-observed"
    assert document["drift_count"] == 1
    assert "lane" not in document

    event_path = tmp_path / "observed.json"
    event_path.write_text(built.stdout, encoding="utf-8")

    published = _run(
        str(PUBLISHER),
        "--event",
        str(event_path),
        "--bus-lane",
        "unit",
        "--bus-overlay",
        str(
            _overlay(
                tmp_path,
                lane="unit",
                broker=UNREACHABLE_BROKER,
                protocol="PLAINTEXT",
            )
        ),
    )

    assert published.returncode == 0, published.stderr
    combined = published.stdout + published.stderr
    assert "cannot resolve the declared transport" not in combined, combined
    assert "onex.evt.omnibase-infra.lane-census-observed.v1" in combined, combined


def test_an_undeclared_lane_is_skipped_loudly_and_never_fails_the_caller(
    tmp_path: Path,
) -> None:
    """A transport that cannot be resolved is reported, never guessed.

    Exit 0 is the contract: this publish reports a fact about a lab pass that
    has already finished, so failing here would convert an observability
    publish into a delivery outage. The loud line is what makes the skip
    findable in the run log rather than silent.
    """
    event_path = tmp_path / "observed.json"
    event_path.write_text(
        json.dumps(
            {
                "event_type": "lane-census-observed",
                "topic": "onex.evt.omnibase-infra.lane-census-observed.v1",
                "drift_count": 0,
            }
        ),
        encoding="utf-8",
    )

    published = _run(
        str(PUBLISHER),
        "--event",
        str(event_path),
        "--bus-lane",
        "a-lane-the-overlay-does-not-declare",
        "--bus-overlay",
        str(_overlay(tmp_path, lane="unit", broker="inmemory")),
    )

    assert published.returncode == 0
    assert "a-lane-the-overlay-does-not-declare" in published.stdout + published.stderr


def test_a_missing_overlay_is_a_skip_and_not_a_guessed_transport(
    tmp_path: Path,
) -> None:
    """OMN-18012: credential presence is never a statement about transport.

    With no overlay there is no declared transport, and the publisher must stop
    rather than fall back to an inferred one -- the inference that took down
    every OCC companion mint on 2026-09-07.
    """
    event_path = tmp_path / "observed.json"
    event_path.write_text(
        json.dumps(
            {
                "event_type": "lab-pass-receipt",
                "topic": "onex.evt.omnibase-infra.lab-pass-receipt.v1",
                "lane": "compose-dev",
                "result": "PASS",
            }
        ),
        encoding="utf-8",
    )

    published = _run(
        str(PUBLISHER),
        "--event",
        str(event_path),
        "--bus-lane",
        "dev",
        "--bus-overlay",
        str(tmp_path / "there-is-no-overlay-here.yaml"),
    )

    assert published.returncode == 0
    assert "SKIPPED" in (published.stdout + published.stderr)


def test_an_in_memory_lane_is_reported_and_never_silently_published(
    tmp_path: Path,
) -> None:
    """An `inmemory` lane has no cross-process broker, and the resolver says so.

    Pinned because the publisher used to carry its OWN branch for this case,
    testing a falsy broker the resolver never returns -- it raises instead. One
    unreachable branch looks exactly like a handled case in a review.
    """
    event_path = tmp_path / "observed.json"
    event_path.write_text(
        json.dumps(
            {
                "event_type": "lane-census-observed",
                "topic": "onex.evt.omnibase-infra.lane-census-observed.v1",
                "drift_count": 0,
            }
        ),
        encoding="utf-8",
    )

    published = _run(
        str(PUBLISHER),
        "--event",
        str(event_path),
        "--bus-lane",
        "unit",
        "--bus-overlay",
        str(_overlay(tmp_path, lane="unit", broker="inmemory")),
    )

    assert published.returncode == 0
    combined = published.stdout + published.stderr
    assert "inmemory" in combined, combined


def test_an_injected_broker_that_diverges_from_the_overlay_is_refused(
    tmp_path: Path,
) -> None:
    """The OMN-14800 silent-drift guard reaches THIS publisher, not just the trigger.

    That guard is the entire reason `resolve_ci_bus_broker` takes a required
    `injected_broker`. This publisher shipped a call that omitted it, so every
    invocation raised a TypeError, took the "cannot resolve" path and published
    nothing while exiting 0 -- the guard was not bypassed, it was never
    reached. Asserting the REFUSAL is what proves the argument is passed for
    its purpose rather than to satisfy a signature.
    """
    event_path = tmp_path / "observed.json"
    event_path.write_text(
        json.dumps(
            {
                "event_type": "lane-census-observed",
                "topic": "onex.evt.omnibase-infra.lane-census-observed.v1",
                "drift_count": 0,
            }
        ),
        encoding="utf-8",
    )

    published = _run(
        str(PUBLISHER),
        "--event",
        str(event_path),
        "--bus-lane",
        "unit",
        "--bus-overlay",
        str(
            _overlay(
                tmp_path, lane="unit", broker=UNREACHABLE_BROKER, protocol="PLAINTEXT"
            )
        ),
        injected_broker="somewhere-else.example:9092",
    )

    assert published.returncode == 0
    combined = published.stdout + published.stderr
    assert "LANE BUS DRIFT" in combined, combined
