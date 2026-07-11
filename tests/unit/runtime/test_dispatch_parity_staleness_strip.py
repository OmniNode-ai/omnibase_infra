# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-14361 — dispatch-parity-gate staleness strip is version-noise-proof.

The dispatch-parity-gate's "Verify committed fixture is not stale" step
regenerates the live snapshot and byte-compares it against the committed
baseline. It strips a set of VOLATILE header fields first so environment/clock
noise does not gate the parity oracle.

Before OMN-14361 it stripped only ``generated_at_utc`` and compared
``installed_package_versions`` verbatim. That re-broke dev's frozen baseline on
every tracked-package release (e.g. omnibase_core 0.46.5 -> 0.46.7) even when the
dispatch corpus was byte-identical.

These tests pin the fix and, critically, prove it did NOT neuter the gate: a
genuine dispatch-corpus drift (a changed route / dispatcher / probe tuple, or a
changed corpus count) still trips the same comparison.

Related:
    - OMN-14361: dispatch-parity-gate re-breaks on package-version drift
    - tests.fixtures.dispatch_parity.harness.strip_volatile_header_fields
    - .github/workflows/dispatch-parity-gate.yml (staleness comparison step)
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from tests.fixtures.dispatch_parity import harness

pytestmark = pytest.mark.unit


def _minimal_snapshot() -> dict[str, Any]:
    """A snapshot shaped like ``build_snapshot()`` output, small enough to reason about.

    Only the structure the staleness comparison touches matters: a ``header`` with
    the volatile fields + a corpus block, plus the dispatch-corpus payload keys
    (``routes`` / ``probes``) that MUST still gate.
    """
    return {
        "header": {
            "fixture_version": "v2",
            "generated_at_utc": "2026-07-09T01:07:03.659145+00:00",
            "installed_package_versions": {
                "omnibase-core": "0.46.5",
                "omnibase-infra": "0.38.4",
                "onex-change-control": "0.5.1",
            },
            "corpus": {
                "contracts_discovered": 119,
                "registered_dispatchers": 83,
                "registered_routes": 99,
                "distinct_subscribe_topics": 88,
                "probe_count": 326,
            },
            "required_missing_packages": [],
        },
        "routes": [
            {"route_id": "r1", "dispatcher_id": "d1"},
            {"route_id": "r2", "dispatcher_id": "d2"},
        ],
        "probes": {
            "P1::topic.a": {
                "selection": {"status": "success", "dispatcher_ids": ["d1"]}
            },
        },
    }


def _diff(committed: dict[str, Any], live: dict[str, Any]) -> bool:
    """Return True when the two snapshots are considered STALE (drifted).

    Mirrors the workflow's staleness step exactly: strip volatile header fields,
    then compare canonical JSON.
    """
    c = harness.strip_volatile_header_fields(copy.deepcopy(committed))
    li = harness.strip_volatile_header_fields(copy.deepcopy(live))
    return json.dumps(c, sort_keys=True) != json.dumps(li, sort_keys=True)


def test_volatile_fields_cover_both_wallclock_and_versions() -> None:
    """Both wall-clock and installed-version metadata are declared volatile."""
    assert "generated_at_utc" in harness.VOLATILE_HEADER_FIELDS
    assert "installed_package_versions" in harness.VOLATILE_HEADER_FIELDS


def test_strip_removes_only_the_volatile_fields() -> None:
    """The helper pops the volatile fields and leaves the corpus untouched."""
    snap = _minimal_snapshot()
    stripped = harness.strip_volatile_header_fields(snap)
    assert stripped is snap  # in-place, returns the same object
    assert "generated_at_utc" not in stripped["header"]
    assert "installed_package_versions" not in stripped["header"]
    # Corpus + required_missing_packages survive — they still gate.
    assert stripped["header"]["corpus"]["registered_routes"] == 99
    assert "required_missing_packages" in stripped["header"]
    assert stripped["routes"]
    assert stripped["probes"]


def test_package_version_bump_alone_is_not_stale() -> None:
    """A new tracked-package release with an IDENTICAL corpus must NOT gate.

    This is the exact OMN-14361 recurrence: dev's frozen baseline records an older
    omnibase_core; CI regenerates with a newer one; nothing else changed.
    """
    committed = _minimal_snapshot()
    live = _minimal_snapshot()
    live["header"]["installed_package_versions"]["omnibase-core"] = "0.46.7"
    live["header"]["generated_at_utc"] = "2026-07-10T12:00:00.000000+00:00"
    assert not _diff(committed, live), (
        "Version/clock-only header delta must not be reported as stale after the "
        "OMN-14361 fix."
    )


def test_route_drift_still_gates() -> None:
    """A changed dispatch route STILL trips the gate (fix did not neuter it)."""
    committed = _minimal_snapshot()
    live = _minimal_snapshot()
    # Same package versions; a real selection change.
    live["routes"][0]["dispatcher_id"] = "d99"
    assert _diff(committed, live), (
        "A genuine route/dispatcher drift must still be reported as stale."
    )


def test_probe_tuple_drift_still_gates() -> None:
    """A changed probe selection tuple STILL trips the gate."""
    committed = _minimal_snapshot()
    live = _minimal_snapshot()
    live["probes"]["P1::topic.a"]["selection"]["status"] = "no_dispatcher"
    assert _diff(committed, live)


def test_corpus_count_drift_still_gates() -> None:
    """A shrunk/grown corpus count (e.g. a dropped contract) STILL trips the gate."""
    committed = _minimal_snapshot()
    live = _minimal_snapshot()
    live["header"]["corpus"]["registered_routes"] = 98
    assert _diff(committed, live)


def test_required_missing_package_drift_still_gates() -> None:
    """A required package going missing STILL trips the gate.

    ``installed_package_versions`` is stripped, but ``required_missing_packages`` is
    NOT — so a real corpus-shrinking dependency loss is still caught.
    """
    committed = _minimal_snapshot()
    live = _minimal_snapshot()
    live["header"]["required_missing_packages"] = ["omnibase-core"]
    assert _diff(committed, live)


def test_strip_is_defensive_on_missing_header() -> None:
    """The helper tolerates a snapshot without a dict header (no crash)."""
    assert harness.strip_volatile_header_fields({}) == {}
    weird: dict[str, Any] = {"header": None}
    assert harness.strip_volatile_header_fields(weird) == {"header": None}
