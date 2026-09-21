# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18949 incident replay — the retired lab ``prod`` lane's stale map rows.

THE INCIDENT, from the object store rather than from a report.

``2e4eec05cae24801f14036dba2fcfd14391dda11`` (2026-09-14T10:22:41Z, OMN-18320)
removed the lab compose ``prod`` lane from ``lane-manifest.yaml``. The lane was
gone; the table generator's own static maps were not updated, and both
``_LANE_PORT_MAP`` and ``_LANE_BOUNDARY`` kept a ``prod`` entry. Nothing
rendered from those rows, because the generator iterates the MANIFEST's lanes,
so the stale entries were invisible to every surface.

They stayed for six days. ``ba9a779c6bc276bbd2a00ce6e18eb6261c5b2046``
(2026-09-20T14:51:49Z, OMN-18890) finally removed the port row, and only because
the new pre-PR verify pool claimed ``28085``/``28086`` — the very numbers the
dead ``prod`` row still named. That commit's own comment states the hazard in
its own words: a map in which two lanes claim one port is a map that will be
read wrong exactly once.

This is the ``generator_map_orphan`` class, and no gate existed for it. These
cases drive the REAL guard functions with the REAL bytes from that window.

THE DISCRIMINATOR IS LOAD-BEARING. A guard that reported an orphan for every
input would satisfy the first case trivially and red every future change, so
the second case requires the same functions to return NOTHING against the
current committed tree, and the third requires the finding to name ``prod`` and
nothing else — a guard that over-reported would be its own defect.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from check_lane_census_drift import evaluate
from generate_claude_lane_block import (
    orphan_map_entries,
    undeclared_lanes,
)

_FIXTURES = _REPO / "tests" / "fixtures" / "omn18949"
_MANIFEST = _FIXTURES / "lane-manifest-2e4eec05c.yaml.captured"
_GENERATOR = _FIXTURES / "generate_claude_lane_block-2e4eec05c.py.captured"

# Recorded so a reformatted or retyped fixture stops being the artifact that
# failed. Both are re-fetchable at the locators in the registry entry.
_MANIFEST_SHA = "132f8c4de187ef7ccc2d7d50e91ec548b3d132aac81f21755afd83efd5be7cf6"
_GENERATOR_SHA = "ec3f57ac4ca9e93d2cd178219d5d1d36ce06b787cf23c1d712be42da1f672bfd"

pytestmark = pytest.mark.unit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _captured_maps() -> tuple[dict[str, Any], dict[str, Any]]:
    """The two static maps as they stood at the failing commit.

    Read with ``ast`` rather than executed. The captured file is our own history
    and would run, but a replay that imports the artifact gains a dependency on
    whatever that revision imported at module scope; parsing the two literals is
    both side-effect free and exactly what is being replayed.
    """
    tree = ast.parse(_GENERATOR.read_text(encoding="utf-8"))
    found: dict[str, Any] = {}
    for node in tree.body:
        targets = getattr(node, "targets", None) or (
            [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        for target in targets:
            if isinstance(target, ast.Name) and target.id in {
                "_LANE_PORT_MAP",
                "_LANE_BOUNDARY",
            }:
                found[target.id] = ast.literal_eval(node.value)  # type: ignore[arg-type]
    assert set(found) == {"_LANE_PORT_MAP", "_LANE_BOUNDARY"}, (
        "the captured generator must still declare both static maps; if it does "
        "not, the fixture is not the artifact this replay is about"
    )
    return found["_LANE_PORT_MAP"], found["_LANE_BOUNDARY"]


def test_the_fixtures_are_the_captured_bytes_and_not_a_reconstruction() -> None:
    """A hand-typed fixture has no locator that resolves; this is the check."""
    assert _sha256(_MANIFEST) == _MANIFEST_SHA
    assert _sha256(_GENERATOR) == _GENERATOR_SHA


def test_the_captured_window_is_the_one_the_incident_describes() -> None:
    """The premise, asserted rather than narrated.

    The manifest at that commit must NOT declare a prod lane, and the generator
    at the SAME commit must still carry prod rows in both maps. If either half
    were untrue the replay would be about something else.
    """
    manifest = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))
    ports, boundary = _captured_maps()
    assert "prod" not in manifest["lanes"], "OMN-18320 removed the lane at this commit"
    assert "prod" in ports and "prod" in boundary, (
        "both static maps still carried the retired lane, which is the defect"
    )
    # The collision that eventually forced the cleanup, from the bytes.
    assert ports["prod"] == {"main": "28085", "effects": "28086"}


def test_the_guard_rejects_the_captured_artifact() -> None:
    """The replay. The real functions, the real bytes, the verdict nobody had."""
    manifest = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))
    ports, boundary = _captured_maps()

    orphans = orphan_map_entries(manifest, port_map=ports, boundary=boundary)
    assert ("prod", "_LANE_PORT_MAP") in orphans
    assert ("prod", "_LANE_BOUNDARY") in orphans

    findings = evaluate(
        manifest,
        {"lanes_checked": list(manifest["lanes"]), "drift_count": 0, "findings": []},
        undeclared=undeclared_lanes(manifest, port_map=ports, boundary=boundary),
        orphans=orphans,
    )
    kinds = {f.kind for f in findings}
    assert "generator_map_orphan" in kinds, (
        "the guard must reject the captured tree; this is the verdict that did "
        "not exist for the six days the stale rows stood"
    )
    # It names the retired lane, and the detail says why an invisible entry is
    # dangerous rather than merely untidy.
    orphan_findings = [f for f in findings if f.kind == "generator_map_orphan"]
    assert {f.lane for f in orphan_findings} == {"prod"}
    assert "reused by something else" in orphan_findings[0].detail


def test_the_guard_does_NOT_reject_the_current_tree() -> None:
    """The discriminator.

    A guard that refused every input would satisfy the case above trivially and
    red every future change. The same two functions, over the live committed
    manifest and the live maps, must return nothing.
    """
    manifest = yaml.safe_load(
        (_REPO / "deploy/lane-census/lane-manifest.yaml").read_text(encoding="utf-8")
    )
    assert orphan_map_entries(manifest) == []
    assert undeclared_lanes(manifest) == []
    # Positive control for those two zeros: the inputs are non-empty, so the
    # agreement is real rather than two empty collections agreeing vacuously.
    assert len(manifest["lanes"]) >= 8


def test_the_guard_reports_ONLY_the_retired_lane_on_the_captured_artifact() -> None:
    """A guard that over-reported would be its own defect.

    Every lane the captured manifest DOES declare was present in both captured
    maps, so the only orphan is the retired one. If this widened, the guard
    would be flagging live lanes and would be switched off.
    """
    manifest = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))
    ports, boundary = _captured_maps()
    assert undeclared_lanes(manifest, port_map=ports, boundary=boundary) == []
    assert {
        lane
        for lane, _ in orphan_map_entries(manifest, port_map=ports, boundary=boundary)
    } == {"prod"}
