# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18949 — the committed census must agree with the manifest.

Nothing gated census drift. The 2026-09-20 silent-gates review measured the
committed snapshot carrying ``drift_count: 5`` with five findings on the
development lane, and no gate anywhere refusing on it. The one CI job touching
the census reads the snapshot's AGE and the manifest's agreement with the
compose files; neither reads the findings.

These cases pin the gate's contract, including the two it must NOT have: it
does not require zero drift, and it does not report a clean verdict over an
input it could not read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from check_lane_census_drift import (
    declared_containers,
    evaluate,
    main,
)
from generate_claude_lane_block import (
    _LANE_BOUNDARY,
    _LANE_PORT_MAP,
    comparable,
    extract_block,
    generate_block,
    orphan_map_entries,
    undeclared_lanes,
)

pytestmark = pytest.mark.unit


def _manifest(**lanes: Any) -> dict[str, Any]:
    return {"schema_version": "1.0.0", "lanes": lanes}


def _lane(*containers: str) -> dict[str, Any]:
    return {
        "compose_project": "omnibase-infra-x",
        "services": [{"name": c, "kind": "service", "replicas": 1} for c in containers],
    }


def _snapshot(
    lanes_checked: list[str], findings: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "emitted_at": "2026-09-18T06:55:01.971641+00:00",
        "lanes_checked": lanes_checked,
        "drift_count": len(findings),
        "findings": findings,
    }


def _kinds(findings: list[Any]) -> list[str]:
    return [f.kind for f in findings]


def _run(m: dict[str, Any], s: dict[str, Any]) -> list[Any]:
    return evaluate(m, s, undeclared=[], orphans=[])


# ---------------------------------------------------------------------------
# The drift the ticket names.
# ---------------------------------------------------------------------------


def test_a_container_adopted_into_the_manifest_and_left_in_the_snapshot_is_drift() -> (
    None
):
    """The load-bearing case.

    A container reported as running-but-undeclared, which the manifest now
    declares, is a finding that reads as live drift forever. Nothing caught it.
    """
    m = _manifest(dev=_lane("omninode-runtime", "onex-api"))
    s = _snapshot(
        ["dev"],
        [{"lane": "dev", "kind": "unexpected_container", "container": "onex-api"}],
    )
    assert _kinds(_run(m, s)) == ["stale_unexpected_container"]


def test_a_genuinely_undeclared_container_is_NOT_drift() -> None:
    """The gate's most important negative.

    Live drift on a mutable lane is a normal reportable state. A gate that
    refused every pull request over it would be switched off within a day, and
    this is the assertion that keeps it from becoming one.
    """
    m = _manifest(dev=_lane("omninode-runtime"))
    s = _snapshot(
        ["dev"],
        [{"lane": "dev", "kind": "unexpected_container", "container": "onex-api"}],
    )
    assert _run(m, s) == []


def test_a_container_absent_finding_outliving_its_declaration_is_drift() -> None:
    m = _manifest(dev=_lane("omninode-runtime"))
    s = _snapshot(
        ["dev"],
        [{"lane": "dev", "kind": "container_absent", "container": "retired-worker"}],
    )
    assert _kinds(_run(m, s)) == ["stale_container_absent"]


def test_a_container_absent_finding_for_a_still_declared_container_is_NOT_drift() -> (
    None
):
    m = _manifest(dev=_lane("omninode-runtime"))
    s = _snapshot(
        ["dev"],
        [{"lane": "dev", "kind": "container_absent", "container": "omninode-runtime"}],
    )
    assert _run(m, s) == []


# ---------------------------------------------------------------------------
# Lanes and self-consistency.
# ---------------------------------------------------------------------------


def test_a_snapshot_naming_a_lane_the_manifest_does_not_declare_is_drift() -> None:
    m = _manifest(dev=_lane("omninode-runtime"))
    s = _snapshot(["dev", "retired-lane"], [])
    assert _kinds(_run(m, s)) == ["snapshot_lane_unknown"]


def test_a_finding_against_an_undeclared_lane_is_reported_once_not_twice() -> None:
    """An unknown lane is one defect; classifying its containers would be noise."""
    m = _manifest(dev=_lane("omninode-runtime"))
    s = _snapshot(
        ["dev"],
        [{"lane": "ghost", "kind": "unexpected_container", "container": "x"}],
    )
    assert _kinds(_run(m, s)) == ["snapshot_lane_unknown"]


def test_a_snapshot_that_miscounts_its_own_findings_is_drift() -> None:
    m = _manifest(dev=_lane("omninode-runtime"))
    s = _snapshot(["dev"], [])
    s["drift_count"] = 3
    assert _kinds(_run(m, s)) == ["finding_count_mismatch"]


# ---------------------------------------------------------------------------
# The generator's static maps, which are the second half of the staleness.
# ---------------------------------------------------------------------------


def test_a_manifest_lane_missing_from_a_generator_map_is_drift() -> None:
    """An em-dash from a forgotten entry looks exactly like a declared absence."""
    m = _manifest(newlane=_lane("x"))
    findings = evaluate(
        m,
        _snapshot(["newlane"], []),
        undeclared=undeclared_lanes(m),
        orphans=[],
    )
    assert set(_kinds(findings)) == {"generator_lane_undeclared"}
    assert {f.subject for f in findings} == {"_LANE_PORT_MAP", "_LANE_BOUNDARY"}


def test_a_generator_map_entry_for_a_retired_lane_is_drift() -> None:
    """How the retired lab prod row survived until its ports were reused."""
    m = _manifest(dev=_lane("x"))
    orphans = orphan_map_entries(m)
    assert orphans, "every real map key except dev is an orphan against this manifest"
    findings = evaluate(m, _snapshot(["dev"], []), undeclared=[], orphans=orphans)
    assert set(_kinds(findings)) == {"generator_map_orphan"}


def test_every_real_manifest_lane_is_declared_in_both_generator_maps() -> None:
    """The live assertion, over the committed manifest rather than a fixture.

    Both of these were absent before this ticket: the dogfood lane from both
    maps, the fleet CI bus from the port map. Each rendered an em-dash for that
    reason alone.
    """
    manifest = yaml.safe_load(
        (
            Path(__file__).resolve().parents[3]
            / "deploy/lane-census/lane-manifest.yaml"
        ).read_text()
    )
    assert undeclared_lanes(manifest) == []
    assert orphan_map_entries(manifest) == []
    # Positive control for the zeros above: the maps are non-empty and the
    # manifest declares lanes, so an empty result is a real agreement rather
    # than two empty inputs agreeing vacuously.
    assert len(manifest["lanes"]) >= 8
    assert len(_LANE_PORT_MAP) >= 8 and len(_LANE_BOUNDARY) >= 8


def test_an_undeclared_port_block_renders_as_undeclared_not_as_an_em_dash() -> None:
    """Silence and a declared absence must not render the same.

    The fleet CI bus declares no runtime ports and says so; the dogfood lane's
    ports have simply never been recorded. Both used to render an em-dash.
    """
    manifest = yaml.safe_load(
        (
            Path(__file__).resolve().parents[3]
            / "deploy/lane-census/lane-manifest.yaml"
        ).read_text()
    )
    block = generate_block(manifest, None)
    dogfood = next(line for line in block.splitlines() if line.startswith("| dogfood"))
    ci_bus = next(line for line in block.splitlines() if line.startswith("| ci-bus"))
    assert "not declared" in dogfood
    assert "not declared" not in ci_bus and "`—`" in ci_bus


# ---------------------------------------------------------------------------
# Rule 16: an unreadable or empty input is an error, never a clean census.
# ---------------------------------------------------------------------------


def test_an_unreadable_snapshot_exits_two_and_never_reports_agreement(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "m.yaml"
    manifest.write_text(yaml.safe_dump(_manifest(dev=_lane("x"))))
    snapshot = tmp_path / "s.json"
    snapshot.write_text("{ not json")
    assert main(["--manifest", str(manifest), "--snapshot", str(snapshot)]) == 2


def test_a_manifest_with_no_lanes_exits_two(tmp_path: Path) -> None:
    manifest = tmp_path / "m.yaml"
    manifest.write_text(yaml.safe_dump({"lanes": {}}))
    snapshot = tmp_path / "s.json"
    snapshot.write_text(json.dumps(_snapshot(["dev"], [])))
    assert main(["--manifest", str(manifest), "--snapshot", str(snapshot)]) == 2


def test_a_snapshot_that_observed_nothing_exits_two(tmp_path: Path) -> None:
    """An empty census cannot corroborate a manifest, and must not read as if it did."""
    manifest = tmp_path / "m.yaml"
    manifest.write_text(yaml.safe_dump(_manifest(dev=_lane("x"))))
    snapshot = tmp_path / "s.json"
    snapshot.write_text(json.dumps(_snapshot([], [])))
    assert main(["--manifest", str(manifest), "--snapshot", str(snapshot)]) == 2


def test_the_committed_tree_passes_the_gate() -> None:
    """The gate is armed against the real files, not only against fixtures."""
    assert main([]) == 0


def test_every_run_prints_the_input_sizes_it_derived_its_verdict_from(capsys) -> None:
    """A zero reported bare is indistinguishable from a sweep that read nothing."""
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "manifest lane(s)" in out and "declared container(s)" in out
    assert "snapshot finding(s)" in out


# ---------------------------------------------------------------------------
# --check: the comparison the generator's docstring claimed and did not have.
# ---------------------------------------------------------------------------


def test_the_check_comparison_ignores_the_generated_timestamp_only() -> None:
    """A byte comparison would differ on every run; a clock is not content."""
    manifest = _manifest(dev=_lane("x"))
    a = generate_block(manifest, None)
    import time

    time.sleep(1.1)
    b = generate_block(manifest, None)
    assert a != b, "the provenance timestamp is expected to differ between runs"
    assert comparable(a) == comparable(b)
    # and the census timestamp IS content: a block verified against a different
    # census must not compare equal.
    c = generate_block(
        manifest, {"emitted_at": "2026-01-01T00:00:00+00:00", "findings": []}
    )
    assert comparable(c) != comparable(a)


def test_check_fails_on_a_target_with_no_block(tmp_path: Path) -> None:
    """Nothing to compare is a failure, not a pass."""
    target = tmp_path / "CLAUDE.md"
    target.write_text("# no generated block here\n")
    rc = main_generator(["--check", str(target)])
    assert rc == 1


def test_check_fails_on_an_unreadable_target(tmp_path: Path) -> None:
    rc = main_generator(["--check", str(tmp_path / "absent.md")])
    assert rc == 1


def test_check_passes_on_a_target_holding_the_current_block(tmp_path: Path) -> None:
    manifest_path = (
        Path(__file__).resolve().parents[3] / "deploy/lane-census/lane-manifest.yaml"
    )
    snapshot_path = (
        Path(__file__).resolve().parents[3] / "deploy/lane-census/census-snapshot.json"
    )
    manifest = yaml.safe_load(manifest_path.read_text())
    snapshot = json.loads(snapshot_path.read_text())
    target = tmp_path / "CLAUDE.md"
    target.write_text("intro\n\n" + generate_block(manifest, snapshot) + "\n\noutro\n")
    assert extract_block(target.read_text()) is not None
    rc = main_generator(
        [
            "--check",
            str(target),
            "--manifest",
            str(manifest_path),
            "--snapshot",
            str(snapshot_path),
        ]
    )
    assert rc == 0


def test_check_fails_when_the_target_block_names_a_stale_lane(tmp_path: Path) -> None:
    """The positive control for the two passes above."""
    manifest_path = (
        Path(__file__).resolve().parents[3] / "deploy/lane-census/lane-manifest.yaml"
    )
    snapshot_path = (
        Path(__file__).resolve().parents[3] / "deploy/lane-census/census-snapshot.json"
    )
    manifest = yaml.safe_load(manifest_path.read_text())
    snapshot = json.loads(snapshot_path.read_text())
    block = generate_block(manifest, snapshot)
    stale = block.replace("| judge |", "| judge-renamed |")
    target = tmp_path / "CLAUDE.md"
    target.write_text(stale)
    rc = main_generator(
        [
            "--check",
            str(target),
            "--manifest",
            str(manifest_path),
            "--snapshot",
            str(snapshot_path),
        ]
    )
    assert rc == 1


def main_generator(argv: list[str]) -> int:
    from generate_claude_lane_block import main as gen_main

    return gen_main(argv)


def test_declared_containers_reads_every_lane() -> None:
    m = _manifest(a=_lane("x", "y"), b=_lane("z"))
    assert declared_containers(m) == {"a": {"x", "y"}, "b": {"z"}}
