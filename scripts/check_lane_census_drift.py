# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""check_lane_census_drift.py — the committed census must agree with the manifest (OMN-18949).

WHY THIS EXISTS
---------------
Nothing gated census drift. The 2026-09-20 silent-gates review measured the
state directly: the committed snapshot carried ``drift_count: 5``, severity
``warning``, five ``unexpected_container`` findings on the development lane —
and **no gate anywhere refused on a non-zero drift count**. The one CI job that
touches the census checks the snapshot's AGE and the manifest's agreement with
the compose files; neither reads the findings. The drift is rendered into the
doctrine file as the prose string ``5 drift item(s) — see census`` and
normalised away by the next refresh.

There is a second, quieter half. ``generate_claude_lane_block.py``'s own
docstring claimed CI fails when "the table in CLAUDE.md does not match what
this script would generate". It does not and never did: the workflow invokes
that script with neither a comparison flag nor an output path, purely to prove
it does not crash. The generated block lives in ``omni_home/CLAUDE.md``, a
different repository this repo's CI does not check out.

WHAT THIS GATE ASSERTS, AND WHAT IT DELIBERATELY DOES NOT
---------------------------------------------------------
It does **not** assert ``drift_count == 0``. Live drift on a mutable lane is a
normal, reportable state and a gate that refused every PR over it would be
turned off within a day. What it asserts is that the two COMMITTED artifacts
still describe the same world, and that the generator can render every lane
either of them names:

1. ``snapshot_lane_unknown`` — the snapshot checked, or filed a finding
   against, a lane the manifest does not declare.
2. ``finding_count_mismatch`` — ``drift_count`` disagrees with the number of
   findings in the same file. A snapshot that miscounts itself cannot be the
   authority for anything downstream.
3. ``stale_unexpected_container`` — a finding says a container is running and
   UNDECLARED, but the manifest now declares it. Somebody adopted the container
   into the manifest and left the snapshot behind. **This is the drift the
   ticket names**, and it is the one that reads as a live finding forever.
4. ``stale_container_absent`` — the mirror: a finding says a declared container
   is missing, and the manifest no longer declares it.
5. ``generator_lane_undeclared`` — a manifest lane with no explicit entry in the
   generator's port or boundary map. Such a lane still renders a row, with an
   em-dash in both columns, indistinguishable from a lane that genuinely has
   neither. The table then reads as complete while saying nothing about a lane
   that exists.
6. ``generator_map_orphan`` — a generator map entry naming a lane the manifest
   has retired. This is how the retired lab ``prod`` row survived OMN-18320
   until its port numbers were claimed by the pre-PR pool.

POSITIVE CONTROL (CLAUDE.md rule 16)
------------------------------------
A clean run is indistinguishable from a run whose inputs were empty, so this
script REFUSES an empty input rather than reporting zero findings over it: a
manifest with no lanes, a snapshot with no ``lanes_checked``, or an unparseable
either is an error, not a pass. Every run prints the input sizes it derived its
verdict from, so a zero is never reported bare.

Exit codes: ``0`` agreement, ``1`` one or more findings, ``2`` an input could
not be read or was empty.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_MANIFEST = _REPO / "deploy" / "lane-census" / "lane-manifest.yaml"
_DEFAULT_SNAPSHOT = _REPO / "deploy" / "lane-census" / "census-snapshot.json"


@dataclass(frozen=True)
class Finding:
    """One disagreement, named by a stable kind a reader can grep for."""

    kind: str
    lane: str
    subject: str
    detail: str


def load_manifest(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return loaded


def load_snapshot(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        loaded = json.load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not parse to an object")
    return loaded


def declared_containers(manifest: dict[str, Any]) -> dict[str, set[str]]:
    """lane -> the set of container names the manifest declares for it."""
    out: dict[str, set[str]] = {}
    for lane, spec in manifest.get("lanes", {}).items():
        services = spec.get("services") or []
        out[lane] = {s["name"] for s in services if isinstance(s, dict) and "name" in s}
    return out


def evaluate(
    manifest: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    undeclared: list[tuple[str, str]],
    orphans: list[tuple[str, str]],
) -> list[Finding]:
    """Every disagreement between the two committed artifacts, in a fixed order."""
    findings: list[Finding] = []
    per_lane = declared_containers(manifest)
    lanes = set(per_lane)

    # 1. a lane the snapshot names and the manifest does not declare
    for lane in snapshot.get("lanes_checked", []):
        if lane not in lanes:
            findings.append(
                Finding(
                    kind="snapshot_lane_unknown",
                    lane=lane,
                    subject="lanes_checked",
                    detail=(
                        f"the snapshot checked lane '{lane}', which the manifest does "
                        "not declare -- one of the two was edited without the other"
                    ),
                )
            )

    snapshot_findings = snapshot.get("findings", [])

    # 2. the snapshot's own count against its own findings
    declared_count = snapshot.get("drift_count")
    if declared_count is not None and declared_count != len(snapshot_findings):
        findings.append(
            Finding(
                kind="finding_count_mismatch",
                lane="(snapshot)",
                subject="drift_count",
                detail=(
                    f"drift_count is {declared_count} and the file carries "
                    f"{len(snapshot_findings)} finding(s); a snapshot that miscounts "
                    "itself cannot be the authority for anything downstream"
                ),
            )
        )

    for item in snapshot_findings:
        lane = item.get("lane", "")
        container = item.get("container", "")
        kind = item.get("kind", "")

        if lane and lane not in lanes:
            findings.append(
                Finding(
                    kind="snapshot_lane_unknown",
                    lane=lane,
                    subject=container or "(no container)",
                    detail=(
                        f"a finding is filed against lane '{lane}', which the manifest "
                        "does not declare"
                    ),
                )
            )
            continue

        # 3. the drift this ticket names: adopted into the manifest, still
        #    reported as undeclared by the snapshot.
        if kind == "unexpected_container" and container in per_lane.get(lane, set()):
            findings.append(
                Finding(
                    kind="stale_unexpected_container",
                    lane=lane,
                    subject=container,
                    detail=(
                        f"the snapshot reports '{container}' as running but UNDECLARED "
                        f"on lane '{lane}', and the manifest now declares it. The "
                        "manifest was updated and the census was not refreshed, so the "
                        "finding reads as live drift forever"
                    ),
                )
            )

        # 4. the mirror
        if kind == "container_absent" and container not in per_lane.get(lane, set()):
            findings.append(
                Finding(
                    kind="stale_container_absent",
                    lane=lane,
                    subject=container,
                    detail=(
                        f"the snapshot reports declared container '{container}' as "
                        f"absent from lane '{lane}', and the manifest no longer "
                        "declares it; the finding outlived the declaration"
                    ),
                )
            )

    # 5/6. the generator's static maps against the manifest
    for lane, which in undeclared:
        findings.append(
            Finding(
                kind="generator_lane_undeclared",
                lane=lane,
                subject=which,
                detail=(
                    f"lane '{lane}' has no explicit entry in {which}, so the generated "
                    "table renders an em-dash for it -- identical to a lane that "
                    "genuinely has none. Declare it explicitly, using the module's "
                    "sentinels where the value really is absent or unknown"
                ),
            )
        )
    for lane, which in orphans:
        findings.append(
            Finding(
                kind="generator_map_orphan",
                lane=lane,
                subject=which,
                detail=(
                    f"{which} carries an entry for lane '{lane}', which the manifest "
                    "does not declare; an entry nothing renders stays invisible until "
                    "its values are reused by something else"
                ),
            )
        )

    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail when the committed lane census disagrees with the manifest (OMN-18949)"
    )
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--snapshot", type=Path, default=_DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the findings as JSON on stdout instead of prose",
    )
    args = parser.parse_args(argv)

    # The inputs are loaded before anything is evaluated, and an unreadable or
    # empty one exits 2 rather than producing an empty finding list. An empty
    # sweep and a clean sweep are the same output otherwise.
    try:
        manifest = load_manifest(args.manifest)
        snapshot = load_snapshot(args.snapshot)
    except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
        print(f"ERROR: could not read the census inputs: {exc}", file=sys.stderr)
        print(
            "  An unreadable input is an error, never a clean census.",
            file=sys.stderr,
        )
        return 2

    lanes = manifest.get("lanes") or {}
    if not lanes:
        print(
            f"ERROR: {args.manifest} declares no lanes; there is nothing to check "
            "against and a zero here would be meaningless",
            file=sys.stderr,
        )
        return 2
    if not snapshot.get("lanes_checked"):
        print(
            f"ERROR: {args.snapshot} records no lanes_checked; the census did not "
            "observe anything and cannot corroborate the manifest",
            file=sys.stderr,
        )
        return 2

    # Imported here so the module is usable without the generator on the path.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate_claude_lane_block import (
        orphan_map_entries,
        undeclared_lanes,
    )

    findings = evaluate(
        manifest,
        snapshot,
        undeclared=undeclared_lanes(manifest),
        orphans=orphan_map_entries(manifest),
    )

    container_total = sum(len(v) for v in declared_containers(manifest).values())
    if args.json:
        json.dump(
            {
                "findings": [asdict(f) for f in findings],
                "manifest_lanes": len(lanes),
                "manifest_containers": container_total,
                "snapshot_findings": len(snapshot.get("findings", [])),
                "snapshot_emitted_at": snapshot.get("emitted_at"),
            },
            sys.stdout,
            indent=2,
        )
        print()
        return 1 if findings else 0

    # The input sizes are printed on EVERY run, pass or fail. A zero reported
    # without them is indistinguishable from a sweep that read nothing.
    print(
        f"lane census drift gate (OMN-18949): read {len(lanes)} manifest lane(s), "
        f"{container_total} declared container(s), "
        f"{len(snapshot.get('findings', []))} snapshot finding(s) emitted "
        f"{snapshot.get('emitted_at', 'at an unrecorded time')}"
    )

    if not findings:
        print("OK: the committed census and the manifest agree.")
        return 0

    print(
        f"FAIL: {len(findings)} disagreement(s) between the committed census and the manifest."
    )
    for f in findings:
        print(f"  [{f.kind}] lane={f.lane} subject={f.subject}")
        print(f"      {f.detail}")
    print(
        "\nThis gate does NOT require drift_count to be zero -- live drift on a mutable "
        "lane is a normal reportable state. It requires the two COMMITTED files to "
        "describe the same world. Refresh the snapshot, or correct the manifest."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
