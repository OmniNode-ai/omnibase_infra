# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""generate_claude_lane_block.py — Generate the GENERATED_LANE_TABLE block for
CLAUDE.md from the lane manifest + a census snapshot (OMN-13034).

THE CLASS FIX (retro B-6): The CLAUDE.md lane table was hand-maintained, which
allowed phantom lanes (documented as running when absent) and undocumented
running lanes (judge was live but absent from the map). This script makes the
table GENERATED — it derives from:

  1. deploy/lane-census/lane-manifest.yaml  — the desired-state authority
  2. deploy/lane-census/census-snapshot.json — the last machine-emitted census

The generated block is delimited by HTML comments so it can be diffed in CI:

  <!-- GENERATED_LANE_TABLE BEGIN
       generated: <ISO-8601>
       source: lane-manifest + census-snapshot
       verified: <census emitted_at> via lane-census-check.sh on 192.168.86.201
  -->
  | Lane | Compose project | ... |
  ...
  <!-- GENERATED_LANE_TABLE END -->

CI (lane-census-staleness.yml) fails if:
  - The snapshot is older than MAX_AGE_DAYS
  - The committed snapshot disagrees with the manifest (check_lane_census_drift.py)
  - A manifest lane has no explicit entry in this module's port or boundary map

CORRECTED 2026-09-21 (OMN-18949). The third line here until today read "the
table in CLAUDE.md does not match what this script would generate". No such
check existed: the workflow invoked this script with neither --check nor
--update-claude-md, as a smoke test that it does not crash, and never compared
its output to anything. The block lives in omni_home/CLAUDE.md, a DIFFERENT
repository that this repo's CI does not check out, which is why the comparison
was never built and why the docstring asserting it stood for months.

--check PATH now performs that comparison for a caller who HAS both trees. What
CI enforces in-repo is the half that needs no second checkout: every manifest
lane must carry an explicit entry in _LANE_PORT_MAP and _LANE_BOUNDARY below.
Those maps are the generator's own static state, separate from the manifest, so
a lane added to the manifest and forgotten here renders an em-dash — visually
identical to a lane that genuinely has no ports. Silence and a declared absence
must not look the same, which is why the sentinels below are explicit values
rather than a missing key.

Usage:
  # Print the generated block to stdout:
  python scripts/generate_claude_lane_block.py

  # Write directly into omni_home/CLAUDE.md (replaces the GENERATED_LANE_TABLE
  # delimited block in-place):
  python scripts/generate_claude_lane_block.py --update-claude-md /path/to/CLAUDE.md
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_MANIFEST = _REPO / "deploy" / "lane-census" / "lane-manifest.yaml"
_DEFAULT_SNAPSHOT = _REPO / "deploy" / "lane-census" / "census-snapshot.json"

# Lane port map — static configuration that doesn't change per census but IS
# part of the lane definition. Kept here rather than in the manifest because
# port assignments are infra topology constants, not census state.
# Two explicit sentinels, so a MISSING key is always a defect and never a
# reading. Rendered output is identical for judge today; the difference is that
# the gate can now tell "this lane declares no runtime ports" apart from "nobody
# has written this lane's ports down", which the em-dash alone cannot.
_PORTS_NONE: dict[str, str] = {"main": "—", "effects": "—"}
_PORTS_UNDECLARED: dict[str, str] = {"main": "not declared", "effects": "not declared"}

_LANE_PORT_MAP: dict[str, dict[str, str]] = {
    "dev": {"main": "8085", "effects": "8086"},
    "stability-test": {"main": "18085", "effects": "18086"},
    # OMN-18890: the `prod` row is GONE from this map, and its removal is part of
    # this change rather than a tidy-up. The lab compose lane named `prod` was
    # shut down on 2026-09-13 (OMN-18320) and removed from lane-manifest.yaml, so
    # the row rendered nothing and was already dead. It cannot merely be left,
    # because the pre-PR pool below now claims 28085/28086 — the very ports that
    # row named — and a map in which two lanes claim one port is a map that will
    # be read wrong exactly once. What those numbers used to mean is recorded in
    # the lane manifest's pool comment, which is where a reader who meets them
    # will be. This is the lab compose lane only; production is the AWS
    # `onex-prod` namespace and has no row here in the first place.
    "judge": _PORTS_NONE,
    # OMN-17143 — collaborator lane for Lakshman Patel. This block was verified free
    # on .201 by read-only `ss -ltn` on 2026-08-30 and is BOUND by that lane as of
    # 2026-09-02 (OMN-17532 — the lane is built and running). Note the near-miss: the
    # adjacent 48085/48086 pair is the JUDGE lane's runtime main/effects
    # (docker/runtime-policy.env), and 49092 is the prod broker's external Kafka
    # port — the reserved block deliberately avoids both.
    "lakshman": {"main": "58085", "effects": "58086"},
    # OMN-18890 — the ephemeral pre-PR verify pool. Enumerated live from the lab
    # host's listening sockets on 2026-09-20 and free at that moment, every port
    # in both blocks. Slot 1's pair is the RETIRED lab `prod` lane's old block,
    # which is precisely why it is free (OMN-18320 shut that lane down on
    # 2026-09-13); the stale `prod` row above is the reason this collision is
    # visible here at all, and it is left standing because it still documents
    # what those numbers used to mean. The two columns this table has cannot
    # carry a slot's other two ports — gateway API 28090/38090 and projection
    # API 23002/33002 — which are declared in the lane manifest's reserved block.
    "prepr-1": {"main": "28085", "effects": "28086"},
    "prepr-2": {"main": "38085", "effects": "38086"},
    # OMN-18949 — both of these were ABSENT from this map and rendered an
    # em-dash for that reason alone, which read exactly like a declared "no
    # ports". They are declared now, and they are declared DIFFERENTLY,
    # because they are different states.
    #
    # ci-bus is a broker and nothing else: its own boundary line below says it
    # is not a runtime lane, so it has no runtime main/effects pair to name.
    # That is a real absence.
    "ci-bus": _PORTS_NONE,
    # dogfood declares a runtime service in the manifest, so it plausibly has a
    # pair — but no port block has ever been recorded for it and this lane
    # cannot invent one by reading a compose file it has not verified on the
    # host. UNDECLARED is the honest value and it renders as such, so a reader
    # is told the number is missing rather than told there is none.
    "dogfood": _PORTS_UNDECLARED,
}

_LANE_BOUNDARY: dict[str, str] = {
    "dev": "fully mutable test platform",
    "stability-test": "preferred proof lane for synthetic integration evidence",
    "judge": "NOT authorized for mutation — read-only",
    "lakshman": (
        "collaborator lane — owned by Lakshman; mutable by him; NOT a proof "
        "lane; never sourced for stability/prod grants"
    ),
    # OMN-18691. NOT a runtime lane: no runtime main, no effects, no Postgres.
    # It is in the census because a running-but-undeclared container is the
    # dangerous direction (retro B-6), and the boundary line has to say what the
    # blank main/effects columns otherwise leave a reader to guess.
    "ci-bus": (
        "fleet CI bus — a broker only; NOT a runtime lane, never a proof lane, "
        "and never sourced for stability/prod grants"
    ),
    # OMN-18890. The boundary is the whole point of the entry: a slot runs one
    # branch for the minutes its verification takes and is then destroyed, so it
    # is a premise for nothing and no other lane may read it.
    "prepr-1": (
        "ephemeral pre-PR verify slot — one branch, destroyed at end of run; "
        "never a proof lane and never sourced for stability/prod grants"
    ),
    "prepr-2": (
        "ephemeral pre-PR verify slot — one branch, destroyed at end of run; "
        "never a proof lane and never sourced for stability/prod grants"
    ),
    # OMN-18949 — absent from this map, so the lane rendered an em-dash in the
    # one column whose whole job is to say what a lane may be used for. The
    # text is the manifest's own dogfood comment, not a new assertion: this
    # lane reads the boundary out of the declaration rather than deciding it.
    "dogfood": (
        "prospective dogfood lane (OMN-18693) — optional, not deployed; no "
        "production or runtime evidence is asserted for it, and it is never "
        "a proof lane"
    ),
}


def undeclared_lanes(
    manifest: dict[str, Any],
    *,
    port_map: dict[str, dict[str, str]] | None = None,
    boundary: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    """Manifest lanes with no explicit entry in this module's static maps.

    The maps are the generator's OWN state, kept deliberately out of the
    manifest because ports and boundaries are topology constants rather than
    census state. The cost of that split is this failure: a lane added to the
    manifest and forgotten here still renders a row, with an em-dash in the
    port and boundary columns — indistinguishable from a lane that genuinely
    has neither. The generated table then reads as complete while saying
    nothing about a lane that exists.

    Returns (lane, which_map) pairs. Empty is the healthy state, and the caller
    owes it a positive control.
    """
    ports = _LANE_PORT_MAP if port_map is None else port_map
    bounds = _LANE_BOUNDARY if boundary is None else boundary
    missing: list[tuple[str, str]] = []
    for lane in manifest.get("lanes", {}):
        if lane not in ports:
            missing.append((lane, "_LANE_PORT_MAP"))
        if lane not in bounds:
            missing.append((lane, "_LANE_BOUNDARY"))
    return missing


def orphan_map_entries(
    manifest: dict[str, Any],
    *,
    port_map: dict[str, dict[str, str]] | None = None,
    boundary: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    """Static-map keys naming a lane the manifest no longer declares.

    The other direction, and the one that produced the stale `prod` row this
    map carried after OMN-18320 retired that lane: an entry nothing renders is
    invisible until its port numbers are reused by something else.
    """
    # The maps are injectable so the OMN-18949 incident replay can drive THIS
    # function -- the real guard -- with the maps captured out of the object
    # store at the commit that failed, rather than a reconstruction of them.
    # Default None means the live module state, which is what CI uses.
    ports = _LANE_PORT_MAP if port_map is None else port_map
    bounds = _LANE_BOUNDARY if boundary is None else boundary
    lanes = set(manifest.get("lanes", {}))
    orphans: list[tuple[str, str]] = []
    for lane in ports:
        if lane not in lanes:
            orphans.append((lane, "_LANE_PORT_MAP"))
    for lane in bounds:
        if lane not in lanes:
            orphans.append((lane, "_LANE_BOUNDARY"))
    return orphans


def extract_block(text: str) -> str | None:
    """Return the GENERATED_LANE_TABLE block from a document, or None."""
    pattern = re.compile(
        re.escape(_BEGIN_MARKER) + r".*?" + re.escape(_END_MARKER),
        re.DOTALL,
    )
    match = pattern.search(text)
    return match.group(0) if match else None


def comparable(block: str) -> list[str]:
    """A block reduced to the lines a drift comparison may depend on.

    The `generated:` line carries the wall-clock time of whichever run wrote
    it, so a byte comparison between a freshly rendered block and a committed
    one differs on EVERY run. Dropping that one line is what makes --check a
    content comparison rather than a clock comparison; every other line,
    including the `verified:` census timestamp, is content and is compared.
    """
    return [
        line.rstrip()
        for line in block.splitlines()
        if not line.strip().startswith("generated:")
    ]


_BEGIN_MARKER = "<!-- GENERATED_LANE_TABLE BEGIN"
_END_MARKER = "<!-- GENERATED_LANE_TABLE END -->"


def _load_manifest(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)  # type: ignore[no-any-return]


def _load_snapshot(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)  # type: ignore[no-any-return]
    except (json.JSONDecodeError, OSError):
        return None


def _service_count(
    lane_name: str, lane_spec: dict[str, Any], snapshot: dict[str, Any] | None
) -> str:
    """Return a human-readable container count for the table."""
    # OMN-17143: a lane declared with an EMPTY service list is a RESERVATION —
    # the port block, compose project and network name are claimed in the
    # desired-state authority before anything is stood up. Falling through to
    # the branches below would render it as "0 desired (not in last census)",
    # which reads like a broken lane rather than an unbuilt one. Say what is
    # actually true: nothing is declared, so nothing is expected to run.
    if not lane_spec.get("services"):
        return "0 — reserved, not built"

    if snapshot is None:
        # No live census — report desired from manifest
        required = sum(
            1
            for s in lane_spec.get("services", [])
            if s.get("kind", "service") == "service"
        )
        return f"{required} desired (no live census)"

    # Extract lane-specific container counts from the snapshot findings.
    # A snapshot with no findings for this lane means census matched desired.
    lanes_checked = snapshot.get("lanes_checked", [])
    if lane_name not in lanes_checked:
        required = sum(
            1
            for s in lane_spec.get("services", [])
            if s.get("kind", "service") == "service"
        )
        return f"{required} desired (not in last census)"

    # OMN-18890: zero findings is TWO different facts. A lane the census
    # reconciled and found clean is running; an optional lane it skipped because
    # nothing of it is up is absent. Both arrive here as an empty findings list,
    # and reading the second as the first is the phantom-lane claim retro B-6
    # made unwritable — a table row asserting N containers running on a lane with
    # none. The planner now says which lanes it skipped, so this reads the fact
    # rather than inferring it. A snapshot predating that key falls through to
    # the old behaviour, which is why the key is `.get`-ed rather than required:
    # an old snapshot is stale, not malformed.
    if lane_name in snapshot.get("lanes_skipped_optional_down", []):
        declared = sum(
            1
            for s in lane_spec.get("services", [])
            if s.get("kind", "service") == "service"
        )
        return f"0 running — optional lane down ({declared} declared)"

    findings = [f for f in snapshot.get("findings", []) if f.get("lane") == lane_name]
    drift_count = len(findings)
    required = sum(
        1
        for s in lane_spec.get("services", [])
        if s.get("kind", "service") == "service"
    )

    if drift_count == 0:
        return f"{required} running (census clean)"
    else:
        return f"{required} declared / {drift_count} drift item(s) — see census"


def generate_block(
    manifest: dict[str, Any],
    snapshot: dict[str, Any] | None,
    *,
    now: datetime | None = None,
) -> str:
    """Generate the full GENERATED_LANE_TABLE block as a string."""
    now = now or datetime.now(UTC)
    generated_ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    if snapshot is not None:
        census_emitted = snapshot.get("emitted_at", "unknown")
        verified_line = (
            f"verified: {census_emitted} via lane-census-check.sh on 192.168.86.201"
        )
    else:
        verified_line = "verified: MISSING — no census snapshot committed"

    lanes = manifest.get("lanes", {})

    header_comment = (
        f"{_BEGIN_MARKER}\n"
        f"     generated: {generated_ts}\n"
        f"     source: deploy/lane-census/lane-manifest.yaml + deploy/lane-census/census-snapshot.json\n"
        f"     {verified_line}\n"
        f"-->"
    )

    table_header = (
        "| Lane | Compose project | Main port | Effects port | Containers | Boundary |"
    )
    table_sep = (
        "|------|-----------------|-----------|--------------|------------|----------|"
    )
    rows: list[str] = []

    for lane_name, lane_spec in lanes.items():
        compose_project = lane_spec.get(
            "compose_project", f"omnibase-infra-{lane_name}"
        )
        ports = _LANE_PORT_MAP.get(lane_name, {"main": "—", "effects": "—"})
        main_port = ports["main"]
        effects_port = ports["effects"]
        containers = _service_count(lane_name, lane_spec, snapshot)
        boundary = _LANE_BOUNDARY.get(lane_name, "—")
        optional_tag = " (optional)" if lane_spec.get("optional") else ""
        rows.append(
            f"| {lane_name}{optional_tag} | `{compose_project}` | `{main_port}` | "
            f"`{effects_port}` | {containers} | {boundary} |"
        )

    table = "\n".join([table_header, table_sep] + rows)

    return "\n".join([header_comment, table, _END_MARKER])


def update_claude_md(claude_md_path: Path, new_block: str) -> bool:
    """Replace the GENERATED_LANE_TABLE block in CLAUDE.md in-place.

    Returns True if the file was updated, False if no block was found.
    """
    content = claude_md_path.read_text(encoding="utf-8")

    # Match everything from BEGIN to END inclusive (multiline).
    pattern = re.compile(
        re.escape(_BEGIN_MARKER) + r".*?" + re.escape(_END_MARKER),
        re.DOTALL,
    )
    if not pattern.search(content):
        print(
            f"WARNING: no GENERATED_LANE_TABLE block found in {claude_md_path}. "
            f"Insert the block manually first.",
            file=sys.stderr,
        )
        return False

    updated = pattern.sub(new_block, content)
    claude_md_path.write_text(updated, encoding="utf-8")
    print(f"Updated GENERATED_LANE_TABLE in {claude_md_path}", file=sys.stdout)
    return True


def _run_check(target: Path, expected_block: str) -> int:
    """--check: exit 0 when the target's block matches, 1 on any other outcome."""
    try:
        content = target.read_text(encoding="utf-8")
    except OSError as exc:
        print(
            f"FAIL: cannot read {target}: {exc}\n"
            "  An unreadable target is a failed check, never a passed one.",
            file=sys.stderr,
        )
        return 1

    found = extract_block(content)
    if found is None:
        print(
            f"FAIL: no GENERATED_LANE_TABLE block in {target}.\n"
            "  There is nothing to compare, which is a failure and not a pass.",
            file=sys.stderr,
        )
        return 1

    want = comparable(expected_block)
    have = comparable(found)
    if want == have:
        print(f"OK: the lane block in {target} matches the manifest and snapshot")
        return 0

    print(
        f"FAIL: the lane block in {target} is STALE against the manifest.",
        file=sys.stderr,
    )
    print(
        "  Regenerate it with --update-claude-md; do not hand-edit the block.",
        file=sys.stderr,
    )
    for line in difflib.unified_diff(
        have, want, fromfile=f"{target} (committed)", tofile="generated", lineterm=""
    ):
        print(f"  {line}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the CLAUDE.md GENERATED_LANE_TABLE block (OMN-13034)"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_DEFAULT_MANIFEST,
        help="Path to lane-manifest.yaml (default: deploy/lane-census/lane-manifest.yaml)",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=_DEFAULT_SNAPSHOT,
        help="Path to census-snapshot.json (default: deploy/lane-census/census-snapshot.json)",
    )
    parser.add_argument(
        "--update-claude-md",
        type=Path,
        metavar="CLAUDE_MD_PATH",
        help="Update the GENERATED_LANE_TABLE block in the given CLAUDE.md in-place",
    )
    parser.add_argument(
        "--check",
        type=Path,
        metavar="CLAUDE_MD_PATH",
        help=(
            "OMN-18949. Compare the block in the given CLAUDE.md against what "
            "this manifest and snapshot would generate and exit 1 on any "
            "difference, writing nothing. The `generated:` provenance line is "
            "excluded from the comparison because it carries the writing run's "
            "wall clock; every other line is content. An absent block, an "
            "unreadable file and a mismatch each exit 1 -- this never passes "
            "by finding nothing to compare"
        ),
    )
    args = parser.parse_args(argv)

    manifest = _load_manifest(args.manifest)
    snapshot = _load_snapshot(args.snapshot)

    if snapshot is None:
        print(
            f"WARNING: census snapshot not found at {args.snapshot}. "
            f"Generating table from manifest desired-state only (no live data).",
            file=sys.stderr,
        )

    block = generate_block(manifest, snapshot)

    if args.check:
        return _run_check(args.check, block)

    if args.update_claude_md:
        success = update_claude_md(args.update_claude_md, block)
        return 0 if success else 1
    else:
        print(block)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
