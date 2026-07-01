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
  - The table in CLAUDE.md does not match what this script would generate

Usage:
  # Print the generated block to stdout:
  python scripts/generate_claude_lane_block.py

  # Write directly into omni_home/CLAUDE.md (replaces the GENERATED_LANE_TABLE
  # delimited block in-place):
  python scripts/generate_claude_lane_block.py --update-claude-md /path/to/CLAUDE.md
"""

from __future__ import annotations

import argparse
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
_LANE_PORT_MAP: dict[str, dict[str, str]] = {
    "dev": {"main": "8085", "effects": "8086"},
    "stability-test": {"main": "18085", "effects": "18086"},
    "prod": {"main": "28085", "effects": "28086"},
    "judge": {"main": "—", "effects": "—"},
}

_LANE_BOUNDARY: dict[str, str] = {
    "dev": "fully mutable test platform",
    "stability-test": "preferred proof lane for synthetic integration evidence",
    "prod": "read-only unless the user explicitly approves production mutation",
    "judge": "NOT authorized for mutation — read-only",
}

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

    if args.update_claude_md:
        success = update_claude_md(args.update_claude_md, block)
        return 0 if success else 1
    else:
        print(block)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
