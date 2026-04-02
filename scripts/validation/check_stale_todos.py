#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
#
# Stale TODO Check
#
# Scans all repos under omni_home for TODO(OMN-XXXX) references and extracts
# the ticket IDs. Optionally checks each ticket against Linear to flag TODOs
# that reference completed/cancelled tickets.
#
# Without --check-linear, this script just lists all TODO(OMN-XXXX) references
# found across repos (useful as a baseline audit).
#
# With --check-linear, it queries the Linear API for each ticket and flags any
# whose status is Done or Canceled.
#
# Usage:
#   python scripts/validation/check_stale_todos.py /path/to/omni_home
#   python scripts/validation/check_stale_todos.py /path/to/omni_home --check-linear
#
# Exit codes:
#   0 = no stale TODOs found (or --check-linear not used)
#   1 = stale TODOs found referencing completed tickets

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TODO_PATTERN = re.compile(r"TODO\s*\(?\s*(OMN-\d+)\s*\)?", re.IGNORECASE)

SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".next",
    ".turbo",
    ".onex_state",
}

SKIP_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
    ".whl",
    ".egg",
    ".lock",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".map",
}


def scan_file(path: Path) -> list[tuple[str, int, str]]:
    """Return list of (ticket_id, line_number, line_text) for TODOs in file."""
    results = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except (OSError, UnicodeDecodeError):
        return results

    for i, line in enumerate(text.splitlines(), start=1):
        for match in TODO_PATTERN.finditer(line):
            ticket_id = match.group(1).upper()
            results.append((ticket_id, i, line.strip()))
    return results


def scan_repos(omni_home: Path) -> dict[str, list[tuple[Path, int, str]]]:
    """Scan all repos under omni_home. Returns {ticket_id: [(file, line, text)]}."""
    findings: dict[str, list[tuple[Path, int, str]]] = {}

    for repo_dir in sorted(omni_home.iterdir()):
        if not repo_dir.is_dir() or repo_dir.name.startswith("."):
            continue

        for file_path in repo_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix in SKIP_EXTENSIONS:
                continue
            if any(part in SKIP_DIRS for part in file_path.parts):
                continue

            for ticket_id, line_no, line_text in scan_file(file_path):
                findings.setdefault(ticket_id, []).append(
                    (file_path, line_no, line_text)
                )

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan for TODO(OMN-XXXX) references across omni_home repos"
    )
    parser.add_argument(
        "omni_home",
        type=Path,
        nargs="?",
        default=Path("/Users/jonah/Code/omni_home"),
        help="Path to omni_home directory",
    )
    parser.add_argument(
        "--check-linear",
        action="store_true",
        help="Query Linear API to flag TODOs referencing completed tickets",
    )
    args = parser.parse_args()

    omni_home = args.omni_home.resolve()
    if not omni_home.is_dir():
        print(f"ERROR: {omni_home} is not a directory", file=sys.stderr)
        return 1

    print(f"Scanning {omni_home} for TODO(OMN-XXXX) references...")
    findings = scan_repos(omni_home)

    if not findings:
        print("No TODO(OMN-XXXX) references found.")
        return 0

    total_refs = sum(len(v) for v in findings.values())
    print(f"Found {total_refs} TODO references across {len(findings)} tickets.\n")

    if not args.check_linear:
        # Just print the inventory
        for ticket_id in sorted(findings):
            locations = findings[ticket_id]
            print(f"  {ticket_id} ({len(locations)} references):")
            for file_path, line_no, line_text in locations[:3]:
                rel = file_path.relative_to(omni_home)
                print(f"    {rel}:{line_no}: {line_text[:120]}")
            if len(locations) > 3:
                print(f"    ... and {len(locations) - 3} more")
        print(
            "\nRun with --check-linear to check ticket status against Linear API."
        )
        return 0

    # Check Linear for ticket status
    try:
        import os
        import json
        import urllib.request
    except ImportError:
        print("ERROR: urllib required for Linear API check", file=sys.stderr)
        return 1

    api_key = os.environ.get("LINEAR_API_KEY", "")
    if not api_key:
        print(
            "ERROR: LINEAR_API_KEY environment variable not set. "
            "Set it to check ticket status.",
            file=sys.stderr,
        )
        return 1

    stale_tickets: list[str] = []
    done_statuses = {"done", "canceled", "cancelled", "duplicate"}

    for ticket_id in sorted(findings):
        query = {
            "query": f"""
            query {{
                issueSearch(filter: {{ identifier: {{ eq: "{ticket_id}" }} }}) {{
                    nodes {{
                        identifier
                        title
                        state {{
                            name
                            type
                        }}
                    }}
                }}
            }}
            """
        }
        req = urllib.request.Request(
            "https://api.linear.app/graphql",
            data=json.dumps(query).encode("utf-8"),
            headers={
                "Authorization": api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                nodes = (
                    data.get("data", {})
                    .get("issueSearch", {})
                    .get("nodes", [])
                )
                for node in nodes:
                    state_type = node.get("state", {}).get("type", "").lower()
                    state_name = node.get("state", {}).get("name", "")
                    if state_type in done_statuses or state_name.lower() in done_statuses:
                        stale_tickets.append(ticket_id)
                        print(
                            f"  STALE: {ticket_id} — status: {state_name} "
                            f"({len(findings[ticket_id])} references)"
                        )
                        for fp, ln, lt in findings[ticket_id][:3]:
                            rel = fp.relative_to(omni_home)
                            print(f"    {rel}:{ln}: {lt[:120]}")
        except Exception as e:
            print(f"  WARN: Could not check {ticket_id}: {e}", file=sys.stderr)

    if stale_tickets:
        print(
            f"\n{len(stale_tickets)} stale TODO(s) referencing completed tickets."
        )
        return 1

    print("\nNo stale TODOs found — all referenced tickets are still open.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
