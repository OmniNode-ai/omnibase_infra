#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Standalone CI script: duplication sweep.

Checks:
  D1 — Drizzle table name duplication across omnidash schema files
  D2 — Topic registration conflicts between omniclaude/topics.py and kafka_boundaries.yaml
  D3 — Migration prefix conflicts (via check-migration-conflicts CLI)
  D4 — Cross-repo model name collisions (class ModelXxx defined in multiple repos)

Usage:
    python run_duplication_sweep.py [--omni-home /path/to/omni_home]
                                    [--checks D1,D2,D3,D4]
                                    [--fail-on-severity error|warning]
                                    [--json]

Exit codes:
    0 — no findings at or above --fail-on-severity
    1 — findings found at or above --fail-on-severity
    2 — usage/configuration error
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ALL_CHECKS = ["D1", "D2", "D3", "D4"]
_SEVERITY_ORDER = {"error": 2, "warning": 1}


# ---------------------------------------------------------------------------
# Check implementations
# ---------------------------------------------------------------------------


def check_d1_drizzle_tables(omni_home: Path) -> dict:
    """D1: Drizzle table name duplication across schema files."""
    schema_dir = omni_home / "omnidash" / "shared"
    if not schema_dir.is_dir():
        return {
            "check_id": "D1",
            "status": "WARN",
            "finding_count": 0,
            "severity": "warning",
            "detail": f"omnidash/shared/ not found at {schema_dir}",
            "findings": [],
        }

    table_pattern = re.compile(r'pgTable\("([^"]+)"')
    table_locations: dict[str, list[str]] = {}

    for schema_file in schema_dir.glob("*schema*.ts"):
        try:
            content = schema_file.read_text(encoding="utf-8")
        except OSError:
            continue
        for match in table_pattern.finditer(content):
            table_name = match.group(1)
            table_locations.setdefault(table_name, []).append(schema_file.name)

    duplicates = {
        name: files for name, files in table_locations.items() if len(files) > 1
    }

    if duplicates:
        detail = (
            f"{len(duplicates)} duplicate table(s): {', '.join(sorted(duplicates))}"
        )
        findings = [
            {
                "table": name,
                "files": files,
                "message": f"Table '{name}' defined in: {', '.join(files)}",
            }
            for name, files in sorted(duplicates.items())
        ]
        return {
            "check_id": "D1",
            "status": "FAIL",
            "finding_count": len(duplicates),
            "severity": "error",
            "detail": detail,
            "findings": findings,
        }

    return {
        "check_id": "D1",
        "status": "PASS",
        "finding_count": 0,
        "severity": "error",
        "detail": "No duplicate Drizzle tables",
        "findings": [],
    }


def check_d2_topic_conflicts(omni_home: Path) -> dict:
    """D2: Topic registration conflicts between omniclaude topics.py and kafka_boundaries.yaml."""
    topics_file = (
        omni_home / "omniclaude" / "src" / "omniclaude" / "hooks" / "topics.py"
    )
    boundaries_file = (
        omni_home
        / "onex_change_control"
        / "src"
        / "onex_change_control"
        / "boundaries"
        / "kafka_boundaries.yaml"
    )

    missing = []
    if not topics_file.is_file():
        missing.append(str(topics_file))
    if not boundaries_file.is_file():
        missing.append(str(boundaries_file))

    if missing:
        return {
            "check_id": "D2",
            "status": "WARN",
            "finding_count": 0,
            "severity": "warning",
            "detail": f"Topic source files not found: {', '.join(missing)}",
            "findings": [],
        }

    # Parse TopicBase enum values from topics.py
    topic_value_re = re.compile(r'=\s*"([^"]+)"')
    omniclaude_topics: set[str] = set()
    try:
        for line in topics_file.read_text(encoding="utf-8").splitlines():
            for m in topic_value_re.finditer(line):
                val = m.group(1)
                if val.startswith("onex."):
                    omniclaude_topics.add(val)
    except OSError:
        pass

    # Parse topic_name entries from kafka_boundaries.yaml (no yaml dep — simple grep)
    boundary_topic_re = re.compile(r"topic_name:\s*([^\s#]+)")
    # Also parse producer_repo for conflict detection
    boundary_entries: list[dict] = []
    try:
        content = boundaries_file.read_text(encoding="utf-8")
        current_topic = None
        current_producer = None
        for line in content.splitlines():
            tm = boundary_topic_re.search(line)
            if tm:
                if current_topic:
                    boundary_entries.append(
                        {"topic": current_topic, "producer_repo": current_producer}
                    )
                current_topic = tm.group(1).strip()
                current_producer = None
            pm = re.search(r"producer_repo:\s*(\S+)", line)
            if pm:
                current_producer = pm.group(1).strip()
        if current_topic:
            boundary_entries.append(
                {"topic": current_topic, "producer_repo": current_producer}
            )
    except OSError:
        pass

    # Cross-reference: find topics claimed by omniclaude that also appear in boundaries
    # with a different producer_repo
    conflicts = []
    for entry in boundary_entries:
        topic = entry["topic"]
        if topic in omniclaude_topics:
            producer = entry.get("producer_repo")
            if producer and producer != "omniclaude":
                conflicts.append(
                    {
                        "topic": topic,
                        "omniclaude_claims": True,
                        "boundary_producer": producer,
                        "message": (
                            f"Topic '{topic}' claimed by omniclaude but "
                            f"kafka_boundaries.yaml lists producer_repo={producer}"
                        ),
                    }
                )

    if conflicts:
        return {
            "check_id": "D2",
            "status": "FAIL",
            "finding_count": len(conflicts),
            "severity": "error",
            "detail": f"{len(conflicts)} conflicting topic(s)",
            "findings": conflicts,
        }

    return {
        "check_id": "D2",
        "status": "PASS",
        "finding_count": 0,
        "severity": "error",
        "detail": "No topic registration conflicts",
        "findings": [],
    }


def check_d3_migration_conflicts(omni_home: Path) -> dict:
    """D3: Migration prefix conflicts via check-migration-conflicts CLI."""
    change_control = omni_home / "onex_change_control"
    if not change_control.is_dir():
        return {
            "check_id": "D3",
            "status": "WARN",
            "finding_count": 0,
            "severity": "warning",
            "detail": "onex_change_control not found",
            "findings": [],
        }

    try:
        result = subprocess.run(
            ["uv", "run", "check-migration-conflicts", "--repos-root", str(omni_home)],
            cwd=str(change_control),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        output = result.stdout + result.stderr
    except FileNotFoundError:
        return {
            "check_id": "D3",
            "status": "WARN",
            "finding_count": 0,
            "severity": "warning",
            "detail": "check-migration-conflicts not available (uv not found)",
            "findings": [],
        }
    except subprocess.TimeoutExpired:
        return {
            "check_id": "D3",
            "status": "WARN",
            "finding_count": 0,
            "severity": "warning",
            "detail": "check-migration-conflicts timed out",
            "findings": [],
        }

    conflict_lines = [
        line
        for line in output.splitlines()
        if "EXACT_DUPLICATE" in line or "NAME_CONFLICT" in line
    ]

    if conflict_lines:
        return {
            "check_id": "D3",
            "status": "FAIL",
            "finding_count": len(conflict_lines),
            "severity": "error",
            "detail": f"{len(conflict_lines)} migration conflict(s)",
            "findings": [{"message": line.strip()} for line in conflict_lines],
        }

    if result.returncode != 0 and not conflict_lines:
        return {
            "check_id": "D3",
            "status": "WARN",
            "finding_count": 0,
            "severity": "warning",
            "detail": f"check-migration-conflicts exited {result.returncode} with no parsed conflicts",
            "findings": [],
        }

    return {
        "check_id": "D3",
        "status": "PASS",
        "finding_count": 0,
        "severity": "error",
        "detail": "No migration prefix conflicts",
        "findings": [],
    }


def check_d4_model_collisions(omni_home: Path) -> dict:
    """D4: Cross-repo model name collisions (same Model class name defined in multiple repos)."""
    class_pattern = re.compile(r"^class\s+(Model[A-Z]\w*)")
    model_locations: dict[str, list[dict]] = {}

    excluded_dirs = {".git", ".venv", "__pycache__", "migrations", "tests", "fixtures"}
    excluded_repos = {"omnibase_core"}

    for src_dir in omni_home.glob("*/src"):
        repo = src_dir.parent.name
        if repo in excluded_repos:
            continue
        for py_file in src_dir.rglob("*.py"):
            if any(part in excluded_dirs for part in py_file.parts):
                continue
            try:
                for line in py_file.read_text(encoding="utf-8").splitlines():
                    m = class_pattern.match(line.strip())
                    if m:
                        class_name = m.group(1)
                        model_locations.setdefault(class_name, []).append(
                            {
                                "repo": repo,
                                "file": str(py_file.relative_to(omni_home)),
                            }
                        )
            except (OSError, UnicodeDecodeError):
                continue

    duplicates = {name: locs for name, locs in model_locations.items() if len(locs) > 1}

    if duplicates:
        findings = [
            {
                "class_name": name,
                "locations": locs,
                "message": f"'{name}' defined in {len(locs)} repos: "
                + ", ".join(loc["repo"] for loc in locs),
            }
            for name, locs in sorted(duplicates.items())
        ]
        return {
            "check_id": "D4",
            "status": "FAIL",
            "finding_count": len(duplicates),
            "severity": "error",
            "detail": f"{len(duplicates)} cross-repo model name collision(s)",
            "findings": findings,
        }

    return {
        "check_id": "D4",
        "status": "PASS",
        "finding_count": 0,
        "severity": "error",
        "detail": "No cross-repo model name collisions",
        "findings": [],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Duplication sweep — standalone CI gate"
    )
    parser.add_argument(
        "--omni-home",
        metavar="PATH",
        default=os.environ.get("OMNI_HOME", str(Path.cwd())),
        help="Path to omni_home (default: $OMNI_HOME or cwd)",
    )
    parser.add_argument(
        "--checks",
        metavar="CHECK[,CHECK...]",
        default=",".join(ALL_CHECKS),
        help=f"Checks to run (default: all). Options: {', '.join(ALL_CHECKS)}",
    )
    parser.add_argument(
        "--fail-on-severity",
        metavar="LEVEL",
        default="error",
        choices=list(_SEVERITY_ORDER.keys()),
        help="Minimum severity to fail on (default: error)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of human-readable text",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    omni_home = Path(args.omni_home)
    if not omni_home.is_dir():
        print(f"ERROR: omni_home not found: {omni_home}", file=sys.stderr)
        return 2

    checks = [c.strip().upper() for c in args.checks.split(",") if c.strip()]
    fail_threshold = _SEVERITY_ORDER[args.fail_on_severity]

    check_map = {
        "D1": check_d1_drizzle_tables,
        "D2": check_d2_topic_conflicts,
        "D3": check_d3_migration_conflicts,
        "D4": check_d4_model_collisions,
    }

    results = []
    for check_id in ALL_CHECKS:
        if check_id not in checks:
            continue
        fn = check_map.get(check_id)
        if fn:
            results.append(fn(omni_home))

    fail_results = [
        r
        for r in results
        if r["status"] == "FAIL"
        and _SEVERITY_ORDER.get(r["severity"], 0) >= fail_threshold
    ]

    overall = "FAIL" if fail_results else "PASS"

    if args.json:
        output = {
            "sweep": "duplication_sweep",
            "omni_home": str(omni_home),
            "checks": checks,
            "fail_on_severity": args.fail_on_severity,
            "results": results,
            "status": overall,
            "blocking_count": len(fail_results),
        }
        print(json.dumps(output, indent=2))
    else:
        print("DUPLICATION SWEEP RESULTS")
        print("=========================")
        print()
        for r in results:
            print(f"{r['check_id']}: {r['status']} — {r['detail']}")
        print()
        print(f"Overall: {overall}")

    return 1 if fail_results else 0


if __name__ == "__main__":
    sys.exit(main())
