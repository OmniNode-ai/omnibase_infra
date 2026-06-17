# SPDX-FileCopyrightText: 2026 OmniNode Team
# SPDX-License-Identifier: MIT
"""PR ratchet check — OMN-13027.

Compares the test failures from a PR run against the dev-branch baseline
(``config/validation/test-failure-baseline.yaml``).  Any failure cluster NOT
present in the baseline causes a non-zero exit, blocking the PR.

Usage (CI)::

    uv run python scripts/check_test_failure_ratchet.py \\
        --junit-xml pr-junit.xml \\
        --baseline config/validation/test-failure-baseline.yaml

Exit codes:
    0  — all failures are in the baseline (PR is clean)
    1  — one or more NEW failure clusters detected (PR blocked)
    2  — baseline file missing or unreadable (gate cannot operate)
"""

from __future__ import annotations

import argparse
import datetime
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Shared helpers (duplicated from publisher to keep scripts self-contained)
# ---------------------------------------------------------------------------


def parse_junit(junit_path: Path) -> dict[str, list[str]]:
    """Return mapping of cluster_key -> list of failed test nodeids."""
    if not junit_path.exists():
        return {}

    try:
        tree = ET.parse(junit_path)  # noqa: S314 — JUnit XML is CI-generated, not untrusted
    except ET.ParseError as exc:
        print(
            f"::warning::Failed to parse JUnit XML {junit_path}: {exc}", file=sys.stderr
        )
        return {}

    root = tree.getroot()
    test_suites = root.findall(".//testsuite") or [root]

    clusters: dict[str, list[str]] = defaultdict(list)

    for suite in test_suites:
        for tc in suite.findall("testcase"):
            failed = tc.find("failure") is not None or tc.find("error") is not None
            if not failed:
                continue
            classname = tc.get("classname", "")
            name = tc.get("name", "")
            cluster = classname.rsplit(".", 1)[0] if "." in classname else classname
            if not cluster:
                cluster = "unknown"
            node_id = f"{classname}::{name}"
            clusters[cluster].append(node_id)

    return dict(clusters)


def load_baseline(path: Path) -> dict[str, object]:
    """Load baseline YAML; return empty clusters dict on missing file."""
    if not path.exists():
        return {}
    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    return dict(data.get("clusters", {}))


def is_expired(entry: dict[str, object], now: datetime.datetime) -> bool:
    """Return True if the baseline entry has passed its TTL."""
    expires_at = str(entry.get("expires_at", ""))
    if not expires_at:
        return False
    return expires_at < now.isoformat()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ratchet gate: fail PR if test failures are not in baseline (OMN-13027)"
    )
    parser.add_argument(
        "--junit-xml",
        default="pr-junit.xml",
        help="JUnit XML produced by the PR test run",
    )
    parser.add_argument(
        "--baseline",
        default="config/validation/test-failure-baseline.yaml",
        help="Path to the test-failure baseline YAML",
    )
    parser.add_argument(
        "--allow-expired",
        action="store_true",
        help="Treat expired baseline entries as still valid (for testing)",
    )
    args = parser.parse_args(argv)

    junit_path = Path(args.junit_xml)
    baseline_path = Path(args.baseline)

    # Gate cannot operate without a baseline — fail loudly rather than silently pass.
    if not baseline_path.exists():
        print(
            f"::error::Baseline file not found: {baseline_path}\n"
            "Run the dev-baseline-publisher workflow to generate it.",
            file=sys.stderr,
        )
        return 2

    now = datetime.datetime.now(datetime.UTC)

    baseline_clusters = load_baseline(baseline_path)
    current_failures = parse_junit(junit_path)

    if not current_failures:
        print("No test failures found in JUnit XML — gate passes.")
        return 0

    new_clusters: list[str] = []
    expired_clusters: list[str] = []

    for cluster, tests in current_failures.items():
        if cluster not in baseline_clusters:
            new_clusters.append(cluster)
            continue

        entry = dict(baseline_clusters[cluster])  # type: ignore[arg-type]
        if not args.allow_expired and is_expired(entry, now):
            expired_clusters.append(cluster)

    if not new_clusters and not expired_clusters:
        total = sum(len(v) for v in current_failures.values())
        print(
            f"All {total} failing tests across {len(current_failures)} clusters "
            "are in the baseline — gate passes."
        )
        return 0

    # Report failures
    print(file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    print("TEST-FAILURE RATCHET GATE BLOCKED (OMN-13027)", file=sys.stderr)
    print("=" * 72, file=sys.stderr)

    if new_clusters:
        print(
            f"\n::error::{len(new_clusters)} NEW failure cluster(s) not in baseline:",
            file=sys.stderr,
        )
        for cluster in sorted(new_clusters):
            examples = current_failures[cluster]
            print(f"\n  Cluster: {cluster}", file=sys.stderr)
            for t in examples[:3]:
                print(f"    - {t}", file=sys.stderr)
            if len(examples) > 3:
                print(f"    ... and {len(examples) - 3} more", file=sys.stderr)

    if expired_clusters:
        print(
            f"\n::error::{len(expired_clusters)} EXPIRED baseline cluster(s) "
            "(TTL passed — fix or renew via dev-baseline-publisher run):",
            file=sys.stderr,
        )
        for cluster in sorted(expired_clusters):
            print(f"  - {cluster}", file=sys.stderr)

    print(
        "\nTo fix:\n"
        "  1. Fix the failing tests, OR\n"
        "  2. For pre-existing failures: run dev-baseline-publisher on dev to "
        "add them to the baseline (a Linear ticket will be auto-filed).\n"
        "  3. Never add entries to the baseline manually without a ticket.\n",
        file=sys.stderr,
    )
    print("=" * 72, file=sys.stderr)

    return 1


if __name__ == "__main__":
    sys.exit(main())
