# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dev-baseline publisher — OMN-13027.

Runs the full test suite on dev, parses JUnit XML output, computes failure
clusters (by test module path), writes ``config/validation/test-failure-baseline.yaml``,
and reports each failing cluster to the cluster-ticket node (OMN-19998),
which owns the ticket decision for the cluster.

Entry-point for the ``dev-baseline-publisher`` scheduled workflow.

Exit codes:
    0  — success (baseline written, failing clusters reported where configured)
    1  — unexpected internal error
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASELINE_FILE = Path("config/validation/test-failure-baseline.yaml")
JUNIT_XML = Path("dev-baseline-junit.xml")
BASELINE_TTL_DAYS = 7
# Number of failing-test examples stored per cluster in the baseline file and
# sent to the cluster-ticket node.
EXAMPLE_COUNT = 5


# ---------------------------------------------------------------------------
# JUnit parsing helpers
# ---------------------------------------------------------------------------


def parse_junit(junit_path: Path) -> dict[str, list[str]]:
    """Return mapping of cluster_key -> list of failed test nodeids.

    cluster_key is the dotted module path of the test (e.g.
    ``tests.unit.test_foo``).  All test cases with ``failure`` or ``error``
    child elements are included.
    """
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
    # JUnit XML may have <testsuites> wrapping <testsuite>, or bare <testsuite>.
    test_suites = root.findall(".//testsuite") or [root]

    clusters: dict[str, list[str]] = defaultdict(list)

    for suite in test_suites:
        for tc in suite.findall("testcase"):
            failed = tc.find("failure") is not None or tc.find("error") is not None
            if not failed:
                continue
            classname = tc.get("classname", "")
            name = tc.get("name", "")
            # classname is typically the dotted module path
            cluster = classname.rsplit(".", 1)[0] if "." in classname else classname
            if not cluster:
                cluster = "unknown"
            node_id = f"{classname}::{name}"
            clusters[cluster].append(node_id)

    return dict(clusters)


# ---------------------------------------------------------------------------
# Baseline I/O
# ---------------------------------------------------------------------------


def load_baseline(path: Path) -> dict[str, Any]:
    """Load existing baseline YAML, returning empty structure on missing file."""
    if not path.exists():
        return {"generated_at": "", "clusters": {}}

    with path.open() as fh:
        data = yaml.safe_load(fh) or {}

    if "clusters" not in data:
        data["clusters"] = {}
    return data


def save_baseline(path: Path, data: dict[str, Any]) -> None:
    """Write baseline YAML atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".yaml.tmp")
    with tmp.open("w") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=True)
    tmp.replace(path)


def prune_expired(clusters: dict[str, Any], now: datetime.datetime) -> dict[str, Any]:
    """Remove entries whose ``expires_at`` is in the past."""
    cutoff = now.isoformat()
    return {k: v for k, v in clusters.items() if v.get("expires_at", "") >= cutoff}


# ---------------------------------------------------------------------------
# Cluster-ticket node reporting (OMN-19998)
# ---------------------------------------------------------------------------


def report_cluster(
    report_argv: list[str],
    *,
    repo: str,
    cluster: str,
    tests: list[str],
    run_ref: str,
    observed_at: str,
    lane: str,
    timeout_s: int,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, str]:
    """Report one failing cluster to the cluster-ticket node; reporting never fails the run.

    ``report_argv`` is the node command already split with ``shlex``; this appends
    ``--input``/``--timeout`` and runs it. The node's stdout ``result`` comes back as
    ``{"state": ..., "ticket": ...}``; a nonzero exit, a timeout or an unparseable answer
    is ``unavailable`` with a warning that names the cluster, never the command.
    """
    node_input = {
        "requested_by_lane": lane,
        "repo": repo,
        "cluster": cluster,
        "failing_count": len(tests),
        "examples": tests[:EXAMPLE_COUNT],
        "run_url": run_ref,
        "observed_at": observed_at,
    }
    cmd = [*report_argv, "--input", json.dumps(node_input), "--timeout", str(timeout_s)]
    try:
        proc = runner(
            cmd, capture_output=True, text=True, check=False, timeout=timeout_s + 30
        )
        if proc.returncode != 0:
            raise ValueError(f"exit {proc.returncode}")
        result = json.loads(proc.stdout)["result"]
        return {"state": str(result["state"]), "ticket": str(result["ticket"] or "")}
    except (subprocess.TimeoutExpired, ValueError, KeyError, TypeError) as exc:
        print(
            f"::warning::Cluster-ticket node gave no answer for cluster {cluster}: "
            f"{type(exc).__name__}",
            file=sys.stderr,
        )
        return {"state": "unavailable", "ticket": ""}


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def run_test_suite() -> int:
    """Run the full test suite, writing JUnit XML. Return exit code."""
    cmd = [
        "uv",
        "run",
        "pytest",
        "tests/",
        "--ignore=tests/integration/docker",
        "-m",
        "not slow and not chaos and not kafka and not performance",
        "--tb=no",
        "-q",
        f"--junitxml={JUNIT_XML}",
        "--timeout=60",
        "--timeout-method=signal",
        "-p",
        "no:cacheprovider",
    ]
    print(f"Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=False)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Publish dev test-failure baseline (OMN-13027)"
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", "OmniNode-ai/omnibase_infra"),
        help="Repo slug (owner/name)",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip running pytest (use existing JUnit XML at dev-baseline-junit.xml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not report clusters to the node or write the baseline",
    )
    parser.add_argument(
        "--report-command",
        default=os.environ.get("BASELINE_REPORT_COMMAND", ""),
        help=(
            "Cluster-ticket node command as one shell-quoted string "
            "(default: env BASELINE_REPORT_COMMAND)"
        ),
    )
    parser.add_argument(
        "--report-lane",
        default=os.environ.get("BASELINE_REPORT_LANE", ""),
        help=(
            "requested_by_lane sent to the cluster-ticket node "
            "(default: env BASELINE_REPORT_LANE)"
        ),
    )
    parser.add_argument(
        "--report-timeout-s",
        type=int,
        default=120,
        help="Timeout in seconds passed to the cluster-ticket node",
    )
    args = parser.parse_args(argv)

    now = datetime.datetime.now(datetime.UTC)

    # 1. Optionally run tests
    if not args.skip_run:
        rc = run_test_suite()
        if rc not in (0, 1):  # 0=all pass, 1=some failures — both are valid
            print(f"::error::pytest exited with unexpected code {rc}", file=sys.stderr)
            return 1

    # 2. Parse failures
    current_failures = parse_junit(JUNIT_XML)
    print(
        f"Parsed {sum(len(v) for v in current_failures.values())} failing tests "
        f"in {len(current_failures)} clusters from {JUNIT_XML}"
    )

    # 3. Load existing baseline
    baseline = load_baseline(BASELINE_FILE)
    existing_clusters: dict[str, Any] = baseline.get("clusters", {})

    # Prune expired entries first
    existing_clusters = prune_expired(existing_clusters, now)

    # 4. Determine new vs existing clusters
    new_clusters = set(current_failures.keys()) - set(existing_clusters.keys())
    print(f"New clusters (not in baseline): {sorted(new_clusters)}")

    # 5. Report every failing cluster to the cluster-ticket node (OMN-19998).
    # The node owns the ticket decision and searches by the cluster's stable
    # signature, so it is consulted on EVERY failing cluster of the run — not
    # only on clusters that are new to the baseline file.
    report_argv = shlex.split(args.report_command) if args.report_command else []
    reporting_on = not args.dry_run and bool(report_argv) and bool(args.report_lane)
    if not args.dry_run and current_failures and not reporting_on:
        print(
            "::warning::Skipping failure-cluster reporting: the report command "
            "or the report lane is not configured",
            file=sys.stderr,
        )

    run_id = os.environ.get("GITHUB_RUN_ID", "")
    run_ref = f"{args.repo} Actions run {run_id}" if run_id else ""
    observed_at = now.isoformat()
    repo_name = args.repo.split("/")[-1]

    # 6. Build updated clusters dict
    expires_at = (now + datetime.timedelta(days=BASELINE_TTL_DAYS)).isoformat()

    for cluster, tests in current_failures.items():
        answered_ticket: str | None = None
        if reporting_on:
            answer = report_cluster(
                report_argv,
                repo=repo_name,
                cluster=cluster,
                tests=tests,
                run_ref=run_ref,
                observed_at=observed_at,
                lane=args.report_lane,
                timeout_s=args.report_timeout_s,
            )
            if answer["state"] in ("created", "commented", "linked"):
                answered_ticket = answer["ticket"]

        if cluster in existing_clusters:
            # Refresh expiry on still-failing clusters; keep the entry's
            # ticket unless the node answered with one this run.
            entry = existing_clusters[cluster]
            if answered_ticket is not None:
                entry["ticket"] = answered_ticket
            entry["expires_at"] = expires_at
            entry["count"] = len(tests)
            entry["examples"] = tests[:EXAMPLE_COUNT]
        else:
            # New failure cluster; ticket comes from the node's answer, or
            # "" when it could not answer.
            existing_clusters[cluster] = {
                "ticket": answered_ticket or "",
                "expires_at": expires_at,
                "count": len(tests),
                "examples": tests[:EXAMPLE_COUNT],
                "first_seen": now.isoformat(),
            }

    # Remove clusters that no longer fail (they passed this run)
    resolved = set(existing_clusters.keys()) - set(current_failures.keys())
    for cluster in resolved:
        del existing_clusters[cluster]
        print(f"Cluster resolved (tests passing): {cluster}")

    # 7. Write updated baseline
    baseline["generated_at"] = now.isoformat()
    baseline["generated_by"] = "dev-baseline-publisher"
    baseline["repo"] = args.repo
    baseline["ttl_days"] = BASELINE_TTL_DAYS
    baseline["clusters"] = existing_clusters

    if not args.dry_run:
        save_baseline(BASELINE_FILE, baseline)
        print(
            f"Baseline written to {BASELINE_FILE} ({len(existing_clusters)} clusters)"
        )
    else:
        print(
            f"[dry-run] Would write {len(existing_clusters)} clusters to {BASELINE_FILE}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
