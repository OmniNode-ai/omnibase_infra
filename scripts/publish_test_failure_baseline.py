# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dev-baseline publisher — OMN-13027.

Runs the full test suite on dev, parses JUnit XML output, computes failure
clusters (by test module path), writes ``config/validation/test-failure-baseline.yaml``,
and auto-files Urgent Linear tickets for any new cluster that has no tracking
ticket yet.

Entry-point for the ``dev-baseline-publisher`` scheduled workflow.

Exit codes:
    0  — success (baseline written, tickets filed where needed)
    1  — unexpected internal error
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASELINE_FILE = Path("config/validation/test-failure-baseline.yaml")
JUNIT_XML = Path("dev-baseline-junit.xml")
BASELINE_TTL_DAYS = 7
LINEAR_API_URL = "https://api.linear.app/graphql"
# Omninode team + "Urgent" priority
LINEAR_PRIORITY_URGENT = 1
LINEAR_TEAM_KEY = "OMN"


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
# Linear ticket filing
# ---------------------------------------------------------------------------


def _graphql(api_key: str, query: str, variables: dict[str, Any]) -> dict[str, Any]:
    """Execute a Linear GraphQL request; raise on HTTP errors."""
    payload = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(  # noqa: S310 — LINEAR_API_URL is a constant HTTPS endpoint
        LINEAR_API_URL,
        data=payload,
        headers={
            "Authorization": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"Linear HTTP {exc.code}: {body}") from exc


def get_team_id(api_key: str, team_key: str) -> str:
    """Resolve Linear team ID by key."""
    query = """
    query GetTeam($key: String!) {
        teams(filter: { key: { eq: $key } }) {
            nodes { id key name }
        }
    }
    """
    resp = _graphql(api_key, query, {"key": team_key})
    nodes = resp.get("data", {}).get("teams", {}).get("nodes", [])
    if not nodes:
        raise RuntimeError(f"No Linear team found with key '{team_key}'")
    return str(nodes[0]["id"])


def create_linear_ticket(
    api_key: str,
    team_id: str,
    cluster: str,
    examples: list[str],
    repo: str,
) -> str:
    """Create an Urgent Linear ticket for a new failure cluster. Returns issue ID."""
    example_list = "\n".join(f"- `{t}`" for t in examples[:5])
    title = f"[dev-baseline] Test failure cluster: {cluster} ({repo})"
    description = (
        f"## Dev-baseline ratchet: new failure cluster detected\n\n"
        f"**Repo:** `{repo}`  \n"
        f"**Cluster:** `{cluster}`  \n\n"
        f"### Failing tests (first {min(5, len(examples))} of {len(examples)})\n\n"
        f"{example_list}\n\n"
        f"This ticket was auto-filed by the `dev-baseline-publisher` workflow "
        f"(OMN-13027). The cluster entry will expire from the baseline after "
        f"{BASELINE_TTL_DAYS} days if not resolved. Fix the failures and delete "
        f"the cluster entry from `config/validation/test-failure-baseline.yaml` "
        f"to drain the ratchet.\n\n"
        f"**Do not ignore this ticket.** No PR may introduce a new failure not "
        f"listed in the baseline."
    )

    mutation = """
    mutation CreateIssue($teamId: String!, $title: String!, $description: String!, $priority: Int!) {
        issueCreate(input: {
            teamId: $teamId
            title: $title
            description: $description
            priority: $priority
        }) {
            success
            issue { id identifier }
        }
    }
    """
    resp = _graphql(
        api_key,
        mutation,
        {
            "teamId": team_id,
            "title": title,
            "description": description,
            "priority": LINEAR_PRIORITY_URGENT,
        },
    )
    issue = resp.get("data", {}).get("issueCreate", {}).get("issue")
    if not issue:
        raise RuntimeError(f"Linear ticket creation failed: {resp}")
    return str(issue["identifier"])


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
        "--timeout-method=thread",
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
        help="Do not file Linear tickets or commit changes",
    )
    parser.add_argument(
        "--linear-api-key",
        default=os.environ.get("LINEAR_API_KEY", ""),
        help="Linear API key for filing tickets",
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

    # 5. File Linear tickets for new clusters
    team_id: str | None = None
    if new_clusters and args.linear_api_key and not args.dry_run:
        try:
            team_id = get_team_id(args.linear_api_key, LINEAR_TEAM_KEY)
        except Exception as exc:  # noqa: BLE001 — graceful degradation: no ticket if API unreachable
            print(f"::warning::Could not resolve Linear team: {exc}", file=sys.stderr)

    # 6. Build updated clusters dict
    expires_at = (now + datetime.timedelta(days=BASELINE_TTL_DAYS)).isoformat()

    for cluster, tests in current_failures.items():
        if cluster in existing_clusters:
            # Refresh expiry on still-failing clusters; preserve ticket
            existing_clusters[cluster]["expires_at"] = expires_at
            existing_clusters[cluster]["count"] = len(tests)
            existing_clusters[cluster]["examples"] = tests[:5]
        else:
            # New failure cluster
            ticket_id = ""
            if team_id and not args.dry_run:
                try:
                    repo_name = args.repo.split("/")[-1]
                    ticket_id = create_linear_ticket(
                        args.linear_api_key, team_id, cluster, tests, repo_name
                    )
                    print(f"Filed Linear ticket {ticket_id} for cluster: {cluster}")
                except Exception as exc:  # noqa: BLE001 — graceful degradation: skip ticket on API failure
                    print(
                        f"::warning::Failed to file ticket for cluster {cluster}: {exc}",
                        file=sys.stderr,
                    )

            existing_clusters[cluster] = {
                "ticket": ticket_id,
                "expires_at": expires_at,
                "count": len(tests),
                "examples": tests[:5],
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
