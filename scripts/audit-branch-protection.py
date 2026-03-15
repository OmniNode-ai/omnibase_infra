#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Audit branch protection required checks against actual CI check names.

Detects mismatches where branch protection requires a status check name
that no longer exists in the CI workflow (e.g., after a job was renamed).

Usage:
    python3 scripts/audit-branch-protection.py          # audit all repos
    python3 scripts/audit-branch-protection.py --fix     # audit + fix mismatches
    python3 scripts/audit-branch-protection.py --repos omniclaude,omniweb
"""

import argparse
import json
import subprocess
import sys

REPOS = [
    "omniclaude",
    "omnibase_core",
    "omnibase_infra",
    "omnibase_spi",
    "omnidash",
    "omniintelligence",
    "omnimemory",
    "omninode_infra",
    "omniweb",
    "onex_change_control",
]

ORG = "OmniNode-ai"


def run_gh(args: list[str], timeout: int = 15) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["gh", *args], capture_output=True, text=True, timeout=timeout, check=False
    )


def get_required_checks(repo: str) -> list[str]:
    r = run_gh(
        [
            "api",
            f"repos/{ORG}/{repo}/branches/main/protection/required_status_checks",
            "--jq",
            ".contexts",
        ]
    )
    if r.returncode != 0:
        return []
    try:
        return json.loads(r.stdout) if r.stdout.strip() else []
    except json.JSONDecodeError:
        return []


def get_actual_checks(repo: str) -> set[str]:
    r = run_gh(
        [
            "pr",
            "list",
            "--repo",
            f"{ORG}/{repo}",
            "--state",
            "all",
            "--limit",
            "1",
            "--json",
            "number",
        ]
    )
    prs = json.loads(r.stdout) if r.stdout.strip() else []
    if not prs:
        return set()
    pr_num = str(prs[0]["number"])
    r2 = run_gh(["pr", "checks", pr_num, "--repo", f"{ORG}/{repo}"])
    checks: set[str] = set()
    for line in (r2.stdout + r2.stderr).strip().split("\n"):
        if "\t" in line:
            checks.add(line.split("\t")[0].strip())
    return checks


def fix_mismatches(repo: str, required: list[str], actual: set[str]) -> bool:
    valid = [r for r in required if r in actual]
    if not valid:
        print(
            "  Cannot fix: no required checks match actual CI. Manual intervention needed."
        )
        return False
    removed = [r for r in required if r not in actual]
    payload = json.dumps({"strict": True, "contexts": valid})
    r = subprocess.run(
        [
            "gh",
            "api",
            "-X",
            "PATCH",
            f"repos/{ORG}/{repo}/branches/main/protection/required_status_checks",
            "--input",
            "-",
        ],
        input=payload,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if r.returncode == 0 or "contexts" in r.stdout:
        print(f"  Fixed: removed {removed}, kept {valid}")
        return True
    print(f"  Fix failed: {r.stderr}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit branch protection check names")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix mismatches by removing stale checks",
    )
    parser.add_argument(
        "--repos", type=str, help="Comma-separated repo names (default: all)"
    )
    args = parser.parse_args()

    repos = args.repos.split(",") if args.repos else REPOS
    mismatches_found = 0

    for repo in repos:
        required = get_required_checks(repo)
        if not required:
            continue

        actual = get_actual_checks(repo)
        if not actual:
            print(f"=== {repo} === (no PRs to check against)")
            continue

        stale = [r for r in required if r not in actual]

        if stale:
            mismatches_found += 1
            print(f"=== {repo} ===")
            print(f"  Required: {required}")
            print(f"  MISMATCHES: {stale}")
            print(f"  Actual CI checks: {sorted(actual)}")
            if args.fix:
                fix_mismatches(repo, required, actual)
            print()
        else:
            print(f"=== {repo} === ✅")

    if mismatches_found == 0:
        print("\nAll repos: branch protection checks match CI. No mismatches found.")
        return 0
    else:
        print(f"\n{mismatches_found} repo(s) with mismatches.")
        if not args.fix:
            print("Run with --fix to auto-remove stale checks.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
