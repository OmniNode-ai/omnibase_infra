#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""CLI wrapper for audit_branch_protection_lib (OMN-9034).

The shell script `audit-branch-protection.sh` invokes this CLI twice:
  1. `audit --owner ORG --repo REPO` → returns JSON describing the audit
     result (status, message, protection_json, etc.).
  2. `fix-payload --protection-json '...'` → returns JSON payload ready to
     PUT to the branch protection endpoint.

This separation keeps all logic testable in Python while letting the shell
retain control of the `gh api PUT` side effect (so no real gh calls leak
into unit tests).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from audit_branch_protection_lib import audit_repo, build_fix_payload


def real_gh(args: list[str]) -> tuple[int, str]:
    """Invoke the real `gh` CLI. Returns (returncode, stdout)."""
    r = subprocess.run(
        ["gh", *args], capture_output=True, text=True, timeout=30, check=False
    )
    return r.returncode, r.stdout


def cmd_audit(args: argparse.Namespace) -> int:
    result = audit_repo(args.owner, args.repo, real_gh, commits_to_scan=args.commits)
    print(json.dumps(result))
    return 0


def cmd_fix_payload(args: argparse.Namespace) -> int:
    payload = build_fix_payload(args.protection_json)
    print(json.dumps(payload))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_audit = sub.add_parser("audit", help="Audit a repo's branch protection")
    p_audit.add_argument("--owner", required=True)
    p_audit.add_argument("--repo", required=True)
    p_audit.add_argument("--commits", type=int, default=5)
    p_audit.set_defaults(func=cmd_audit)

    p_fix = sub.add_parser("fix-payload", help="Build the PUT payload for --fix")
    p_fix.add_argument("--protection-json", required=True)
    p_fix.set_defaults(func=cmd_fix_payload)

    ns = parser.parse_args()
    return int(ns.func(ns))


if __name__ == "__main__":
    sys.exit(main())
