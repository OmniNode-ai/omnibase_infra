#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Validate PR ticket contracts — Layer 1 gate (OMN-8909).

Enforces:
- PRs with OMN-XXXX ticket refs touching runtime code must have a contract YAML
- Contract dod_evidence must be non-empty for runtime-touching PRs

Inputs (env vars):
    PR_DIFF_FILES   newline-separated changed file paths
    PR_BRANCH       branch name
    PR_TITLE        PR title
    PR_BODY         PR body text (optional)
    CONTRACTS_PATH  path to contracts directory
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import yaml

TICKET_PATTERN = re.compile(r"OMN-(\d+)", re.IGNORECASE)

RUNTIME_PATH_PATTERNS = [
    re.compile(r"^src/.*/nodes/.*/handler.*\.py$"),
    re.compile(r"^src/.*/nodes/.*\.py$"),
    re.compile(r"^plugins/onex/skills/.*/SKILL\.md$"),
]


def extract_ticket_ids(branch: str, title: str, body: str) -> set[str]:
    text = f"{branch} {title} {body}"
    return {f"OMN-{m}" for m in TICKET_PATTERN.findall(text)}


def is_runtime_file(path: str) -> bool:
    return any(p.match(path) for p in RUNTIME_PATH_PATTERNS)


def has_runtime_changes(diff_files: list[str]) -> bool:
    return any(is_runtime_file(f) for f in diff_files)


class Finding:
    def __init__(self, ticket_id: str, message: str, severity: str = "error") -> None:
        self.ticket_id = ticket_id
        self.message = message
        self.severity = severity

    def to_dict(self) -> dict[str, str]:
        return {
            "ticket_id": self.ticket_id,
            "message": self.message,
            "severity": self.severity,
        }


def validate(
    diff_files: list[str],
    branch: str,
    title: str,
    body: str,
    contracts_path: Path,
) -> list[Finding]:
    ticket_ids = extract_ticket_ids(branch, title, body)

    if not ticket_ids:
        print("SKIP: No OMN ticket references found — skipping contract gate.")
        return []

    runtime_touched = has_runtime_changes(diff_files)

    if not runtime_touched:
        print(
            f"SKIP: No runtime files changed — skipping contract gate. Tickets: {sorted(ticket_ids)}"
        )
        return []

    findings: list[Finding] = []

    for ticket_id in sorted(ticket_ids):
        contract_file = contracts_path / f"{ticket_id}.yaml"

        if not contract_file.exists():
            findings.append(
                Finding(
                    ticket_id=ticket_id,
                    message=f"Missing contract: {contract_file.name} not found in {contracts_path}",
                )
            )
            continue

        try:
            with open(contract_file, encoding="utf-8") as f:
                contract_data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            findings.append(
                Finding(
                    ticket_id=ticket_id,
                    message=f"Contract file {contract_file.name} is invalid YAML: {exc}",
                )
            )
            continue

        if contract_data is None:
            findings.append(
                Finding(
                    ticket_id=ticket_id,
                    message=f"Contract file {contract_file.name} is empty or invalid YAML",
                )
            )
            continue

        if not isinstance(contract_data, dict):
            findings.append(
                Finding(
                    ticket_id=ticket_id,
                    message=f"Contract file {contract_file.name} must contain a YAML object at root",
                )
            )
            continue

        dod_evidence = contract_data.get("dod_evidence", [])
        if not dod_evidence:
            findings.append(
                Finding(
                    ticket_id=ticket_id,
                    message=f"Contract {contract_file.name} has empty dod_evidence — runtime-touching PRs require non-empty dod_evidence",
                )
            )

    return findings


def main() -> int:
    diff_files_raw = os.environ.get("PR_DIFF_FILES", "")
    branch = os.environ.get("PR_BRANCH", "")
    title = os.environ.get("PR_TITLE", "")
    body = os.environ.get("PR_BODY", "")
    contracts_path_str = os.environ.get("CONTRACTS_PATH", "contracts")

    diff_files = [f.strip() for f in diff_files_raw.strip().splitlines() if f.strip()]
    contracts_path = Path(contracts_path_str)

    findings = validate(diff_files, branch, title, body, contracts_path)

    if not findings:
        print("PASS: All contract checks passed.")
        return 0

    print(f"FAIL: {len(findings)} contract violation(s) found:\n")
    for f in findings:
        print(f"  [{f.severity.upper()}] {f.ticket_id}: {f.message}")

    print(f"\n{json.dumps([f.to_dict() for f in findings], indent=2)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
