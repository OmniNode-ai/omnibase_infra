#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Audit GitHub Actions runner routing for trusted OmniNode CI.

The incident this protects against: repo-level GitHub variables drifted
OMNI_TRUSTED_CI_RUNS_ON_JSON back to ["ubuntu-latest"], so trusted CI silently
used GitHub-hosted minutes even though the workflow selector looked correct.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ORG = "OmniNode-ai"
DEFAULT_POLICY = Path("config/runner_routing_policy.yaml")


@dataclass(frozen=True)
class Finding:
    scope: str
    message: str


def _load_policy(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


def _canonical_json(value: str) -> str:
    return json.dumps(json.loads(value), separators=(",", ":"))


def _run_gh(args: list[str], timeout: int = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["gh", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _variables(args: list[str]) -> list[dict[str, Any]]:
    result = _run_gh(
        [
            "variable",
            "list",
            *args,
            "--json",
            "name,value",
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh variable list failed: {result.stderr.strip()}")
    values = json.loads(result.stdout or "[]")
    if not isinstance(values, list):
        raise RuntimeError("gh variable list returned a non-list payload")
    return values


def _variable_value(values: list[dict[str, Any]], name: str) -> str | None:
    for item in values:
        if item.get("name") == name:
            value = item.get("value")
            return value if isinstance(value, str) else None
    return None


def audit_github_variables(policy: dict[str, Any]) -> list[Finding]:
    variable = policy["trusted_runner_variable"]
    name = str(variable["name"])
    expected = _canonical_json(str(variable["expected_json"]))
    findings: list[Finding] = []
    org_actual = _variable_value(_variables(["--org", ORG]), name)
    if org_actual is None:
        findings.append(Finding(ORG, f"missing org variable {name}"))
    else:
        try:
            org_normalized = _canonical_json(org_actual)
        except json.JSONDecodeError:
            findings.append(Finding(ORG, f"{name} is not valid JSON: {org_actual!r}"))
            org_normalized = ""
        if org_normalized != expected:
            findings.append(
                Finding(
                    ORG,
                    f"{name} drifted to {org_actual!r}; expected {variable['expected_json']!r}",
                )
            )
    for repo in policy.get("repositories", []):
        repo_name = str(repo)
        actual = _variable_value(_variables(["--repo", f"{ORG}/{repo_name}"]), name)
        if actual is None:
            continue
        try:
            normalized = _canonical_json(actual)
        except json.JSONDecodeError:
            findings.append(Finding(repo_name, f"{name} is not valid JSON: {actual!r}"))
            continue
        if normalized != expected:
            findings.append(
                Finding(
                    repo_name,
                    f"{name} drifted to {actual!r}; expected {variable['expected_json']!r}",
                )
            )
    return findings


def _workflow_paths(repo_root: Path) -> list[Path]:
    workflow_dir = repo_root / ".github" / "workflows"
    if not workflow_dir.exists():
        return []
    return sorted(
        path
        for path in workflow_dir.iterdir()
        if path.suffix in {".yml", ".yaml"} and path.is_file()
    )


def audit_local_workflows(policy: dict[str, Any], repo_root: Path) -> list[Finding]:
    allowlist = {
        str(item["path"])
        for item in policy.get("hosted_runner_allowlist", [])
        if isinstance(item, dict) and "path" in item
    }
    bare_hosted = re.compile(
        r"^\s*runs-on:\s*(?:\[)?\s*ubuntu-latest\b", re.MULTILINE
    )
    findings: list[Finding] = []
    for path in _workflow_paths(repo_root):
        rel = path.relative_to(repo_root).as_posix()
        text = path.read_text(encoding="utf-8")
        if not bare_hosted.search(text):
            continue
        if rel in allowlist:
            continue
        findings.append(
            Finding(
                rel,
                "bare runs-on: ubuntu-latest is not allowed; use OMNI_RUNNER_SELECTOR_V1 or add an explicit policy exception",
            )
        )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--repo-root", type=Path, default=Path())
    parser.add_argument("--local-workflows", action="store_true")
    parser.add_argument("--github-vars", action="store_true")
    args = parser.parse_args()

    if not args.local_workflows and not args.github_vars:
        args.local_workflows = True
        args.github_vars = True

    policy = _load_policy(args.policy)
    findings: list[Finding] = []
    if args.local_workflows:
        findings.extend(audit_local_workflows(policy, args.repo_root))
    if args.github_vars:
        findings.extend(audit_github_variables(policy))

    if findings:
        for finding in findings:
            print(f"::error title=Runner routing drift::{finding.scope}: {finding.message}")
        return 1

    print("Runner routing audit passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
