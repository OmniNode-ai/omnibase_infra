#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Standalone CI script: handler contract compliance sweep.

Scans Python handler files for:
  HARDCODED_TOPIC      — topic strings like "onex.evt.*" embedded in code
  UNDECLARED_TRANSPORT — direct psycopg/httpx/requests/etc. imports
  LOGIC_IN_NODE        — business logic classes/methods in node.py files

Usage:
    python run_compliance_sweep.py [--repo /path/to/repo] [--repos /path/to/repo1,/path/to/repo2]
                                   [--checks hardcoded-topics,undeclared-transport,logic-in-node]
                                   [--fail-on-severity error|warning|critical]
                                   [--json]

Exit codes:
    0 — no findings at or above --fail-on-severity
    1 — findings found at or above --fail-on-severity
    2 — usage/configuration error
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (ported from handler_compliance_sweep.py)
# ---------------------------------------------------------------------------

_EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    "migrations",
}

_HARDCODED_TOPIC_RE = re.compile(r'"onex\.[a-z]+\.[a-z]+\.[a-z]')

_TRANSPORT_IMPORTS = {
    "psycopg",
    "psycopg2",
    "asyncpg",
    "httpx",
    "requests",
    "aiohttp",
    "sqlalchemy",
    "boto3",
}

_LOGIC_INDICATORS = [
    re.compile(r"class\s+\w+.*:"),
    re.compile(r"def\s+(handle|process|execute)\s*\("),
]

_SEVERITY_ORDER = {"critical": 3, "error": 2, "warning": 1}

ALL_CHECKS = ["hardcoded-topics", "undeclared-transport", "logic-in-node"]


# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []


def _infer_node_name(handler_file: Path, repo_root: Path) -> str:
    for part in handler_file.relative_to(repo_root).parts:
        if part.startswith("node_"):
            return part
    return handler_file.stem


def _find_handler_files(root: Path) -> list[Path]:
    results = []
    for py_file in root.rglob("*.py"):
        if any(part in _EXCLUDED_DIRS for part in py_file.parts):
            continue
        if (
            "handler" in py_file.stem
            or py_file.parent.name == "handlers"
            or py_file.stem == "node"
            or py_file.name == "__init__.py"
        ):
            results.append(py_file)
    return sorted(results)


def _check_hardcoded_topics(
    repo: str, rel_path: str, node: str, lines: list[str]
) -> list[dict]:
    findings = []
    for i, line in enumerate(lines, 1):
        if _HARDCODED_TOPIC_RE.search(line):
            findings.append(
                {
                    "repo": repo,
                    "file": rel_path,
                    "node": node,
                    "violation_type": "HARDCODED_TOPIC",
                    "severity": "error",
                    "line": i,
                    "message": f"Hardcoded topic string: {line.strip()[:80]}",
                }
            )
    return findings


def _check_transport_imports(
    repo: str, rel_path: str, node: str, handler_file: Path
) -> list[dict]:
    findings = []
    try:
        source = handler_file.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return []

    for ast_node in ast.walk(tree):
        if isinstance(ast_node, ast.Import):
            for alias in ast_node.names:
                root_module = alias.name.split(".")[0]
                if root_module in _TRANSPORT_IMPORTS:
                    findings.append(
                        {
                            "repo": repo,
                            "file": rel_path,
                            "node": node,
                            "violation_type": "UNDECLARED_TRANSPORT",
                            "severity": "warning",
                            "line": ast_node.lineno,
                            "message": f"Transport import: {alias.name}",
                        }
                    )
        elif isinstance(ast_node, ast.ImportFrom) and ast_node.module:
            root_module = ast_node.module.split(".")[0]
            if root_module in _TRANSPORT_IMPORTS:
                findings.append(
                    {
                        "repo": repo,
                        "file": rel_path,
                        "node": node,
                        "violation_type": "UNDECLARED_TRANSPORT",
                        "severity": "warning",
                        "line": ast_node.lineno,
                        "message": f"Transport import: from {ast_node.module}",
                    }
                )
    return findings


def _check_logic_in_node(
    repo: str, rel_path: str, node: str, lines: list[str]
) -> list[dict]:
    findings = []
    for i, line in enumerate(lines, 1):
        for pattern in _LOGIC_INDICATORS:
            if pattern.search(line):
                findings.append(
                    {
                        "repo": repo,
                        "file": rel_path,
                        "node": node,
                        "violation_type": "LOGIC_IN_NODE",
                        "severity": "warning",
                        "line": i,
                        "message": f"Business logic in node file: {line.strip()[:80]}",
                    }
                )
    return findings


def sweep_repo(repo_root: Path, checks: list[str]) -> tuple[int, list[dict]]:
    """Scan a single repo root. Returns (handlers_scanned, findings)."""
    repo_name = repo_root.name
    handler_files = _find_handler_files(repo_root)
    findings: list[dict] = []

    for handler_file in handler_files:
        node_name = _infer_node_name(handler_file, repo_root)
        rel_path = str(handler_file.relative_to(repo_root))
        lines = _read_lines(handler_file)

        if "hardcoded-topics" in checks:
            findings.extend(
                _check_hardcoded_topics(repo_name, rel_path, node_name, lines)
            )
        if "undeclared-transport" in checks:
            findings.extend(
                _check_transport_imports(repo_name, rel_path, node_name, handler_file)
            )
        if "logic-in-node" in checks and (
            "node.py" in handler_file.name or handler_file.name == "__init__.py"
        ):
            findings.extend(_check_logic_in_node(repo_name, rel_path, node_name, lines))

    return len(handler_files), findings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Handler contract compliance sweep — standalone CI gate"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--repo",
        metavar="PATH",
        help="Single repo root to scan",
    )
    group.add_argument(
        "--repos",
        metavar="PATH[,PATH...]",
        help="Comma-separated repo roots to scan",
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

    # Resolve repo paths
    if args.repo:
        repo_paths = [Path(args.repo)]
    elif args.repos:
        repo_paths = [Path(p.strip()) for p in args.repos.split(",") if p.strip()]
    else:
        repo_paths = [Path.cwd()]

    checks = [c.strip() for c in args.checks.split(",") if c.strip()]
    fail_threshold = _SEVERITY_ORDER[args.fail_on_severity]

    all_findings: list[dict] = []
    total_handlers = 0
    skipped_repos: list[str] = []

    for repo_path in repo_paths:
        if not repo_path.is_dir():
            skipped_repos.append(str(repo_path))
            continue
        scanned, findings = sweep_repo(repo_path, checks)
        total_handlers += scanned
        all_findings.extend(findings)

    # Count by severity
    severity_counts: dict[str, int] = dict.fromkeys(_SEVERITY_ORDER, 0)
    for f in all_findings:
        sev = f["severity"].lower()
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    blocking_findings = [
        f
        for f in all_findings
        if _SEVERITY_ORDER.get(f["severity"].lower(), 0) >= fail_threshold
    ]

    if args.json:
        output = {
            "sweep": "compliance_sweep",
            "repos_scanned": [str(p) for p in repo_paths if p.is_dir()],
            "repos_skipped": skipped_repos,
            "handlers_scanned": total_handlers,
            "checks": checks,
            "fail_on_severity": args.fail_on_severity,
            "findings": all_findings,
            "summary": {
                "total": len(all_findings),
                "by_severity": severity_counts,
                "blocking": len(blocking_findings),
            },
            "status": "FAIL" if blocking_findings else "PASS",
        }
        print(json.dumps(output, indent=2))
    else:
        print("Compliance Sweep Results")
        print("========================")
        print(f"Repos scanned: {len([p for p in repo_paths if p.is_dir()])}")
        if skipped_repos:
            print(f"Repos not found: {', '.join(skipped_repos)}")
        print(f"Handlers scanned: {total_handlers}")
        print(f"Checks: {', '.join(checks)}")
        print()

        if all_findings:
            # Group by type for summary
            by_type: dict[str, int] = {}
            for f in all_findings:
                by_type[f["violation_type"]] = by_type.get(f["violation_type"], 0) + 1
            print("Violations by type:")
            for vtype, count in sorted(by_type.items()):
                print(f"  {vtype}: {count}")
            print()
            print("Findings:")
            for f in all_findings:
                sev_marker = "ERROR" if f["severity"] == "error" else "WARN "
                print(
                    f"  [{sev_marker}] {f['repo']}/{f['file']}:{f['line']} "
                    f"[{f['violation_type']}] {f['message']}"
                )
        else:
            print("No violations found.")

        print()
        status = "FAIL" if blocking_findings else "PASS"
        print(
            f"Overall: {status} "
            f"({len(all_findings)} findings, {len(blocking_findings)} blocking)"
        )

    return 1 if blocking_findings else 0


if __name__ == "__main__":
    sys.exit(main())
