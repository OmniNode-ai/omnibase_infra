#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""
container-error-to-linear.py — Scan Docker containers for persistent errors
and optionally create Linear tickets for them.

Usage:
    uv run python scripts/container-error-to-linear.py [--execute] [--team-id TEAM_ID]

Modes:
    (default)   Dry-run: scan containers, print findings, create no tickets.
    --execute   Create Linear tickets via the Linear MCP server for persistent errors.
    --team-id   Linear team ID (required for --execute).

Container selection:
    Scans all containers in the omnibase-infra Docker compose project.
    Filters for containers that are in restart loop or have repeated error log lines.

Error detection heuristics:
    1. Container restart count > 3 (restart loop)
    2. OOMKilled flag set
    3. Exit code != 0 with repeated restarts
    4. Repeated ERROR/FATAL log lines in the last 100 log lines

Output:
    JSON array of detected issues, each with:
      - container_name
      - error_type: restart_loop | oom_killed | crash_loop | log_errors
      - details: human-readable description
      - log_sample: last relevant log lines (truncated)
      - suggested_ticket_title
      - suggested_ticket_body

[OMN-5150]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass


@dataclass
class ContainerIssue:
    container_name: str
    error_type: str  # restart_loop | oom_killed | crash_loop | log_errors
    details: str
    log_sample: list[str]
    suggested_ticket_title: str
    suggested_ticket_body: str


def run_cmd(cmd: list[str], *, timeout: int = 30) -> str:
    """Run a command and return stdout. Returns empty string on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def get_infra_containers() -> list[dict]:
    """Get all containers from the omnibase-infra project."""
    output = run_cmd(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            "label=com.docker.compose.project=omnibase-infra",
            "--format",
            "{{.Names}}",
        ]
    )
    if not output:
        return []

    containers = []
    for name in output.splitlines():
        name = name.strip()
        if not name:
            continue
        inspect_json = run_cmd(["docker", "inspect", name])
        if inspect_json:
            try:
                data = json.loads(inspect_json)
                if data:
                    containers.append(data[0])
            except json.JSONDecodeError:
                pass
    return containers


def get_container_logs(name: str, tail: int = 100) -> list[str]:
    """Get the last N log lines from a container."""
    output = run_cmd(["docker", "logs", "--tail", str(tail), name])
    return output.splitlines() if output else []


def detect_issues(containers: list[dict]) -> list[ContainerIssue]:
    """Scan containers for persistent error conditions."""
    issues: list[ContainerIssue] = []

    for container in containers:
        name = container.get("Name", "").lstrip("/")
        state = container.get("State", {})
        restart_count = state.get("RestartCount", 0)
        oom_killed = state.get("OOMKilled", False)
        exit_code = state.get("ExitCode", 0)
        status = state.get("Status", "")

        # Check for OOM kills
        if oom_killed:
            logs = get_container_logs(name, tail=50)
            issues.append(
                ContainerIssue(
                    container_name=name,
                    error_type="oom_killed",
                    details=f"Container was OOM killed. Restart count: {restart_count}.",
                    log_sample=logs[-20:] if logs else [],
                    suggested_ticket_title=f"fix(infra): {name} OOMKilled — increase memory limit",
                    suggested_ticket_body=(
                        f"## Problem\n"
                        f"Container `{name}` was OOM killed.\n"
                        f"- Restart count: {restart_count}\n"
                        f"- Status: {status}\n\n"
                        f"## Suggested Fix\n"
                        f"Increase memory limit in docker-compose.yml or optimize memory usage.\n\n"
                        f"## Log Sample\n```\n{''.join(ln + chr(10) for ln in logs[-10:])}\n```"
                    ),
                )
            )
            continue

        # Check for restart loops
        if restart_count > 3:
            logs = get_container_logs(name, tail=50)
            issues.append(
                ContainerIssue(
                    container_name=name,
                    error_type="restart_loop",
                    details=f"Container in restart loop. Restart count: {restart_count}, exit code: {exit_code}.",
                    log_sample=logs[-20:] if logs else [],
                    suggested_ticket_title=f"fix(infra): {name} restart loop (exit {exit_code}, {restart_count} restarts)",
                    suggested_ticket_body=(
                        f"## Problem\n"
                        f"Container `{name}` is in a restart loop.\n"
                        f"- Restart count: {restart_count}\n"
                        f"- Exit code: {exit_code}\n"
                        f"- Status: {status}\n\n"
                        f"## Log Sample\n```\n{''.join(ln + chr(10) for ln in logs[-10:])}\n```"
                    ),
                )
            )
            continue

        # Check for crash loop (exited with non-zero, not running)
        if status != "running" and exit_code != 0 and restart_count > 0:
            logs = get_container_logs(name, tail=50)
            issues.append(
                ContainerIssue(
                    container_name=name,
                    error_type="crash_loop",
                    details=f"Container exited with code {exit_code}. Restart count: {restart_count}.",
                    log_sample=logs[-20:] if logs else [],
                    suggested_ticket_title=f"fix(infra): {name} crash loop (exit {exit_code})",
                    suggested_ticket_body=(
                        f"## Problem\n"
                        f"Container `{name}` is crashing.\n"
                        f"- Exit code: {exit_code}\n"
                        f"- Restart count: {restart_count}\n"
                        f"- Status: {status}\n\n"
                        f"## Log Sample\n```\n{''.join(ln + chr(10) for ln in logs[-10:])}\n```"
                    ),
                )
            )
            continue

        # Check running containers for repeated errors in logs
        if status == "running":
            logs = get_container_logs(name, tail=100)
            error_lines = [
                line
                for line in logs
                if any(
                    kw in line.upper()
                    for kw in ["ERROR", "FATAL", "PANIC", "EXCEPTION"]
                )
            ]
            # Only flag if there are many repeated errors (>5 in last 100 lines)
            if len(error_lines) > 5:
                # Deduplicate by taking unique error messages
                unique_errors = list(dict.fromkeys(error_lines))[:5]
                issues.append(
                    ContainerIssue(
                        container_name=name,
                        error_type="log_errors",
                        details=f"Container has {len(error_lines)} error lines in last 100 log lines.",
                        log_sample=unique_errors,
                        suggested_ticket_title=f"fix(infra): {name} persistent log errors ({len(error_lines)} errors)",
                        suggested_ticket_body=(
                            f"## Problem\n"
                            f"Container `{name}` is running but logging persistent errors.\n"
                            f"- Error lines in last 100 log lines: {len(error_lines)}\n"
                            f"- Unique error patterns: {len(unique_errors)}\n\n"
                            f"## Sample Errors\n```\n{''.join(ln + chr(10) for ln in unique_errors)}\n```"
                        ),
                    )
                )

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan containers for persistent errors"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Create Linear tickets (default: dry-run)",
    )
    parser.add_argument(
        "--team-id", type=str, help="Linear team ID for ticket creation"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.execute and not args.team_id:
        print("ERROR: --team-id is required when using --execute", file=sys.stderr)
        sys.exit(1)

    print("[container-error-to-linear] Scanning Docker containers...", file=sys.stderr)
    containers = get_infra_containers()
    if not containers:
        print(
            "[container-error-to-linear] No omnibase-infra containers found.",
            file=sys.stderr,
        )
        if args.json:
            print("[]")
        sys.exit(0)

    print(
        f"[container-error-to-linear] Found {len(containers)} containers.",
        file=sys.stderr,
    )

    issues = detect_issues(containers)

    if args.json:
        print(json.dumps([asdict(i) for i in issues], indent=2))
    elif not issues:
        print("[container-error-to-linear] No persistent errors detected.")
    else:
        print(f"\n[container-error-to-linear] Found {len(issues)} issue(s):\n")
        for issue in issues:
            print(f"  [{issue.error_type}] {issue.container_name}")
            print(f"    {issue.details}")
            print(f"    Ticket: {issue.suggested_ticket_title}")
            print()

    if args.execute and issues:
        print(
            "[container-error-to-linear] --execute: Linear ticket creation not yet implemented.",
            file=sys.stderr,
        )
        print(
            "[container-error-to-linear] Ticket data is available in --json output for manual creation.",
            file=sys.stderr,
        )
        # Future: integrate with Linear MCP tools or API
        # For now, dry-run outputs the ticket data so it can be piped to the Linear MCP


if __name__ == "__main__":
    main()
