#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Emit a Repowise workspace freshness receipt.

The receipt binds Repowise index metadata to live git state so tech-debt
findings can cite the repository SHAs and freshness state that produced them.
It is stdlib-only because the sync wrapper runs outside project venvs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _run(cmd: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_branch(repo_dir: Path) -> str | None:
    return _run(["git", "branch", "--show-current"], repo_dir)


def _git_head(repo_dir: Path) -> str | None:
    return _run(["git", "rev-parse", "HEAD"], repo_dir)


def _strip_yaml_scalar(value: str) -> str | None:
    value = value.strip().strip("'\"")
    if value in {"", "null", "Null", "NULL", "~"}:
        return None
    return value


def _load_workspace_repos(omni_home: Path) -> list[dict[str, Any]]:
    workspace_path = omni_home / ".repowise-workspace.yaml"
    if not workspace_path.exists():
        raise FileNotFoundError(f"missing workspace config: {workspace_path}")

    repos: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    in_repos = False

    for raw_line in workspace_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "repos:":
            in_repos = True
            continue
        if not in_repos:
            continue
        if line and not line.startswith(" ") and not stripped.startswith("- "):
            break
        # Any "- " list item starts a new repo record (order-tolerant).
        if stripped.startswith("- "):
            if current is not None:
                repos.append(current)
            current = {}
            # The first key may be inline with the list marker (e.g. "- path: foo").
            rest = stripped[2:].strip()
            if rest and ":" in rest:
                key, value = rest.split(":", 1)
                current[key.strip()] = _strip_yaml_scalar(value)
            continue
        if current is None or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        current[key.strip()] = _strip_yaml_scalar(value)

    if current is not None:
        repos.append(current)
    # Discard entries that lack a path field (malformed YAML blocks).
    return [r for r in repos if r.get("path") is not None]


def _age_days(indexed_at: str | None, now: datetime) -> float | None:
    if indexed_at is None:
        return None
    try:
        parsed = datetime.fromisoformat(indexed_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return (now - parsed).total_seconds() / 86400.0


def _docs_mode(entry: dict[str, Any]) -> str:
    docs_mode = entry.get("docs_mode") or entry.get("docs")
    if docs_mode is not None:
        return str(docs_mode)
    if entry.get("indexed_at") is None:
        return "none"
    return "generated"


def emit(omni_home: Path, out_path: Path | None = None) -> dict[str, Any]:
    now = datetime.now(UTC)
    run_id = f"freshness-{now.strftime('%Y%m%dT%H%M%SZ')}"
    records: list[dict[str, Any]] = []
    failure_summaries: list[str] = []

    for entry in _load_workspace_repos(omni_home):
        rel_path = str(entry.get("path") or "")
        alias = str(entry.get("alias") or rel_path or "omni_home")
        repo_dir = omni_home if rel_path in {"", "."} else omni_home / rel_path
        exists = repo_dir.is_dir()
        branch = _git_branch(repo_dir) if exists else None
        head_sha = _git_head(repo_dir) if exists else None

        indexed_at = entry.get("indexed_at")
        index_head_sha = entry.get("last_commit_at_index") or entry.get(
            "index_head_sha"
        )
        index_age_days = _age_days(indexed_at, now)
        no_index = indexed_at is None
        stale = bool(head_sha and index_head_sha and head_sha != index_head_sha)
        # A repo that exists and was indexed but has an unreadable HEAD is an
        # unknown state — flag it explicitly rather than letting it appear fresh.
        head_unreadable = exists and not no_index and head_sha is None

        failure = None
        if not exists:
            failure = "repo directory not found"
        elif no_index:
            failure = "never indexed by Repowise"
        elif head_unreadable:
            failure = "repo exists but git HEAD is unreadable (not a git repo?)"
        elif index_age_days is None:
            failure = f"unparseable indexed_at: {indexed_at}"
        elif stale:
            failure = f"HEAD {head_sha[:9]} != index SHA {str(index_head_sha)[:9]}"

        if failure is not None:
            failure_summaries.append(f"{alias}: {failure}")

        records.append(
            {
                "alias": alias,
                "path": rel_path,
                "exists": exists,
                "branch": branch,
                "head_sha": head_sha,
                "indexed_at": indexed_at,
                "index_age_days": (
                    round(index_age_days, 2) if index_age_days is not None else None
                ),
                "index_head_sha": index_head_sha,
                "stale": stale,
                "no_index": no_index,
                "docs_mode": _docs_mode(entry),
                "failure": failure,
            }
        )

    receipt = {
        "run_id": run_id,
        "generated_at": now.isoformat(),
        "omni_home": str(omni_home),
        "repos": records,
        "summary": {
            "total": len(records),
            "indexed": sum(1 for record in records if not record["no_index"]),
            "stale": sum(1 for record in records if record["stale"]),
            "no_index": sum(1 for record in records if record["no_index"]),
            "failures": sum(1 for record in records if record["failure"] is not None),
        },
        "failure_summaries": failure_summaries,
    }

    if out_path is None:
        out_path = omni_home / ".onex_state" / "repowise-sync" / f"{run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    latest_path = out_path.parent / "latest-freshness.json"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    try:
        latest_path.symlink_to(out_path.name)
    except OSError:
        # Some CI/workspace filesystems disallow symlinks; keep latest durable.
        latest_path.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
    return receipt


def _print_summary(receipt: dict[str, Any], out_path: Path) -> None:
    summary = receipt["summary"]
    print(f"Repowise freshness receipt: {receipt['run_id']}")
    print(f"  total: {summary['total']}")
    print(f"  indexed: {summary['indexed']}")
    print(f"  stale: {summary['stale']}")
    print(f"  no_index: {summary['no_index']}")
    print(f"  failures: {summary['failures']}")
    print(f"  receipt: {out_path}")
    for failure in receipt["failure_summaries"]:
        print(f"  - {failure}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit Repowise freshness receipt")
    parser.add_argument(
        "--omni-home",
        default=os.environ.get("OMNI_HOME"),
        help="Path to omni_home. Defaults to $OMNI_HOME, then current directory.",
    )
    parser.add_argument("--out", help="Output receipt path.")
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()

    omni_home = Path(args.omni_home or ".").expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve() if args.out else None
    try:
        receipt = emit(omni_home=omni_home, out_path=out_path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    default_out_path = (
        omni_home / ".onex_state" / "repowise-sync" / f"{receipt['run_id']}.json"
    )
    receipt_path = out_path or default_out_path
    if args.json_only:
        print(receipt_path)
    else:
        _print_summary(receipt, receipt_path)


if __name__ == "__main__":
    main()
