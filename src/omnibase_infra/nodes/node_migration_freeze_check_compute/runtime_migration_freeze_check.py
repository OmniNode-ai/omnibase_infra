# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""CLI runtime for the migration-freeze check: pre-commit hook and CI entrypoint.

This module is the EFFECT boundary of the node: it reads ``.migration_freeze``,
today's date and the diff's added paths, hands them to the pure
``NodeMigrationFreezeCheckCompute`` handler and prints the report.

Usage::

    python -m omnibase_infra.nodes.node_migration_freeze_check_compute.runtime_migration_freeze_check
    python -m omnibase_infra.nodes.node_migration_freeze_check_compute.runtime_migration_freeze_check --base origin/dev

Without ``--base`` the staged files are inspected (pre-commit). With ``--base
<ref>`` the files added or renamed by ``<ref>...HEAD`` are inspected (CI).

Exit codes: 0 for PASS or WARN (a warning never fails), 1 for FAIL, 2 when the
check could not run (git failed).

Ticket: OMN-20568.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from omnibase_core.models.validation.model_validation_report import (
    ModelValidationReport,
)
from omnibase_infra.nodes.node_migration_freeze_check_compute.handler import (
    FREEZE_EXPIRE_DAYS,
    NodeMigrationFreezeCheckCompute,
)
from omnibase_infra.nodes.node_migration_freeze_check_compute.models.model_migration_freeze_check_input import (
    ModelMigrationFreezeCheckInput,
)

__all__ = ["main"]

_FREEZE_FILE = ".migration_freeze"


def _write(text: str) -> None:
    sys.stdout.write(text + "\n")


def _added_paths(base: str | None) -> tuple[str, ...]:
    """Added or renamed (destination) paths of the diff under inspection."""
    args = ["git", "diff", "--name-only", "--diff-filter=AR"]
    args += [f"{base}...HEAD"] if base else ["--cached"]
    completed = subprocess.run(
        args, capture_output=True, text=True, timeout=30, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed: {completed.stderr.strip()}")
    return tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())


def _render(report: ModelValidationReport) -> str:
    lines: list[str] = []
    for finding in report.findings:
        if finding.rule_id == "freeze-expired":
            lines += ["ERROR: " + finding.message]
        elif finding.rule_id == "freeze-expiring":
            lines += ["WARNING: " + finding.message]
        elif finding.rule_id == "new-migration-file":
            lines += [f"  {finding.message}"]
    if report.overall_status == "FAIL" and any(
        f.rule_id == "new-migration-file" for f in report.findings
    ):
        lines += [
            "",
            f"Schema migrations are FROZEN (see {_FREEZE_FILE}).",
            "ALLOWED during freeze: modifying existing migration files, deleting them.",
            "NOT ALLOWED during freeze: adding NEW migration files (A) or renames (R).",
            f"To lift the freeze, remove {_FREEZE_FILE} and reference OMN-2055.",
        ]
    if any(f.rule_id == "freeze-expired" for f in report.findings):
        lines += [
            f"Freezes are invalidated after {FREEZE_EXPIRE_DAYS} days: lift the freeze "
            f"(remove {_FREEZE_FILE}) or renew freeze_date= with a justification.",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="migration-freeze-check",
        description="Block new migration files while .migration_freeze exists (OMN-20568).",
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Inspect <base>...HEAD (CI) instead of the staged files (pre-commit).",
    )
    parsed = parser.parse_args(argv)

    freeze = Path(_FREEZE_FILE)
    if not freeze.is_file():
        _write("Migration Freeze: inactive (no .migration_freeze file)")
        return 0

    try:
        added = _added_paths(parsed.base)
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2

    report = NodeMigrationFreezeCheckCompute().handle(
        ModelMigrationFreezeCheckInput(
            freeze_active=True,
            freeze_text=freeze.read_text(encoding="utf-8"),
            today=datetime.now(tz=UTC).date(),
            added_paths=added,
        )
    )
    rendered = _render(report)
    if rendered:
        _write(rendered)
    if report.overall_status in ("PASS", "WARN"):
        _write(f"Migration Freeze: {report.overall_status} (freeze active)")
        return 0
    _write("Migration Freeze: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
