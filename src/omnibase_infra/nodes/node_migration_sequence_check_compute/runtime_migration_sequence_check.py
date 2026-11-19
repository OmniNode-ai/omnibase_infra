# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""I/O boundary and CLI for the migration-sequence COMPUTE check (OMN-20568).

With no arguments, inspect the current repository's staged paths and migration
directories. An optional repository path and --verbose match the old script's
CLI. Exit 0 for PASS, 1 for FAIL, 2 when the check cannot run and 128 when git
cannot find a repository at the path (git's own discovery status, printed with
git's own ``fatal:`` line: the environment's fault, not the change's).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omnibase_core.models.validation.model_validation_report import (
    ModelValidationReport,
)
from omnibase_infra.handlers.git_staged_paths import GitDiscoveryError, staged_paths
from omnibase_infra.nodes.node_migration_sequence_check_compute.handler import (
    NodeMigrationSequenceCheckCompute,
    migration_scan_paths,
)
from omnibase_infra.nodes.node_migration_sequence_check_compute.models import (
    ModelMigrationSequenceCheckInput,
)

__all__ = ["GitDiscoveryError", "collect_input", "main"]

GIT_DISCOVERY_EXIT = 128


def _staged_paths(repo_path: Path) -> tuple[str, ...]:
    return staged_paths(
        repo_path,
        timeout=10,
    )


def collect_input(repo_path: Path) -> ModelMigrationSequenceCheckInput:
    """Read git's staged paths and the script's nonrecursive *.sql directory scans."""
    request = ModelMigrationSequenceCheckInput(staged_paths=_staged_paths(repo_path))
    if not migration_scan_paths(request):
        return request
    migration_paths: list[str] = []
    for directory in request.migration_dirs:
        absolute = repo_path / directory
        if absolute.is_dir():
            migration_paths.extend(
                str(path.relative_to(repo_path))
                for path in sorted(absolute.glob("*.sql"))
            )
    return ModelMigrationSequenceCheckInput(
        staged_paths=request.staged_paths,
        migration_paths=tuple(migration_paths),
    )


def _render(
    report: ModelValidationReport, request: ModelMigrationSequenceCheckInput
) -> str:
    paths = migration_scan_paths(request)
    if not paths:
        return "Migration Sequence: no migration files staged — skipped"
    if report.overall_status == "PASS":
        return f"Migration Sequence: PASS ({len(paths)} file(s) scanned, no duplicates)"
    lines = [
        "ERROR: DUPLICATE MIGRATION SEQUENCE NUMBER",
        "=" * 60,
        "",
        f"Found {len(report.findings)} duplicate sequence number(s):",
        "",
        *(finding.message for finding in report.findings),
        "",
        "=" * 60,
        "Each migration file must have a unique leading sequence number.",
        "docker/ and src/ migration sets share the same namespace.",
        "",
        "To fix: renumber the new migration to use the next available sequence.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate migration sequence uniqueness"
    )
    parser.add_argument("repo_path", nargs="?", default=".")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    repo_path = Path(args.repo_path).resolve()
    try:
        request = collect_input(repo_path)
        report = NodeMigrationSequenceCheckCompute().handle(request)
        if (
            args.verbose
            or report.overall_status == "FAIL"
            or migration_scan_paths(request)
        ):
            sys.stdout.write(_render(report, request) + "\n")
        return 1 if report.overall_status == "FAIL" else 0
    except GitDiscoveryError as exc:
        sys.stderr.write(
            f"Error: git diff --cached failed (exit {exc.diff_exit}) in {repo_path}\n"
            f"{exc.detail}\n"
        )
        return GIT_DISCOVERY_EXIT
    except RuntimeError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"Unexpected error: {exc}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
