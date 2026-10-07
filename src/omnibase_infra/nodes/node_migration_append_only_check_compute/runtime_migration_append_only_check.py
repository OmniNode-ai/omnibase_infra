# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Git/file EFFECT boundary and CLI for the append-only COMPUTE check.

    uv run python -m omnibase_infra.nodes.node_migration_append_only_check_compute.runtime_migration_append_only_check --staged
    uv run python -m omnibase_infra.nodes.node_migration_append_only_check_compute.runtime_migration_append_only_check --base origin/dev

Exit codes and text match scripts/validation/check_migration_append_only.py:
0 for PASS, 1 for migration violations, 2 for an unavailable/invalid check.
Committed mode intentionally reads the working-tree ledgers and successor files,
as the original script does; staged mode reads the index.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Final

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.nodes.node_migration_append_only_check_compute.handler import (
    FORWARD_PREFIX,
    MANIFEST_REPO_PATH,
    SUPERSESSIONS_REPO_PATH,
    AppendOnlyViolationError,
    NodeMigrationAppendOnlyCheckCompute,
    validate_base_manifest,
)
from omnibase_infra.nodes.node_migration_append_only_check_compute.models import (
    ModelMigrationAppendOnlyCheckInput,
)

DEFAULT_INTEGRATION_REF: Final[str] = "origin/dev"
_FAILURE_TEXT: Final[str] = (
    "FAIL: applied migration history was rewritten (OMN-16705). "
    "The forward-migration runner records a content_sha256 per applied "
    "migration in platform_catalog.schema_migrations and refuses every "
    "later run when the file no longer matches:"
)


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        env=scrub_git_location_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AppendOnlyViolationError(
            f"git {' '.join(args)} failed ({result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def _git_show(repo_root: Path, ref: str, repo_path: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{ref}:{repo_path}"],
        env=scrub_git_location_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def _git_show_index(repo_root: Path, repo_path: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "show", f":{repo_path}"],
        env=scrub_git_location_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def _read_worktree_file(repo_root: Path, repo_path: str) -> str | None:
    target = repo_root / repo_path
    return target.read_text(encoding="utf-8") if target.is_file() else None


def _repo_path_exists(repo_root: Path, repo_path: str, *, staged: bool) -> bool:
    if not staged:
        return (repo_root / repo_path).is_file()
    result = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "-e", f":{repo_path}"],
        env=scrub_git_location_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _changed_paths(diff_output: str) -> dict[str, str]:
    """Map renames/copies to an old-path mutation and a destination addition."""
    changed: dict[str, str] = {}
    for raw in diff_output.splitlines():
        if not raw.strip():
            continue
        fields = raw.split("\t")
        status = fields[0][:1]
        if status in {"R", "C"} and len(fields) >= 3:
            changed[fields[1]] = status
            changed[fields[2]] = "A"
        elif len(fields) >= 2:
            changed[fields[1]] = status
    return changed


def _resolve_staged_base(repo_root: Path, base: str | None) -> str:
    """Use the integration merge-base, never HEAD as a fallback."""
    candidate = base or DEFAULT_INTEGRATION_REF
    result = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", candidate, "HEAD"],
        env=scrub_git_location_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    raise AppendOnlyViolationError(
        f"could not resolve integration base {candidate!r}; fetch the integration "
        "reference or pass --base explicitly before running the append-only guard"
    )


def collect_request(
    repo_root: Path, *, base: str | None, staged: bool
) -> ModelMigrationAppendOnlyCheckInput:
    """Collect the same git/file inputs check() used, with its error ordering."""
    if staged:
        base_ref = _resolve_staged_base(repo_root, base)
        diff_output = _git(repo_root, "diff", "--cached", "--name-status", base_ref)
    else:
        if base is None:
            raise AppendOnlyViolationError(
                "a base ref is required outside --staged mode"
            )
        base_ref = _git(repo_root, "merge-base", base, "HEAD").strip()
        diff_output = _git(repo_root, "diff", "--name-status", base_ref, "HEAD")
    changed = _changed_paths(diff_output)
    base_manifest_text = _git_show(repo_root, base_ref, MANIFEST_REPO_PATH)
    validate_base_manifest(base_manifest_text, base_ref)
    manifest_text = (
        _git_show_index(repo_root, MANIFEST_REPO_PATH)
        if staged
        else (repo_root / MANIFEST_REPO_PATH).read_text(encoding="utf-8")
    )
    supersessions_text = _git_show_index(repo_root, SUPERSESSIONS_REPO_PATH)
    if not staged:
        supersessions_text = _read_worktree_file(repo_root, SUPERSESSIONS_REPO_PATH)
    existing_paths = frozenset(
        path
        for path, status in changed.items()
        if status == "A"
        and path.startswith(FORWARD_PREFIX)
        and _repo_path_exists(repo_root, path, staged=staged)
    )
    return ModelMigrationAppendOnlyCheckInput(
        base_ref=base_ref,
        changed_paths=tuple(changed.items()),
        base_manifest_text=base_manifest_text,
        base_supersessions_text=_git_show(repo_root, base_ref, SUPERSESSIONS_REPO_PATH),
        manifest_text=manifest_text,
        supersessions_text=supersessions_text,
        existing_paths=existing_paths,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Append-only enforcement for declared node migrations (OMN-16705)."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="repository root (default: the current directory, the repo root under pre-commit and CI)",
    )
    parser.add_argument(
        "--base", default=None, help="base ref to diff against, e.g. origin/dev"
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="check the staged index against HEAD (pre-commit mode)",
    )
    args = parser.parse_args(argv)
    try:
        report = NodeMigrationAppendOnlyCheckCompute().handle(
            collect_request(args.repo_root, base=args.base, staged=args.staged)
        )
    except AppendOnlyViolationError as exc:
        sys.stderr.write(f"FAIL: {exc}\n")
        return 2
    if report.overall_status == "FAIL":
        sys.stderr.write(_FAILURE_TEXT + "\n")
        for finding in report.findings:
            sys.stderr.write(f"  - {finding.message}\n")
        return 1
    sys.stdout.write(
        "PASS: no declared node migration was modified, deleted, or renamed.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
