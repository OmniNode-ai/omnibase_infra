#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Append-only enforcement for declared node migrations (OMN-16705).

## The class this closes

``docker/migrations/forward/_ledger/bootstrap.sql`` records a
``content_sha256`` for every node migration it applies and then, on every
subsequent run, refuses to continue when a recorded checksum no longer matches
the file on disk::

    ERROR:  conflicting migration checksum in canonical node history

That predicate is correct and fail-closed. It means a single in-place edit to an
already-applied migration file permanently bricks forward-migration on every
database that applied the old bytes, and the damage is discovered only at the
next deploy -- by whoever happens to deploy next, not by whoever caused it.

It has happened three times on one pair of files:

* ``88f4ac346`` (2026-08-22 20:44) rewrote
  ``node:node_delegation_routing_reducer:0001`` 35 minutes AFTER the .201 dev
  lane applied it.
* ``7de798a4a`` (2026-08-24 11:44, OMN-16450 / #2866) rewrote the same file a
  second time, and added 46 lines plus five ``SET NOT NULL`` to
  ``node:node_projection_tenant_credentials:0000``.

The dev lane's ``forward-migration`` one-shot then exited 3 on every run
(OMN-16705). The existing gates did not catch it: the manifest validator checks
that the DECLARED checksum matches the CURRENT file (it does -- both were
updated together), and the OMN-15376 shape gates check SHAPE. Nothing checked
IMMUTABILITY.

## What this asserts

For every file changed by the diff under test: if its manifest path was already
declared in ``_ledger/application-migrations.tsv`` AT THE BASE REF, it may not be
modified, deleted, or renamed.

The base ref is the INTEGRATION BRANCH in both modes -- ``--base`` in CI, and
the merge-base with ``origin/dev`` in ``--staged`` (pre-commit) mode. It is
deliberately NOT ``HEAD``: a migration introduced by the first commit of a
branch must still be amendable by its second, or the rule would be commit-only
rather than append-only.

The only escape is an explicit new-ordinal supersession, declared in
``_ledger/migration-supersessions.tsv`` (TSV, 4 columns:
``artifact_path``, ``superseded_by``, ``ticket``, ``reason``). A row permits the
change only when all of the following hold:

* both paths are ``nodes/<node>/<file>.sql`` under the SAME node,
* ``superseded_by``'s leading ordinal is strictly greater than
  ``artifact_path``'s,
* ``superseded_by`` is ADDED by this very diff -- so an applied migration can
  only be touched in the same change that lands its successor, and the row
  cannot be re-used later to authorise a second edit,
* ``superseded_by`` exists on disk and is declared in the manifest,
* the ticket is a real ``OMN-NNNN`` reference.

## Scope, stated rather than implied

This guards the paths the canonical ledger actually checksums: the node
migrations declared in ``application-migrations.tsv``. The flat migrations in
``docker/migrations/forward/*.sql`` are NOT covered, because
``onex_application_migration_manifest`` -- the relation bootstrap.sql joins to
raise the conflict above -- is node-only, so editing a flat migration cannot
produce this failure. That is a deliberate scope boundary, not an oversight.

Ticket: OMN-16705
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

FORWARD_PREFIX = "docker/migrations/forward/"
MANIFEST_REPO_PATH = f"{FORWARD_PREFIX}_ledger/application-migrations.tsv"
SUPERSESSIONS_REPO_PATH = f"{FORWARD_PREFIX}_ledger/migration-supersessions.tsv"

_TICKET = re.compile(r"^OMN-[0-9]+$")
_ARTIFACT = re.compile(
    r"^nodes/(?P<node>[A-Za-z0-9_][A-Za-z0-9_.-]*)/(?P<file>[0-9]+[A-Za-z0-9_.-]*\.sql)$"
)
_ORDINAL = re.compile(r"^(?P<ordinal>[0-9]+)")

# Statuses that change or remove the bytes of an existing path.
_MUTATING_STATUSES = frozenset({"M", "D", "T", "R", "C"})


class AppendOnlyViolationError(Exception):
    """A declared migration's bytes changed without an authorised supersession."""


@dataclass(frozen=True, slots=True)
class Supersession:
    artifact_path: str
    superseded_by: str
    ticket: str
    reason: str


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
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
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def _git_show_index(repo_root: Path, repo_path: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "show", f":{repo_path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def _repo_path_exists(repo_root: Path, repo_path: str, *, staged: bool) -> bool:
    if not staged:
        return (repo_root / repo_path).is_file()
    result = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "-e", f":{repo_path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def declared_artifacts(manifest_text: str | None) -> frozenset[str]:
    """The ``artifact_path`` column of a manifest revision."""
    if not manifest_text:
        return frozenset()
    return frozenset(
        line.split("\t", 1)[0] for line in manifest_text.splitlines() if line.strip()
    )


def parse_supersessions(text: str | None) -> tuple[Supersession, ...]:
    if not text:
        return ()
    rows: list[Supersession] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        fields = raw.split("\t")
        if len(fields) != 4 or any(field == "" for field in fields):
            raise AppendOnlyViolationError(
                f"{SUPERSESSIONS_REPO_PATH}:{line_number}: expected 4 non-empty TSV fields"
            )
        rows.append(Supersession(*fields))
    return tuple(rows)


def _ordinal(artifact_path: str) -> int:
    match = _ARTIFACT.match(artifact_path)
    if match is None:
        raise AppendOnlyViolationError(
            f"supersession path must be nodes/<node>/<ordinal>_<name>.sql: {artifact_path!r}"
        )
    ordinal_match = _ORDINAL.match(match.group("file"))
    assert ordinal_match is not None  # guaranteed by _ARTIFACT
    return int(ordinal_match.group("ordinal"))


def _node(artifact_path: str) -> str:
    match = _ARTIFACT.match(artifact_path)
    if match is None:
        raise AppendOnlyViolationError(
            f"supersession path must be nodes/<node>/<ordinal>_<name>.sql: {artifact_path!r}"
        )
    return match.group("node")


def _changed_paths(diff_output: str) -> dict[str, str]:
    """Map repo-relative path -> single-letter status, from ``--name-status``.

    A rename reports the OLD path as changed (its bytes stop existing at that
    path, which is exactly what the ledger cannot tolerate) and the NEW path as
    an addition.
    """
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


DEFAULT_INTEGRATION_REF = "origin/dev"


def _resolve_staged_base(repo_root: Path, base: str | None) -> str:
    """Merge-base of the index's branch with the integration branch.

    Fails loudly when the integration ref is unavailable. Falling back to
    ``HEAD`` changes the predicate from branch-append-only to commit-only and
    can freeze a migration added by an earlier branch commit.
    """
    candidate = base or DEFAULT_INTEGRATION_REF
    result = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", candidate, "HEAD"],
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


def _authorised(
    artifact_path: str,
    supersessions: tuple[Supersession, ...],
    added_artifacts: frozenset[str],
    current_manifest: frozenset[str],
    repo_root: Path,
    *,
    staged: bool,
) -> str | None:
    """Return the failure reason, or ``None`` when the change is authorised."""
    rows = [row for row in supersessions if row.artifact_path == artifact_path]
    if not rows:
        return (
            "no supersession row in "
            f"{SUPERSESSIONS_REPO_PATH}. An already-declared migration is applied "
            "history and its bytes are frozen: add a NEW file with the next "
            "ordinal in the same node directory instead of editing this one."
        )
    problems: list[str] = []
    for row in rows:
        if _TICKET.fullmatch(row.ticket) is None:
            problems.append(f"invalid ticket {row.ticket!r}")
            continue
        if _node(row.superseded_by) != _node(artifact_path):
            problems.append(f"{row.superseded_by} is not in the same node directory")
            continue
        if _ordinal(row.superseded_by) <= _ordinal(artifact_path):
            problems.append(
                f"{row.superseded_by} does not carry a higher ordinal than "
                f"{artifact_path}"
            )
            continue
        if row.superseded_by not in added_artifacts:
            problems.append(
                f"{row.superseded_by} is not ADDED by this change; a supersession "
                "only authorises the change that lands the successor, it is not a "
                "standing waiver"
            )
            continue
        successor_repo_path = f"{FORWARD_PREFIX}{row.superseded_by}"
        if not _repo_path_exists(repo_root, successor_repo_path, staged=staged):
            problems.append(f"{row.superseded_by} does not exist on disk")
            continue
        if row.superseded_by not in current_manifest:
            problems.append(
                f"{row.superseded_by} is not declared in {MANIFEST_REPO_PATH}"
            )
            continue
        return None
    return "; ".join(problems)


def check(repo_root: Path, *, base: str | None, staged: bool) -> list[str]:
    """Return a list of human-readable violations (empty means pass)."""
    if staged:
        # The base is the INTEGRATION BRANCH, never HEAD. Diffing the index
        # against HEAD would freeze a migration the moment its own first commit
        # landed on the branch, so the second commit of the very PR that
        # introduces it could not amend it -- which is not append-only, it is
        # commit-only. Learned by execution: that is exactly what this guard did
        # to its own repair PR's follow-up commit.
        base_ref = _resolve_staged_base(repo_root, base)
        diff_output = _git(repo_root, "diff", "--cached", "--name-status", base_ref)
    else:
        if base is None:
            raise AppendOnlyViolationError(
                "a base ref is required outside --staged mode"
            )
        merge_base = _git(repo_root, "merge-base", base, "HEAD").strip()
        diff_output = _git(repo_root, "diff", "--name-status", merge_base, "HEAD")
        base_ref = merge_base

    changed = _changed_paths(diff_output)
    base_declared = declared_artifacts(
        _git_show(repo_root, base_ref, MANIFEST_REPO_PATH)
    )
    if not base_declared:
        raise AppendOnlyViolationError(
            f"anti-vacuity: {MANIFEST_REPO_PATH} at {base_ref} declared no migrations; "
            "the guard would pass everything"
        )

    manifest_text = (
        _git_show_index(repo_root, MANIFEST_REPO_PATH)
        if staged
        else (repo_root / MANIFEST_REPO_PATH).read_text(encoding="utf-8")
    )
    current_manifest = declared_artifacts(manifest_text)
    supersessions_text = _git_show_index(repo_root, SUPERSESSIONS_REPO_PATH)
    if not staged:
        supersessions_path = repo_root / SUPERSESSIONS_REPO_PATH
        supersessions_text = (
            supersessions_path.read_text(encoding="utf-8")
            if supersessions_path.is_file()
            else None
        )
    supersessions = parse_supersessions(supersessions_text)

    added_artifacts = frozenset(
        path[len(FORWARD_PREFIX) :]
        for path, status in changed.items()
        if status == "A" and path.startswith(FORWARD_PREFIX)
    )

    violations: list[str] = []
    for path, status in sorted(changed.items()):
        if status not in _MUTATING_STATUSES or not path.startswith(FORWARD_PREFIX):
            continue
        artifact_path = path[len(FORWARD_PREFIX) :]
        if artifact_path not in base_declared:
            continue
        reason = _authorised(
            artifact_path,
            supersessions,
            added_artifacts,
            current_manifest,
            repo_root,
            staged=staged,
        )
        if reason is not None:
            violations.append(f"{path} ({status}): {reason}")
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (default: this script's repo)",
    )
    parser.add_argument(
        "--base",
        default=None,
        help="base ref to diff against, e.g. origin/dev",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="check the staged index against HEAD (pre-commit mode)",
    )
    args = parser.parse_args(argv)

    try:
        violations = check(args.repo_root, base=args.base, staged=args.staged)
    except AppendOnlyViolationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    if violations:
        print(
            "FAIL: applied migration history was rewritten (OMN-16705). "
            "bootstrap.sql records a content_sha256 per applied migration and "
            "refuses every later run when the file no longer matches:",
            file=sys.stderr,
        )
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        return 1

    print("PASS: no declared node migration was modified, deleted, or renamed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
