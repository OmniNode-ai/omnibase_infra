#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Deployed-migration-tree sync gate (OMN-13415).

During a stability promotion the forward-migration ran against a STALE migration
tree: ``~/.omnibase/infra/deployed/<ver>/docker/migrations/forward`` was
bind-mounted and carried an old ``0016`` with no ``0018``/``0019``. The lane
looked "deployed" while running stale migration SQL — a silent footgun. The fix
required an out-of-band rsync from the canonical clone before recreate, which is
exactly the kind of manual step a gate must replace.

This check asserts the **deployed (bind-mounted) forward-migration tree is
byte-identical to the canonical clone at the target SHA**.  A governed dirty
workspace hotpatch instead compares it to an explicitly supplied frozen source
tree, using a SHA-256 inventory of every regular file.  It is meant to run
immediately before the forward-migration phase: if the deployed tree drifted
from the clone @ target SHA (stale, missing, or extra files), the deploy ABORTS
instead of silently applying the wrong migration set.

Comparison source of truth
--------------------------
The canonical clone's git object at the target ref is authoritative. For every
file under ``<tree_rel_path>`` recorded in the clone @ ``--ref`` (via
``git ls-tree``), the deployed file must exist and its bytes must equal the bytes
from ``git show <ref>:<path>``. Files present in the deployed tree but ABSENT
from the clone @ ref are also a drift (stale leftovers from an older deploy).

Usage::

    python scripts/check_deployed_migration_tree_sync.py \\
        --deployed-tree ~/.omnibase/infra/deployed/1.2.3/docker/migrations/forward \\
        --clone-root /path/to/canonical/omnibase_infra \\
        --ref db3ae8527 \\
        --tree-rel-path docker/migrations/forward

    python scripts/check_deployed_migration_tree_sync.py \\
        --deployed-tree ~/.omnibase/infra/deployed/1.2.3/docker/migrations/forward \\
        --source-tree /frozen/hotpatch/omnibase_infra/docker/migrations/forward

Exit codes:
    0 — deployed tree is byte-identical to clone @ ref or frozen source tree
    1 — drift: a missing/stale/extra/modified migration file
    2 — configuration error (bad ref, not a git clone, missing deployed tree)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _git(clone_root: Path, args: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(clone_root), *args],
        capture_output=True,
        check=False,
    )


def _resolve_ref(clone_root: Path, ref: str) -> str:
    """Resolve a ref to a full commit SHA; raise on failure (config error)."""
    proc = _git(clone_root, ["rev-parse", "--verify", f"{ref}^{{commit}}"])
    if proc.returncode != 0:
        raise ValueError(
            f"cannot resolve ref {ref!r} in {clone_root}: "
            f"{proc.stderr.decode(errors='replace').strip()}"
        )
    return proc.stdout.decode().strip()


def _clone_files_at_ref(
    clone_root: Path, ref: str, tree_rel_path: str
) -> dict[str, str]:
    """Return {path_relative_to_tree: blob_sha} for files under tree_rel_path @ ref."""
    proc = _git(clone_root, ["ls-tree", "-r", "-z", ref, "--", tree_rel_path])
    if proc.returncode != 0:
        raise ValueError(
            f"git ls-tree failed for {tree_rel_path} @ {ref} in {clone_root}: "
            f"{proc.stderr.decode(errors='replace').strip()}"
        )
    result: dict[str, str] = {}
    prefix = tree_rel_path.rstrip("/") + "/"
    for entry in proc.stdout.decode().split("\0"):
        if not entry.strip():
            continue
        # Format: "<mode> <type> <objectsha>\t<path>"
        meta, _, path = entry.partition("\t")
        parts = meta.split()
        if len(parts) < 3 or parts[1] != "blob":
            continue
        blob_sha = parts[2]
        rel = path[len(prefix) :] if path.startswith(prefix) else path
        result[rel] = blob_sha
    return result


def _clone_blob_bytes(clone_root: Path, blob_sha: str) -> bytes:
    proc = _git(clone_root, ["cat-file", "blob", blob_sha])
    if proc.returncode != 0:
        raise ValueError(f"git cat-file failed for blob {blob_sha}")
    return proc.stdout


def _tree_file_hashes(tree: Path, *, label: str) -> dict[str, str]:
    """Return SHA-256 hashes for every regular file in a safe tree.

    A migration comparison must not follow a symlink outside the attested tree,
    nor silently omit sockets/FIFOs/devices.  The caller chooses how an invalid
    source or deployed tree becomes a configuration or drift verdict.
    """
    if not tree.is_dir() or tree.is_symlink():
        raise ValueError(f"{label} {tree} is not a real directory")

    files: dict[str, str] = {}
    for path in tree.rglob("*"):
        rel = str(path.relative_to(tree))
        if path.is_symlink():
            raise ValueError(f"{label} contains forbidden symlink: {rel}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"{label} contains non-regular entry: {rel}")
        files[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def _file_set_digest(files: dict[str, str]) -> str:
    """Make the complete source inventory auditable, including the ledger."""
    inventory = json.dumps(
        sorted(files.items()), separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(inventory).hexdigest()


def _compare(
    deployed_tree: Path, clone_root: Path, ref: str, tree_rel_path: str
) -> list[str]:
    """Return a list of drift findings (empty == in sync)."""
    findings: list[str] = []
    clone_files = _clone_files_at_ref(clone_root, ref, tree_rel_path)
    try:
        deployed = _tree_file_hashes(deployed_tree, label="deployed tree")
    except ValueError as exc:
        return [f"INVALID deployed tree: {exc}"]

    for rel, blob_sha in sorted(clone_files.items()):
        target = deployed_tree / rel
        if not target.is_file():
            findings.append(
                f"MISSING in deployed tree: {rel} (present in clone @ {ref})"
            )
            continue
        if target.read_bytes() != _clone_blob_bytes(clone_root, blob_sha):
            findings.append(
                f"STALE/MODIFIED in deployed tree: {rel} "
                f"(bytes differ from clone @ {ref})"
            )

    extra = set(deployed) - set(clone_files)
    for rel in sorted(extra):
        findings.append(
            f"EXTRA in deployed tree: {rel} (absent from clone @ {ref}; stale leftover)"
        )
    return findings


def _compare_source_tree(
    deployed_tree: Path, source_tree: Path
) -> tuple[list[str], str]:
    """Compare an actual deployed tree to one frozen hotpatch source tree."""
    source = _tree_file_hashes(source_tree, label="source tree")
    if source_tree.resolve() == deployed_tree.resolve():
        raise ValueError(
            "source tree must be a frozen attested source, not the deployed "
            "destination being checked"
        )
    source_digest = _file_set_digest(source)
    try:
        deployed = _tree_file_hashes(deployed_tree, label="deployed tree")
    except ValueError as exc:
        return [f"INVALID deployed tree: {exc}"], source_digest

    findings: list[str] = []
    for rel, expected_hash in sorted(source.items()):
        actual_hash = deployed.get(rel)
        if actual_hash is None:
            findings.append(
                f"MISSING in deployed tree: {rel} (present in frozen source)"
            )
        elif actual_hash != expected_hash:
            findings.append(
                f"STALE/MODIFIED in deployed tree: {rel} "
                "(sha256 differs from frozen source)"
            )
    for rel in sorted(set(deployed) - set(source)):
        findings.append(f"EXTRA in deployed tree: {rel} (absent from frozen source)")
    return findings, source_digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deployed-tree",
        required=True,
        help="Path to the bind-mounted deployed forward-migration tree.",
    )
    comparison_source = parser.add_mutually_exclusive_group(required=True)
    comparison_source.add_argument(
        "--clone-root",
        help="Path to the canonical clone (git repo root).",
    )
    comparison_source.add_argument(
        "--source-tree",
        help="Frozen hotpatch forward-migration source tree to attest.",
    )
    parser.add_argument("--ref", help="Target git ref/SHA the deploy claims to ship.")
    parser.add_argument(
        "--tree-rel-path",
        default="docker/migrations/forward",
        help="Path of the migration tree relative to the clone root.",
    )
    args = parser.parse_args(argv)

    deployed_tree = Path(args.deployed_tree).expanduser()
    if not deployed_tree.is_dir():
        print(
            f"ERROR: deployed tree {deployed_tree} does not exist — cannot assert "
            "migration sync (refusing to run forward-migration against an absent tree).",
            file=sys.stderr,
        )
        return 2
    if args.source_tree:
        if args.ref:
            parser.error("--ref is only valid with --clone-root")
        source_tree = Path(args.source_tree).expanduser()
        try:
            findings, source_digest = _compare_source_tree(deployed_tree, source_tree)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        expected_label = (
            f"frozen source tree {source_tree} (file-set sha256={source_digest})"
        )
    else:
        if not args.ref:
            parser.error("--ref is required with --clone-root")
        clone_root = Path(args.clone_root).expanduser()
        if not (clone_root / ".git").exists():
            print(
                f"ERROR: {clone_root} is not a git clone (no .git) — cannot resolve the "
                "canonical migration tree at the target SHA.",
                file=sys.stderr,
            )
            return 2
        try:
            resolved = _resolve_ref(clone_root, args.ref)
            findings = _compare(deployed_tree, clone_root, resolved, args.tree_rel_path)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        expected_label = f"canonical clone @ {args.ref} ({resolved[:12]})"

    if findings:
        print(
            f"FAIL: deployed migration tree {deployed_tree} is OUT OF SYNC with the "
            f"{expected_label} — "
            f"{len(findings)} drift finding(s) (OMN-13415):",
            file=sys.stderr,
        )
        for f in findings:
            print(f"  - {f}", file=sys.stderr)
        print(
            "Re-sync the deployed migration tree from the selected attested source "
            "before running forward-migration; never apply migrations from a stale "
            "bind-mounted tree.",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: deployed migration tree {deployed_tree} is byte-identical to the "
        f"{expected_label}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
