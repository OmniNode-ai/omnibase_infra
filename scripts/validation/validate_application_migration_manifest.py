#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Validate the deterministic application-migration declaration set.

The forward runner also performs a portable POSIX-shell validation before it
touches PostgreSQL.  This typed gate is the richer CI/operator proof: every
vendored node SQL artifact is either bound to one stream, owner, domain,
version, and content SHA-256, or is explicitly blocked by a ticket.  There are
no root-name defaults and no environment-provided exceptions.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TICKET = re.compile(r"^OMN-[0-9]+$")
_NODE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_CLOUD_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_LEGACY_SOURCE_CHECKSUM = re.compile(r"^[A-Za-z0-9_.:-]+$")
_ALLOWED_DOMAINS = frozenset({"tenant", "omninode_internal"})


class ManifestError(ValueError):
    """A checked-in declaration is incomplete, ambiguous, or contradictory."""


@dataclass(frozen=True, slots=True)
class MigrationDeclaration:
    artifact_path: str
    migration_stream: str
    owner: str
    domain: str
    version: str
    checksum: str


@dataclass(frozen=True, slots=True)
class BlockedMigration:
    artifact_path: str
    version: str
    checksum: str
    ticket: str
    reason: str


@dataclass(frozen=True, slots=True)
class LegacyNodeMigrationDeclaration:
    migration_stream: str
    owner: str
    domain: str
    version: str
    source_checksum: str
    ticket: str


@dataclass(frozen=True, slots=True)
class CloudAlias:
    migration_name: str
    runner_version: str


@dataclass(frozen=True, slots=True)
class ValidationResult:
    declarations: tuple[MigrationDeclaration, ...]
    blocked: tuple[BlockedMigration, ...]
    legacy_node_declarations: tuple[LegacyNodeMigrationDeclaration, ...]
    cloud_aliases: tuple[CloudAlias, ...]


def _read_tsv(
    path: Path, field_count: int, *, allow_empty: bool = False
) -> list[tuple[int, list[str]]]:
    if not path.is_file():
        raise ManifestError(f"required declaration file is missing: {path}")
    rows: list[tuple[int, list[str]]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = raw_line.split("\t")
        if len(fields) != field_count or any(field == "" for field in fields):
            raise ManifestError(
                f"{path}:{line_number}: expected {field_count} non-empty TSV fields"
            )
        rows.append((line_number, fields))
    if not rows and not allow_empty:
        raise ManifestError(f"declaration file must not be empty: {path}")
    return rows


def _node_identity(artifact_path: str) -> tuple[str, str, str]:
    parts = artifact_path.split("/")
    if len(parts) != 3 or parts[0] != "nodes" or not parts[2].endswith(".sql"):
        raise ManifestError(
            f"artifact must be nodes/<node>/<filename>.sql: {artifact_path!r}"
        )
    _, node_name, filename = parts
    if _NODE_NAME.fullmatch(node_name) is None:
        raise ManifestError(f"invalid node name in artifact: {artifact_path!r}")
    stream = f"node:{node_name}"
    version = f"{stream}:{filename}"
    return stream, version, filename


def _content_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _node_identity_from_version(version: str) -> tuple[str, str, str]:
    parts = version.split(":")
    if len(parts) != 3 or parts[0] != "node" or not parts[2].endswith(".sql"):
        raise ManifestError(f"invalid historical node migration version: {version!r}")
    _, node_name, filename = parts
    if _NODE_NAME.fullmatch(node_name) is None:
        raise ManifestError(f"invalid node name in historical version: {version!r}")
    if _NODE_NAME.fullmatch(filename.removesuffix(".sql")) is None:
        raise ManifestError(f"invalid filename in historical version: {version!r}")
    return f"node:{node_name}", node_name, filename


def validate_manifests(
    migrations_dir: Path,
    declaration_path: Path,
    blocked_path: Path,
    legacy_node_declaration_path: Path,
    cloud_alias_path: Path,
    *,
    require_complete: bool = False,
) -> ValidationResult:
    """Parse and validate all application migration declarations."""

    declarations: list[MigrationDeclaration] = []
    blocked: list[BlockedMigration] = []
    legacy_node_declarations: list[LegacyNodeMigrationDeclaration] = []
    aliases: list[CloudAlias] = []

    for line_number, fields in _read_tsv(declaration_path, 6):
        declaration = MigrationDeclaration(*fields)
        expected_stream, expected_version, _ = _node_identity(declaration.artifact_path)
        if declaration.migration_stream != expected_stream:
            raise ManifestError(
                f"{declaration_path}:{line_number}: unknown migration stream "
                f"{declaration.migration_stream!r}; expected {expected_stream!r}"
            )
        if declaration.owner != expected_stream:
            raise ManifestError(
                f"{declaration_path}:{line_number}: owner must equal the node stream"
            )
        if declaration.domain not in _ALLOWED_DOMAINS:
            raise ManifestError(
                f"{declaration_path}:{line_number}: unknown domain "
                f"{declaration.domain!r}"
            )
        if declaration.version != expected_version:
            raise ManifestError(
                f"{declaration_path}:{line_number}: version must preserve exact "
                f"runner identity {expected_version!r}"
            )
        if _SHA256.fullmatch(declaration.checksum) is None:
            raise ManifestError(
                f"{declaration_path}:{line_number}: checksum is not lowercase SHA-256"
            )
        artifact = migrations_dir / declaration.artifact_path
        if not artifact.is_file():
            raise ManifestError(f"declared artifact is missing: {artifact}")
        actual_checksum = _content_sha256(artifact)
        if actual_checksum != declaration.checksum:
            raise ManifestError(
                f"conflicting checksum for {declaration.version}: "
                f"declared={declaration.checksum}, actual={actual_checksum}"
            )
        declarations.append(declaration)

    for line_number, fields in _read_tsv(blocked_path, 5, allow_empty=True):
        item = BlockedMigration(*fields)
        _, expected_version, _ = _node_identity(item.artifact_path)
        if item.version != expected_version:
            raise ManifestError(
                f"{blocked_path}:{line_number}: blocked version must preserve exact "
                f"runner identity {expected_version!r}"
            )
        if _SHA256.fullmatch(item.checksum) is None:
            raise ManifestError(
                f"{blocked_path}:{line_number}: checksum is not lowercase SHA-256"
            )
        if _TICKET.fullmatch(item.ticket) is None:
            raise ManifestError(
                f"{blocked_path}:{line_number}: invalid blocker ticket {item.ticket!r}"
            )
        artifact = migrations_dir / item.artifact_path
        if not artifact.is_file():
            raise ManifestError(f"blocked artifact is missing: {artifact}")
        actual_checksum = _content_sha256(artifact)
        if actual_checksum != item.checksum:
            raise ManifestError(
                f"conflicting checksum for blocked {item.version}: "
                f"declared={item.checksum}, actual={actual_checksum}"
            )
        blocked.append(item)

    for line_number, fields in _read_tsv(
        legacy_node_declaration_path, 6, allow_empty=True
    ):
        legacy_item = LegacyNodeMigrationDeclaration(*fields)
        expected_stream, node_name, filename = _node_identity_from_version(
            legacy_item.version
        )
        if legacy_item.migration_stream != expected_stream:
            raise ManifestError(
                f"{legacy_node_declaration_path}:{line_number}: unknown migration "
                f"stream {legacy_item.migration_stream!r}; expected {expected_stream!r}"
            )
        if legacy_item.owner != expected_stream:
            raise ManifestError(
                f"{legacy_node_declaration_path}:{line_number}: owner must equal "
                "the node stream"
            )
        if legacy_item.domain not in _ALLOWED_DOMAINS:
            raise ManifestError(
                f"{legacy_node_declaration_path}:{line_number}: unknown domain "
                f"{legacy_item.domain!r}"
            )
        if _LEGACY_SOURCE_CHECKSUM.fullmatch(legacy_item.source_checksum) is None:
            raise ManifestError(
                f"{legacy_node_declaration_path}:{line_number}: malformed legacy "
                "source checksum"
            )
        if _TICKET.fullmatch(legacy_item.ticket) is None:
            raise ManifestError(
                f"{legacy_node_declaration_path}:{line_number}: invalid blocker "
                f"ticket {legacy_item.ticket!r}"
            )
        artifact = migrations_dir / "nodes" / node_name / filename
        if artifact.exists():
            raise ManifestError(
                f"{legacy_node_declaration_path}:{line_number}: legacy declaration "
                f"has vendored artifact: {artifact.relative_to(migrations_dir)}"
            )
        legacy_node_declarations.append(legacy_item)

    for line_number, fields in _read_tsv(cloud_alias_path, 2):
        alias = CloudAlias(*fields)
        if (
            _CLOUD_NAME.fullmatch(alias.migration_name) is None
            or _CLOUD_NAME.fullmatch(alias.runner_version) is None
            or not alias.runner_version.endswith(".sql")
        ):
            raise ManifestError(
                f"{cloud_alias_path}:{line_number}: malformed cloud migration alias"
            )
        aliases.append(alias)

    declared_paths = [item.artifact_path for item in declarations]
    blocked_paths = [item.artifact_path for item in blocked]
    legacy_versions = [item.version for item in legacy_node_declarations]
    declared_identities = [
        (item.migration_stream, item.domain, item.version) for item in declarations
    ]
    _reject_duplicates(declared_paths, "double migration declaration")
    _reject_duplicates(blocked_paths, "duplicate blocked migration")
    _reject_duplicates(legacy_versions, "duplicate historical node migration")
    _reject_duplicates(declared_identities, "duplicate migration version")
    _reject_duplicates(
        [item.migration_name for item in aliases], "duplicate cloud migration alias"
    )
    _reject_duplicates(
        [item.runner_version for item in aliases], "duplicate cloud runner version"
    )

    overlap = sorted(set(declared_paths) & set(blocked_paths))
    if overlap:
        raise ManifestError(
            f"double migration declaration across active/blocked sets: {overlap!r}"
        )
    active_versions = {item.version for item in declarations}
    blocked_versions = {item.version for item in blocked}
    legacy_overlap = sorted((active_versions | blocked_versions) & set(legacy_versions))
    if legacy_overlap:
        raise ManifestError(
            "historical node migration conflicts with an active or blocked "
            f"declaration: {legacy_overlap!r}"
        )

    filesystem_paths = {
        str(path.relative_to(migrations_dir))
        for path in (migrations_dir / "nodes").glob("*/*.sql")
        if path.is_file()
    }
    manifest_paths = set(declared_paths) | set(blocked_paths)
    missing = sorted(filesystem_paths - manifest_paths)
    extra = sorted(manifest_paths - filesystem_paths)
    if missing or extra:
        raise ManifestError(
            "migration declaration set differs from the vendored node tree: "
            f"missing={missing!r}, extra={extra!r}"
        )
    if require_complete and blocked:
        blockers = sorted({item.ticket for item in blocked})
        raise ManifestError(
            f"migration declaration set is incomplete: {len(blocked)} blocked "
            f"artifact(s), blockers={blockers!r}"
        )

    return ValidationResult(
        tuple(declarations),
        tuple(blocked),
        tuple(legacy_node_declarations),
        tuple(aliases),
    )


def _reject_duplicates(values: Sequence[Hashable], label: str) -> None:
    seen: set[Hashable] = set()
    duplicates: set[Hashable] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise ManifestError(f"{label}: {sorted(map(str, duplicates))!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[2]
    default_migrations = repo_root / "docker" / "migrations" / "forward"
    default_ledger = default_migrations / "_ledger"
    parser.add_argument("--migrations-dir", type=Path, default=default_migrations)
    parser.add_argument(
        "--declarations",
        type=Path,
        default=default_ledger / "application-migrations.tsv",
    )
    parser.add_argument(
        "--blocked",
        type=Path,
        default=default_ledger / "application-migration-blocks.tsv",
    )
    parser.add_argument(
        "--legacy-node-declarations",
        type=Path,
        default=default_ledger / "legacy-node-migrations.tsv",
    )
    parser.add_argument(
        "--cloud-aliases",
        type=Path,
        default=default_ledger / "cloud-migration-aliases.tsv",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail when any checked-in blocker remains.",
    )
    args = parser.parse_args(argv)

    try:
        result = validate_manifests(
            args.migrations_dir,
            args.declarations,
            args.blocked,
            args.legacy_node_declarations,
            args.cloud_aliases,
            require_complete=args.require_complete,
        )
    except ManifestError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(
        "PASS: deterministic application migration declarations validated "
        f"({len(result.declarations)} active, {len(result.blocked)} blocked, "
        f"{len(result.legacy_node_declarations)} historical, "
        f"{len(result.cloud_aliases)} cloud aliases)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
