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
_VERIFIED_AT = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
# OMN-16919: the applied_at values a cross-source reconciliation preserves. These
# are compared against live timestamptz columns by bootstrap.sql, so the spelling
# is pinned here rather than left to whatever a hand-edit produces.
_APPLIED_AT = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?\+00$"
)
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
class VerifiedChecksumAdoption:
    """OMN-15857: one hand-written sentinel checksum, proven safe to adopt.

    ``manifest_checksum`` pins the content SHA-256 the equivalence proof ran
    against.  It is re-checked against the live manifest on every validation, so
    rewriting the migration file after the proof invalidates the adoption at PR
    time instead of at deploy time.
    """

    version: str
    source_checksum: str
    manifest_checksum: str
    ticket: str
    receipt_sha256: str
    verified_at: str


@dataclass(frozen=True, slots=True)
class VerifiedDivergentAdoption:
    """OMN-16915: one DIVERGENT but well-formed content hash, proven safe.

    The mirror image of ``VerifiedChecksumAdoption``. That one covers a
    *sentinel* -- a hand-written non-hash standing in for an unproven hand-apply.
    This one covers a row whose checksum is a perfectly good sha256 that names an
    *earlier revision* of the checked-in file: the lane applied that revision and
    was never re-converged.

    The validity rules are therefore inverted. A sentinel adoption is rejected if
    its source checksum IS 64-hex; a divergent adoption is rejected if it is NOT,
    and again if it happens to EQUAL the manifest checksum (that row needs no
    adoption -- bootstrap.sql accepts it directly, and declaring it would claim a
    proof for a question nobody asked).
    """

    version: str
    source_checksum: str
    manifest_checksum: str
    ticket: str
    receipt_sha256: str
    verified_at: str


@dataclass(frozen=True, slots=True)
class VerifiedCrossSourceAdoption:
    """OMN-16919: one version declared by BOTH source ledgers at once.

    The two ledgers above each answer "what did THIS row's checksum mean?". This
    one answers a different question entirely: what happens when
    ``public.schema_migrations`` and ``public.omnimarket_schema_migrations`` both
    declare the same version, having recorded the same application at different
    times through different runner generations.

    It is not a content dispute -- admission requires that both sides already
    resolve to the same manifest checksum, through whichever governed surface
    applies to each. What is reconciled here is only ``applied_at`` and
    provenance. Both source checksums and both timestamps are preserved verbatim,
    and ``reconciled_applied_at`` must be exactly one of the two observed values:
    the declaration states which, so nothing is invented at runtime.
    """

    version: str
    node_source_checksum: str
    omnimarket_source_checksum: str
    manifest_checksum: str
    node_applied_at: str
    omnimarket_applied_at: str
    reconciled_applied_at: str
    ticket: str
    receipt_sha256: str
    verified_at: str


@dataclass(frozen=True, slots=True)
class CloudAlias:
    migration_name: str
    runner_version: str


@dataclass(frozen=True, slots=True)
class ValidationResult:
    declarations: tuple[MigrationDeclaration, ...]
    blocked: tuple[BlockedMigration, ...]
    legacy_node_declarations: tuple[LegacyNodeMigrationDeclaration, ...]
    verified_adoptions: tuple[VerifiedChecksumAdoption, ...]
    verified_divergent_adoptions: tuple[VerifiedDivergentAdoption, ...]
    verified_cross_source_adoptions: tuple[VerifiedCrossSourceAdoption, ...]
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
    verified_adoption_path: Path,
    verified_divergent_adoption_path: Path,
    verified_cross_source_adoption_path: Path,
    cloud_alias_path: Path,
    *,
    require_complete: bool = False,
) -> ValidationResult:
    """Parse and validate all application migration declarations."""

    declarations: list[MigrationDeclaration] = []
    blocked: list[BlockedMigration] = []
    legacy_node_declarations: list[LegacyNodeMigrationDeclaration] = []
    verified_adoptions: list[VerifiedChecksumAdoption] = []
    verified_divergent_adoptions: list[VerifiedDivergentAdoption] = []
    verified_cross_source_adoptions: list[VerifiedCrossSourceAdoption] = []
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

    # OMN-15857: verified-adoption declarations.  Validated after the active
    # declaration set is parsed so the pinned manifest checksum can be compared
    # against the live one.
    declared_by_version = {item.version: item for item in declarations}
    for line_number, fields in _read_tsv(verified_adoption_path, 6, allow_empty=True):
        adoption = VerifiedChecksumAdoption(*fields)
        declared = declared_by_version.get(adoption.version)
        if declared is None:
            raise ManifestError(
                f"{verified_adoption_path}:{line_number}: verified adoption for "
                f"{adoption.version!r} has no active migration declaration; an "
                "adoption can only ever restate a checksum the manifest already "
                "owns"
            )
        if adoption.manifest_checksum != declared.checksum:
            raise ManifestError(
                f"{verified_adoption_path}:{line_number}: verified adoption for "
                f"{adoption.version!r} was proven against content "
                f"{adoption.manifest_checksum!r} but the manifest now declares "
                f"{declared.checksum!r}. The migration file changed after the "
                "proof, so the proof no longer covers it -- re-run "
                "scripts/migrations/verify_migration_checksum_adoption.py and "
                "commit the new receipt."
            )
        if _SHA256.fullmatch(adoption.manifest_checksum) is None:
            raise ManifestError(
                f"{verified_adoption_path}:{line_number}: malformed manifest "
                f"checksum {adoption.manifest_checksum!r}"
            )
        if _SHA256.fullmatch(adoption.receipt_sha256) is None:
            raise ManifestError(
                f"{verified_adoption_path}:{line_number}: malformed receipt "
                f"sha256 {adoption.receipt_sha256!r}"
            )
        if _LEGACY_SOURCE_CHECKSUM.fullmatch(adoption.source_checksum) is None:
            raise ManifestError(
                f"{verified_adoption_path}:{line_number}: malformed adopted "
                f"source checksum {adoption.source_checksum!r}"
            )
        if _SHA256.fullmatch(adoption.source_checksum) is not None:
            raise ManifestError(
                f"{verified_adoption_path}:{line_number}: {adoption.version!r} "
                "carries a 64-hex source checksum, which bootstrap.sql already "
                "compares directly; a verified adoption exists only for a "
                "non-canonical sentinel"
            )
        if adoption.source_checksum == "applied-by-runner":
            raise ManifestError(
                f"{verified_adoption_path}:{line_number}: {adoption.version!r} "
                "carries the runner literal, which bootstrap.sql already "
                "adopts; a verified adoption exists only for a sentinel that "
                "would otherwise abort the run"
            )
        if _TICKET.fullmatch(adoption.ticket) is None:
            raise ManifestError(
                f"{verified_adoption_path}:{line_number}: invalid adoption "
                f"ticket {adoption.ticket!r}"
            )
        if _VERIFIED_AT.fullmatch(adoption.verified_at) is None:
            raise ManifestError(
                f"{verified_adoption_path}:{line_number}: malformed "
                f"verified_at {adoption.verified_at!r}; expected YYYY-MM-DD"
            )
        verified_adoptions.append(adoption)

    # OMN-16915: verified DIVERGENT-bytes adoptions.  Same shape, inverted rules.
    for line_number, fields in _read_tsv(
        verified_divergent_adoption_path, 6, allow_empty=True
    ):
        divergent = VerifiedDivergentAdoption(*fields)
        declared = declared_by_version.get(divergent.version)
        if declared is None:
            raise ManifestError(
                f"{verified_divergent_adoption_path}:{line_number}: verified "
                f"divergent adoption for {divergent.version!r} has no active "
                "migration declaration; an adoption can only ever restate a "
                "checksum the manifest already owns"
            )
        if divergent.manifest_checksum != declared.checksum:
            raise ManifestError(
                f"{verified_divergent_adoption_path}:{line_number}: verified "
                f"divergent adoption for {divergent.version!r} was proven against "
                f"content {divergent.manifest_checksum!r} but the manifest now "
                f"declares {declared.checksum!r}. The migration file changed "
                "after the proof, so the proof no longer covers it -- re-run "
                "scripts/migrations/verify_migration_checksum_adoption.py and "
                "commit the new receipt."
            )
        if _SHA256.fullmatch(divergent.manifest_checksum) is None:
            raise ManifestError(
                f"{verified_divergent_adoption_path}:{line_number}: malformed "
                f"manifest checksum {divergent.manifest_checksum!r}"
            )
        if _SHA256.fullmatch(divergent.receipt_sha256) is None:
            raise ManifestError(
                f"{verified_divergent_adoption_path}:{line_number}: malformed "
                f"receipt sha256 {divergent.receipt_sha256!r}"
            )
        # Inverted rule 1: this file exists only for real content hashes.
        if _SHA256.fullmatch(divergent.source_checksum) is None:
            raise ManifestError(
                f"{verified_divergent_adoption_path}:{line_number}: "
                f"{divergent.version!r} carries source checksum "
                f"{divergent.source_checksum!r}, which is not a 64-hex content "
                "hash. A sentinel is not a divergent-bytes case and must not be "
                "laundered into one -- it belongs in "
                "_ledger/verified-checksum-adoptions.tsv under OMN-15857."
            )
        # Inverted rule 2: a row that already agrees needs no adoption at all.
        if divergent.source_checksum == divergent.manifest_checksum:
            raise ManifestError(
                f"{verified_divergent_adoption_path}:{line_number}: "
                f"{divergent.version!r} declares a source checksum identical to "
                "the manifest checksum, so nothing diverges; bootstrap.sql "
                "accepts that row directly and this declaration would assert a "
                "proof for a question nobody asked"
            )
        if _TICKET.fullmatch(divergent.ticket) is None:
            raise ManifestError(
                f"{verified_divergent_adoption_path}:{line_number}: invalid "
                f"adoption ticket {divergent.ticket!r}"
            )
        if _VERIFIED_AT.fullmatch(divergent.verified_at) is None:
            raise ManifestError(
                f"{verified_divergent_adoption_path}:{line_number}: malformed "
                f"verified_at {divergent.verified_at!r}; expected YYYY-MM-DD"
            )
        verified_divergent_adoptions.append(divergent)

    # OMN-16919: cross-source reconciliations.  Parsed after the two per-row
    # adoption files because the overlap rule below is expressed in terms of all
    # three.
    for line_number, fields in _read_tsv(
        verified_cross_source_adoption_path, 10, allow_empty=True
    ):
        cross = VerifiedCrossSourceAdoption(*fields)
        declared = declared_by_version.get(cross.version)
        if declared is None:
            raise ManifestError(
                f"{verified_cross_source_adoption_path}:{line_number}: cross-source "
                f"reconciliation for {cross.version!r} has no active manifest "
                "declaration"
            )
        if cross.manifest_checksum != declared.checksum:
            raise ManifestError(
                f"{verified_cross_source_adoption_path}:{line_number}: cross-source "
                f"reconciliation for {cross.version!r} was proven against content "
                f"{cross.manifest_checksum!r} but the manifest now declares "
                f"{declared.checksum!r}; re-run the verification"
            )
        if _SHA256.fullmatch(cross.manifest_checksum) is None:
            raise ManifestError(
                f"{verified_cross_source_adoption_path}:{line_number}: malformed "
                f"manifest checksum {cross.manifest_checksum!r}"
            )
        if _SHA256.fullmatch(cross.receipt_sha256) is None:
            raise ManifestError(
                f"{verified_cross_source_adoption_path}:{line_number}: malformed "
                f"receipt sha256 {cross.receipt_sha256!r}"
            )
        # The omnimarket side always carries a real content hash -- that relation
        # has no sentinel spelling.  The node side may legitimately be either a
        # hash or a sentinel, so it is not shape-checked here; bootstrap.sql
        # compares it against the live row verbatim.
        if _SHA256.fullmatch(cross.omnimarket_source_checksum) is None:
            raise ManifestError(
                f"{verified_cross_source_adoption_path}:{line_number}: "
                f"{cross.version!r} carries omnimarket source checksum "
                f"{cross.omnimarket_source_checksum!r}, which is not a 64-hex "
                "content hash"
            )
        for label, value in (
            ("node_applied_at", cross.node_applied_at),
            ("omnimarket_applied_at", cross.omnimarket_applied_at),
            ("reconciled_applied_at", cross.reconciled_applied_at),
        ):
            if _APPLIED_AT.fullmatch(value) is None:
                raise ManifestError(
                    f"{verified_cross_source_adoption_path}:{line_number}: "
                    f"malformed {label} {value!r}; expected "
                    "'YYYY-MM-DD HH:MM:SS[.ffffff]+00'"
                )
        # The reconciled timestamp is DECLARED, not derived at runtime -- but it
        # may only ever be one of the two values actually observed. This is the
        # line that stops a reconciliation from inventing a third history.
        if cross.reconciled_applied_at not in (
            cross.node_applied_at,
            cross.omnimarket_applied_at,
        ):
            raise ManifestError(
                f"{verified_cross_source_adoption_path}:{line_number}: "
                f"{cross.version!r} declares reconciled_applied_at "
                f"{cross.reconciled_applied_at!r}, which is neither the node "
                f"value {cross.node_applied_at!r} nor the omnimarket value "
                f"{cross.omnimarket_applied_at!r}. A reconciliation chooses "
                "between the two observed applications; it does not invent a "
                "third."
            )
        # Two sources recording the same instant is not a cross-source conflict
        # -- there is nothing to reconcile, and bootstrap.sql converges on its
        # own. Declaring one would claim a decision nobody had to make.
        if cross.node_applied_at == cross.omnimarket_applied_at:
            raise ManifestError(
                f"{verified_cross_source_adoption_path}:{line_number}: "
                f"{cross.version!r} declares identical node and omnimarket "
                "applied_at values, so there is no cross-source disagreement to "
                "reconcile"
            )
        if _TICKET.fullmatch(cross.ticket) is None:
            raise ManifestError(
                f"{verified_cross_source_adoption_path}:{line_number}: invalid "
                f"adoption ticket {cross.ticket!r}"
            )
        if _VERIFIED_AT.fullmatch(cross.verified_at) is None:
            raise ManifestError(
                f"{verified_cross_source_adoption_path}:{line_number}: malformed "
                f"verified_at {cross.verified_at!r}; expected YYYY-MM-DD"
            )
        verified_cross_source_adoptions.append(cross)

    # OMN-16919: the overlap rule, extended rather than relaxed.
    #
    # OMN-16915 forbade a version appearing in both per-row adoption files
    # outright: a row has one applied checksum, so it cannot simultaneously be an
    # unproven hand-apply and a stale revision. That reasoning holds for a single
    # row and is kept. What it did not anticipate is that the SAME VERSION can be
    # carried by two DIFFERENT source relations, each with its own row and its own
    # honest answer -- which is exactly the state the cross-source file declares.
    #
    # So overlap is admissible only when a cross-source record covers the version.
    # Without one, the two declarations are still making incompatible claims about
    # one row and the original refusal stands.
    _cross_versions = {item.version for item in verified_cross_source_adoptions}
    _sentinel_versions = {item.version for item in verified_adoptions}
    _divergent_versions = {item.version for item in verified_divergent_adoptions}
    for version in sorted(_sentinel_versions & _divergent_versions):
        if version not in _cross_versions:
            raise ManifestError(
                f"{verified_divergent_adoption_path}: {version!r} is declared in "
                "BOTH the sentinel and the divergent adoption ledgers with no "
                f"cross-source record in {verified_cross_source_adoption_path}. A "
                "version has one applied checksum per source relation; it cannot "
                "simultaneously be an unproven hand-apply and a stale revision of "
                "the same row."
            )

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
    _reject_duplicates(
        [item.version for item in verified_adoptions],
        "duplicate verified checksum adoption",
    )
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
        tuple(verified_adoptions),
        tuple(verified_divergent_adoptions),
        tuple(verified_cross_source_adoptions),
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
        "--verified-adoptions",
        type=Path,
        default=default_ledger / "verified-checksum-adoptions.tsv",
    )
    parser.add_argument(
        "--verified-divergent-adoptions",
        type=Path,
        default=default_ledger / "verified-divergent-adoptions.tsv",
    )
    parser.add_argument(
        "--verified-cross-source-adoptions",
        type=Path,
        default=default_ledger / "verified-cross-source-adoptions.tsv",
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
            args.verified_adoptions,
            args.verified_divergent_adoptions,
            args.verified_cross_source_adoptions,
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
        f"{len(result.verified_adoptions)} verified adoptions, "
        f"{len(result.verified_divergent_adoptions)} verified divergent adoptions, "
        f"{len(result.verified_cross_source_adoptions)} cross-source "
        "reconciliations, "
        f"{len(result.cloud_aliases)} cloud aliases)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
