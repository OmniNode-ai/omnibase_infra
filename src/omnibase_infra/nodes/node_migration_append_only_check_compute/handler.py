# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pure OMN-16705 append-only migration decision and canonical report."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from omnibase_core.models.validation.model_validation_report import (
    ModelValidationFindingEmbed,
    ModelValidationReport,
    ModelValidationRequestRef,
)
from omnibase_infra.nodes.node_migration_append_only_check_compute.models import (
    ModelMigrationAppendOnlyCheckInput,
)

VALIDATOR_ID: Final[str] = "migration-append-only"
FORWARD_PREFIX: Final[str] = "docker/migrations/forward/"
MANIFEST_REPO_PATH: Final[str] = f"{FORWARD_PREFIX}_ledger/application-migrations.tsv"
SUPERSESSIONS_REPO_PATH: Final[str] = (
    f"{FORWARD_PREFIX}_ledger/migration-supersessions.tsv"
)

_TICKET = re.compile(r"^OMN-[0-9]+$")
_ARTIFACT = re.compile(
    r"^nodes/(?P<node>[A-Za-z0-9_][A-Za-z0-9_.-]*)/(?P<file>[0-9]+[A-Za-z0-9_.-]*\.sql)$"
)
_ORDINAL = re.compile(r"^(?P<ordinal>[0-9]+)")
_MUTATING_STATUSES = frozenset({"M", "D", "T", "R", "C"})


class AppendOnlyViolationError(Exception):
    """The check cannot decide from an invalid or unavailable ledger snapshot."""


@dataclass(frozen=True, slots=True)
class Supersession:
    """One four-column ledger row, retaining the original parsing semantics."""

    artifact_path: str
    superseded_by: str
    ticket: str
    reason: str


def declared_artifacts(manifest_text: str | None) -> frozenset[str]:
    """Read the artifact_path column without normalising whitespace."""
    if not manifest_text:
        return frozenset()
    return frozenset(
        line.split("\t", 1)[0] for line in manifest_text.splitlines() if line.strip()
    )


def validate_base_manifest(manifest_text: str | None, base_ref: str) -> frozenset[str]:
    """Fail before current-ledger reads, preserving the script's error ordering."""
    declared = declared_artifacts(manifest_text)
    if not declared:
        raise AppendOnlyViolationError(
            f"anti-vacuity: {MANIFEST_REPO_PATH} at {base_ref} declared no migrations; "
            "the guard would pass everything"
        )
    return declared


def parse_supersessions(text: str | None) -> tuple[Supersession, ...]:
    """Parse every nonblank row, including rows unrelated to the diff."""
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
    assert ordinal_match is not None
    return int(ordinal_match.group("ordinal"))


def _node(artifact_path: str) -> str:
    match = _ARTIFACT.match(artifact_path)
    if match is None:
        raise AppendOnlyViolationError(
            f"supersession path must be nodes/<node>/<ordinal>_<name>.sql: {artifact_path!r}"
        )
    return match.group("node")


def _authorised(
    artifact_path: str,
    supersessions: tuple[Supersession, ...],
    added_artifacts: frozenset[str],
    current_manifest: frozenset[str],
    existing_paths: frozenset[str],
) -> str | None:
    """Return the script's exact refusal reason, or None for an authorised edit."""
    rows = [row for row in supersessions if row.artifact_path == artifact_path]
    if not rows:
        return (
            "no supersession row in "
            f"{SUPERSESSIONS_REPO_PATH}. An already-declared migration is applied "
            "history and its bytes are frozen: add a NEW file with the next "
            "ordinal in the same node directory instead of editing this one. "
            "Before writing a supersession row, ASK THE LANE whether this "
            "migration is already applied -- "
            "scripts/migrations/check_migration_applied_on_lane.py reads "
            "platform_catalog.schema_migrations, the table the runner gates on; "
            "onex_application_migration_manifest is a per-session TEMP table and "
            "reads clean on every lane (OMN-17139)."
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
        if f"{FORWARD_PREFIX}{row.superseded_by}" not in existing_paths:
            problems.append(f"{row.superseded_by} does not exist on disk")
            continue
        if row.superseded_by not in current_manifest:
            problems.append(
                f"{row.superseded_by} is not declared in {MANIFEST_REPO_PATH}"
            )
            continue
        return None
    return "; ".join(problems)


class NodeMigrationAppendOnlyCheckCompute:
    """Definition-B handler: explicit frozen input to the OMN-2362 report."""

    def handle(
        self, request: ModelMigrationAppendOnlyCheckInput
    ) -> ModelValidationReport:
        """Preserve check()'s decisions without filesystem, git, clock or env I/O."""
        base_declared = validate_base_manifest(
            request.base_manifest_text, request.base_ref
        )
        current_manifest = declared_artifacts(request.manifest_text)
        supersessions = parse_supersessions(request.supersessions_text)
        changed = dict(request.changed_paths)
        added_artifacts = frozenset(
            path[len(FORWARD_PREFIX) :]
            for path, status in changed.items()
            if status == "A" and path.startswith(FORWARD_PREFIX)
        )
        findings: list[ModelValidationFindingEmbed] = []
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
                request.existing_paths,
            )
            if reason is not None:
                findings.append(
                    ModelValidationFindingEmbed(
                        validator_id=VALIDATOR_ID,
                        severity="FAIL",
                        location=path,
                        message=f"{path} ({status}): {reason}",
                        remediation="Add a new migration with a higher ordinal, or declare a valid supersession with its successor in this change.",
                        rule_id="applied-migration-mutated",
                    )
                )
        return ModelValidationReport.from_findings(
            findings=tuple(findings),
            request=ModelValidationRequestRef(profile="default"),
            validators_run=(VALIDATOR_ID,),
        )
