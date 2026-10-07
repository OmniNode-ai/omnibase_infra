# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pure migration-sequence duplicate detection (OMN-20568).

Preserves the script's staged-file gate, inclusion of deleted staged paths,
and pairing with the first path in sorted order. All paths are explicit inputs;
no filesystem, git, clock or environment access occurs here.
"""

from __future__ import annotations

from pathlib import PurePath
from typing import Final

from omnibase_core.models.validation.model_validation_report import (
    ModelValidationFindingEmbed,
    ModelValidationReport,
    ModelValidationRequestRef,
)
from omnibase_infra.nodes.node_migration_sequence_check_compute.models import (
    ModelMigrationSequenceCheckInput,
)

__all__ = [
    "VALIDATOR_ID",
    "NodeMigrationSequenceCheckCompute",
    "migration_scan_paths",
]

VALIDATOR_ID: Final[str] = "migration-sequence"


def _sequence_number(filename: str) -> int | None:
    path = PurePath(filename)
    if path.suffix.lower() != ".sql":
        return None
    prefix = ""
    for character in path.stem:
        if not character.isdigit():
            break
        prefix += character
    return int(prefix) if prefix else None


def _excluded(path: str, prefixes: tuple[str, ...]) -> bool:
    return path.replace("\\", "/").startswith(prefixes)


def migration_scan_paths(request: ModelMigrationSequenceCheckInput) -> tuple[str, ...]:
    """Apply the staged-migration gate and return the script's sorted scan paths.

    Staged migrations absent from disk are included, even staged deletions.
    Files without a leading number still contribute to the scan count.
    """
    staged_migrations: set[str] = set()
    for path in request.staged_paths:
        if _excluded(path, request.excluded_subtree_prefixes):
            continue
        for directory in request.migration_dirs:
            if path.startswith((directory + "/", directory.replace("/", "\\") + "\\")):
                if _sequence_number(PurePath(path).name) is not None:
                    staged_migrations.add(path)
                    break
    if not staged_migrations:
        return ()
    all_files = [
        path
        for path in request.migration_paths
        if not _excluded(path, request.excluded_subtree_prefixes)
    ]
    for path in staged_migrations:
        if path not in all_files:
            all_files.append(path)
    return tuple(sorted(all_files))


class NodeMigrationSequenceCheckCompute:
    """Definition-B handler: typed request in, canonical OMN-2362 report out."""

    def handle(
        self, request: ModelMigrationSequenceCheckInput
    ) -> ModelValidationReport:
        findings: list[ModelValidationFindingEmbed] = []
        seen: dict[int, str] = {}
        for path in migration_scan_paths(request):
            sequence = _sequence_number(PurePath(path).name)
            if sequence is None:
                continue
            if sequence in seen:
                first = seen[sequence]
                findings.append(
                    ModelValidationFindingEmbed(
                        validator_id=VALIDATOR_ID,
                        severity="FAIL",
                        location=path,
                        message=f"  seq {sequence:03d}: {first!r}  <-->  {path!r}",
                        remediation=(
                            "renumber the new migration to use the next available sequence."
                        ),
                        rule_id="duplicate-sequence",
                        evidence={
                            "sequence": sequence,
                            "file_a": first,
                            "file_b": path,
                        },
                    )
                )
            else:
                seen[sequence] = path
        return ModelValidationReport.from_findings(
            findings=tuple(findings),
            request=ModelValidationRequestRef(profile="default"),
            validators_run=(VALIDATOR_ID,),
        )
