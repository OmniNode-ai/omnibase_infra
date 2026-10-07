# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""NodeMigrationFreezeCheckCompute: migration-freeze decision COMPUTE handler.

Collapses the three implementations that enforced the DB-per-repo migration
freeze (OMN-2055): ``scripts/validation/validate_migration_freeze.py`` (the
pre-commit path through ``validate.py migration_freeze``),
``scripts/check_migration_freeze.sh`` (the CI path) and the python
``--check-committed`` mode. One decision, one report shape (OMN-2362).

Pure: no filesystem, git, clock or environment access. The freeze file text,
today's date and the diff's added paths arrive on the request.

Freeze age policy (``freeze_date=YYYY-MM-DD`` in the freeze file): WARN at 30+
days, FAIL at 60+ days. A new file under a migration directory is a FAIL; a
modified or deleted one is allowed. An absent or unparseable ``freeze_date`` is
no age check, as in the scripts.

Ticket: OMN-20568 (validator conversion program, omnibase_infra first batch).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Final, Literal

from omnibase_core.models.validation.model_validation_report import (
    ModelValidationFindingEmbed,
    ModelValidationReport,
    ModelValidationRequestRef,
)
from omnibase_infra.nodes.node_migration_freeze_check_compute.models.model_migration_freeze_check_input import (
    ModelMigrationFreezeCheckInput,
)

__all__ = [
    "FREEZE_EXPIRE_DAYS",
    "FREEZE_WARN_DAYS",
    "VALIDATOR_ID",
    "NodeMigrationFreezeCheckCompute",
    "parse_freeze_date",
]

VALIDATOR_ID: Final[str] = "migration-freeze"
FREEZE_WARN_DAYS: Final[int] = 30
FREEZE_EXPIRE_DAYS: Final[int] = 60
_PROFILE: Final[Literal["strict", "default", "advisory"]] = "default"


def parse_freeze_date(freeze_text: str) -> date | None:
    """Parse the first ``freeze_date=`` line, as validate_migration_freeze.py did.

    Returns None when the field is absent or its first occurrence does not parse.
    """
    for line in freeze_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("freeze_date="):
            raw = stripped.split("=", 1)[1].strip()
            try:
                return datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=UTC).date()
            except ValueError:
                return None
    return None


class NodeMigrationFreezeCheckCompute:
    """COMPUTE handler deciding whether the migration freeze is violated or expired."""

    def handle(self, request: ModelMigrationFreezeCheckInput) -> ModelValidationReport:
        """Definition-B entry point: typed request in, canonical report out."""
        findings: list[ModelValidationFindingEmbed] = []
        if request.freeze_active:
            findings.extend(self._age_findings(request))
            findings.extend(self._violation_findings(request))
        return ModelValidationReport.from_findings(
            findings=tuple(findings),
            request=ModelValidationRequestRef(profile=_PROFILE),
            validators_run=(VALIDATOR_ID,),
        )

    @staticmethod
    def _age_findings(
        request: ModelMigrationFreezeCheckInput,
    ) -> list[ModelValidationFindingEmbed]:
        freeze_date = parse_freeze_date(request.freeze_text)
        if freeze_date is None:
            return []
        age_days = (request.today - freeze_date).days
        if age_days >= FREEZE_EXPIRE_DAYS:
            return [
                ModelValidationFindingEmbed(
                    validator_id=VALIDATOR_ID,
                    severity="FAIL",
                    location=".migration_freeze",
                    message=(
                        f"Migration freeze has EXPIRED: freeze date {freeze_date} "
                        f"({age_days} days ago); freezes are automatically "
                        f"invalidated after {FREEZE_EXPIRE_DAYS} days"
                    ),
                    remediation=(
                        "lift the freeze (remove .migration_freeze) if the DB "
                        "boundary work is complete, or renew freeze_date= with a "
                        "justification comment"
                    ),
                    rule_id="freeze-expired",
                )
            ]
        if age_days >= FREEZE_WARN_DAYS:
            return [
                ModelValidationFindingEmbed(
                    validator_id=VALIDATOR_ID,
                    severity="WARN",
                    location=".migration_freeze",
                    message=(
                        f"Migration freeze is approaching expiry: freeze date "
                        f"{freeze_date} ({age_days} days ago); it becomes an ERROR "
                        f"in {FREEZE_EXPIRE_DAYS - age_days} day(s)"
                    ),
                    remediation=(
                        "review the freeze status and update .migration_freeze "
                        "if still needed"
                    ),
                    rule_id="freeze-expiring",
                )
            ]
        return []

    @staticmethod
    def _violation_findings(
        request: ModelMigrationFreezeCheckInput,
    ) -> list[ModelValidationFindingEmbed]:
        return [
            ModelValidationFindingEmbed(
                validator_id=VALIDATOR_ID,
                severity="FAIL",
                location=path,
                message=f"NEW: {path}",
                remediation=(
                    "migrations are frozen; modify an existing migration or lift "
                    "the freeze by removing .migration_freeze (OMN-2055)"
                ),
                rule_id="new-migration-file",
            )
            for path in request.added_paths
            if path.startswith(request.migration_dirs)
        ]
