# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Flake detection result model aggregating flake analysis across checks.

Ticket: OMN-2151
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.validation.model_flake_record import ModelFlakeRecord


class ModelFlakeDetectionResult(BaseModel):
    """Aggregate result of flake detection across all checks.

    Attributes:
        records: Tuple of individual flake records per check.
        has_suspected_flakes: Whether any flakes were suspected.
        quarantine_check_codes: Check codes that triggered quarantine.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    records: tuple[ModelFlakeRecord, ...] = Field(
        default_factory=tuple,
        description="Individual flake records per check.",
    )
    has_suspected_flakes: bool = Field(
        default=False,
        description="Whether any flakes were suspected.",
    )
    quarantine_check_codes: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Check codes that triggered quarantine.",
    )

    @property
    def quarantine_reasons(self) -> tuple[str, ...]:
        """Human-readable reasons for quarantine.

        Returns:
            Tuple of quarantine reason strings.
        """
        return tuple(
            f"{r.check_code}: flake suspected (first={r.first_passed}, rerun={r.rerun_passed})"
            for r in self.records
            if r.is_flake_suspected
        )


__all__: list[str] = ["ModelFlakeDetectionResult"]
