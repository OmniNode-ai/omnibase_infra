# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Aggregated verification report across all checks for a contract."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_validation_verdict import EnumValidationVerdict
from omnibase_infra.verification.models.model_contract_check_result import (
    ModelContractCheckResult,
)


class ModelContractVerificationReport(BaseModel):
    """Aggregated report across all verification checks for a single contract.

    Contains the full set of individual check results and an overall verdict
    computed from the check outcomes. The report includes execution context
    metadata required for operationally meaningful QUARANTINE interpretation.

    Attributes:
        contract_name: Name of the contract being verified.
        node_type: The ONEX node type (e.g., ORCHESTRATOR, REDUCER, EFFECT, COMPUTE).
        checks: Tuple of individual check results.
        overall_verdict: Aggregated verdict (FAIL > QUARANTINE > PASS).
        checked_at: UTC timestamp when verification was performed.
        probe_mode: Whether primary or fallback probe mechanisms were used.
        degraded_probes: Which probes had to use fallback mechanisms.
        runtime_target: The runtime endpoint targeted (e.g., localhost:8085).
        duration_ms: Total verification time in milliseconds.
        report_fingerprint: SHA-256 of sorted check results for dedup.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    contract_name: str = Field(
        ...,
        description="Name of the contract being verified.",
    )
    node_type: str = Field(
        ...,
        description="The ONEX node type (e.g., ORCHESTRATOR, REDUCER, EFFECT, COMPUTE).",
    )
    checks: tuple[ModelContractCheckResult, ...] = Field(
        ...,
        description="Tuple of individual check results.",
    )
    overall_verdict: EnumValidationVerdict = Field(
        ...,
        description="Aggregated verdict (FAIL > QUARANTINE > PASS).",
    )
    checked_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when verification was performed.",
    )
    probe_mode: str = Field(
        default="primary",
        description="Whether primary or fallback probe mechanisms were used.",
    )
    degraded_probes: tuple[str, ...] = Field(
        default=(),
        description="Which probes had to use fallback mechanisms.",
    )
    runtime_target: str = Field(
        default_factory=lambda: os.environ["ONEX_RUNTIME_TARGET"],
        description="The runtime endpoint targeted.",
    )
    duration_ms: int = Field(
        default=0,
        description="Total verification time in milliseconds.",
    )
    report_fingerprint: str = Field(
        default="",
        description="SHA-256 of sorted check results for dedup.",
    )

    def __bool__(self) -> bool:
        """Return True only when overall_verdict is PASS."""
        return self.overall_verdict == EnumValidationVerdict.PASS

    @staticmethod
    def compute_fingerprint(
        checks: tuple[ModelContractCheckResult, ...],
    ) -> str:
        """Compute SHA-256 fingerprint from sorted check results.

        Args:
            checks: The check results to fingerprint.

        Returns:
            Hex-encoded SHA-256 digest of the sorted, serialized check results.
        """
        sorted_checks = sorted(
            checks,
            key=lambda c: (c.contract_name, c.check_type.value),
        )
        serialized = json.dumps(
            [c.model_dump(mode="json") for c in sorted_checks],
            sort_keys=True,
        )
        return hashlib.sha256(serialized.encode()).hexdigest()


__all__: list[str] = ["ModelContractVerificationReport"]
