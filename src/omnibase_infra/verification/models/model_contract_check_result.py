# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Per-check result for a single runtime contract verification probe."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_check_severity import EnumCheckSeverity
from omnibase_infra.enums.enum_contract_check_type import EnumContractCheckType
from omnibase_infra.enums.enum_validation_verdict import EnumValidationVerdict


class ModelContractCheckResult(BaseModel):
    """Result of a single runtime verification check against a contract.

    Each probe (subscription, publication, projection, etc.) produces one or more
    of these results. The ``verdict`` indicates whether the check passed, failed,
    or could not be determined (quarantine).

    Attributes:
        check_type: The class of verification check performed.
        severity: Whether this check is required, recommended, or informational.
        verdict: The outcome of the check (PASS, FAIL, QUARANTINE).
        evidence: Human-readable evidence supporting the verdict.
        contract_name: The name of the contract being verified.
        message: Short summary message for display.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    check_type: EnumContractCheckType = Field(
        ...,
        description="The class of verification check performed.",
    )
    severity: EnumCheckSeverity = Field(
        ...,
        description="Whether this check is required, recommended, or informational.",
    )
    verdict: EnumValidationVerdict = Field(
        ...,
        description="The outcome of the check (PASS, FAIL, QUARANTINE).",
    )
    evidence: str = Field(
        ...,
        description="Human-readable evidence supporting the verdict.",
    )
    contract_name: str = Field(
        ...,
        description="The name of the contract being verified.",
    )
    message: str = Field(
        ...,
        description="Short summary message for display.",
    )


__all__: list[str] = ["ModelContractCheckResult"]
