# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""One evaluated check.

Ticket: OMN-19572
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.lab_proof.enum_lab_proof_check import EnumLabProofCheck


class ModelLabProofCheckResult(BaseModel):
    """A named check, whether it held, and the evidence line behind it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    check: EnumLabProofCheck
    passed: bool
    detail: str = Field(min_length=1)
    evidence_items: tuple[str, ...] = Field(
        default=(),
        description="The distinct failing items behind a set-valued check (for "
        "no_wiring_failures, the contracts that failed to wire), so a base control "
        "can be compared item by item.",
    )


__all__ = ["ModelLabProofCheckResult"]
