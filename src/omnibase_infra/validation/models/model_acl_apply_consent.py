# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The durable operator-consent row a live privilege change is authorized by."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelAclApplyConsent(BaseModel):
    """One resolved OPERATOR-CONSENT ledger row, both scope halves included."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ledger_path: str
    line_number: int = Field(..., ge=1)
    lane: str
    approved_by: str
    recorded_at: datetime
    approved_scope: str
    out_of_scope: str

    @property
    def citation(self) -> str:
        """Render the citation exactly as a command line carries it."""
        return f"{self.ledger_path}:{self.line_number}"


__all__ = ["ModelAclApplyConsent"]
