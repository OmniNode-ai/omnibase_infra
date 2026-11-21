# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The claims open in the window, oldest first, each with its CLAIM row (OMN-20547)."""

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_ledger_open_claim import ModelLedgerOpenClaim


class ModelLedgerOpenClaimsResult(BaseModel):
    """Empty when nothing is open; a read that failed raises instead and is never empty."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    claims: tuple[ModelLedgerOpenClaim, ...]
