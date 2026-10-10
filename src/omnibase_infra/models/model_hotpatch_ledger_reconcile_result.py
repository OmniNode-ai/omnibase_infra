# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Outcome of a hot-patch ledger row retirement (OMN-17427)."""

from pydantic import BaseModel, ConfigDict


class ModelHotpatchLedgerReconcileResult(BaseModel):
    """``success`` is true only when the row was retired and the ledger rewritten.

    A refusal carries its reason in both ``reason`` and ``error_message`` and
    leaves the ledger untouched.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    success: bool
    reason: str
    error_message: str | None = None
    backup_path: str = ""
    merge_commit: str = ""
    deployed_ref: str = ""
