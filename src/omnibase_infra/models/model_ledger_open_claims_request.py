# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request for the claims open in a window, read through the ledger reader node (OMN-20547)."""

from pydantic import BaseModel, ConfigDict, Field


class ModelLedgerOpenClaimsRequest(BaseModel):
    """Name the reader command, the reader and the window; none of them has a default.

    ``reader_command`` is the interpreter and module that run the reader node, one shell-quoted
    string; the caller owns it, so this package names no other node's module.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    reader_command: str = Field(min_length=1)
    reader: str = Field(min_length=1)
    window_hours: int = Field(gt=0)
