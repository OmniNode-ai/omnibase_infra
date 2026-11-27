# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request for the rows of one window and some row kinds, read through the ledger reader node (OMN-26507)."""

from pydantic import BaseModel, ConfigDict, Field


class ModelLedgerRowsRequest(BaseModel):
    """Name the reader command, the reader, the window and the row kinds; none has a default.

    ``reader_command`` is the interpreter and module that run the reader node, one shell-quoted
    string; the caller owns it, so this package names no other node's module. ``kinds`` are the
    ledger row types to keep (HOLD, RELEASE, CLAIM ...).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    reader_command: str = Field(min_length=1)
    reader: str = Field(min_length=1)
    window_hours: int = Field(gt=0)
    kinds: tuple[str, ...] = Field(min_length=1)
