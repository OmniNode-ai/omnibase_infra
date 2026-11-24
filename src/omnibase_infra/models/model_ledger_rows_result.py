# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The rows of a window, oldest first, as the ledger reader node answered them (OMN-26507)."""

from pydantic import BaseModel, ConfigDict


class ModelLedgerRowsResult(BaseModel):
    """Empty when the window holds no row of the kinds; a read that failed raises instead."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    rows: tuple[str, ...]
