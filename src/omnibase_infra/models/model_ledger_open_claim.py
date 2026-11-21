# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One open CLAIM of the rolling ledger, as the reader node reports it (OMN-20547)."""

from pydantic import BaseModel, ConfigDict


class ModelLedgerOpenClaim(BaseModel):
    """The claim's stamp and lane, and the text of its own CLAIM row."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    stamp: str
    lane: str
    row: str
