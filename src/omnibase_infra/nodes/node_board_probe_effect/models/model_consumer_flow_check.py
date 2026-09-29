# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One pure C28 clause expectation and its evidence."""

from pydantic import BaseModel, ConfigDict


class ModelConsumerFlowCheck(BaseModel):
    """Every failing check becomes one board-result reason."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    clause: str
    ok: bool
    evidence: str
