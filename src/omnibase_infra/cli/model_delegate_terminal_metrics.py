# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The metrics block of a delegation terminal, narrowed to what is written (OMN-18569).

Only the field the customer artifacts actually carry is mirrored. Widening this
means a customer-visible field was added on purpose, which is the right size of
decision for a receipt.

.. versionadded:: OMN-18569
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelDelegateTerminalMetrics"]


class ModelDelegateTerminalMetrics(BaseModel):
    """What a delegation run cost, as recorded on its terminal."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    cost_usd: float | None = Field(default=None)
