# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The event that carries one canary receipt from the canary lane to the dev agent (OMN-26463)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_runtime_canary_receipt import (
    ModelRuntimeCanaryReceipt,
)

__all__: list[str] = ["ModelRuntimeCanaryReceiptEvent"]


class ModelRuntimeCanaryReceiptEvent(BaseModel):
    """One PASS or FAIL receipt, whole, so the reader needs nothing from the canary host."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    receipt: ModelRuntimeCanaryReceipt
