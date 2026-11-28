# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Where the dev agent recorded a delivered canary receipt (OMN-26463)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums.enum_runtime_canary_verdict import EnumRuntimeCanaryVerdict

__all__: list[str] = ["ModelRuntimeCanaryReceiptRecorded"]


class ModelRuntimeCanaryReceiptRecorded(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    receipt_path: str
    verdict: EnumRuntimeCanaryVerdict
