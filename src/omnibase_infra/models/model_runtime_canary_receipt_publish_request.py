# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Which candidate's receipt the canary lane publishes, and where its run wrote it (OMN-26463)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_runtime_canary_candidate import (
    ModelRuntimeCanaryCandidate,
)

__all__: list[str] = ["ModelRuntimeCanaryReceiptPublishRequest"]


class ModelRuntimeCanaryReceiptPublishRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate: ModelRuntimeCanaryCandidate
    receipt_dir: Path
