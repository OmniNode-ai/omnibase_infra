# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request for the contracts-revision guard (OMN-25856)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelContractsRevisionRequest(BaseModel):
    """Where the image's build-time manifest is and which contracts to judge."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    stamp_path: str
    runtime_contracts_dir: str
