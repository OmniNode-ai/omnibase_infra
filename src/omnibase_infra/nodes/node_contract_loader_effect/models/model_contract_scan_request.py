# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request for the contract filesystem scan handler."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelContractScanRequest(BaseModel):
    """Request to scan directories for ONEX contract YAML files."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    scan_paths: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Override paths to scan; when empty the handler's default scan path is used"
        ),
    )


__all__ = ["ModelContractScanRequest"]
