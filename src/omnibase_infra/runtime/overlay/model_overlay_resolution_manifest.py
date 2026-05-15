# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Stub ModelOverlayResolutionManifest — replaced by omnibase_core.models.overlay once Task 2 merges."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelOverlayResolutionManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    overlay_file_hash: str = Field(...)
    overlay_version: str = Field(...)
    overlay_scope_stack: tuple[str, ...] = Field(...)
    contract_requirements_hash: str = Field(...)
    resolved_config_hash: str = Field(...)
    resolved_transports: tuple[str, ...] = Field(...)
    required_transports: tuple[str, ...] = Field(...)
    runtime_version: str = Field(...)
    timestamp: datetime = Field(...)
    config_source: str = Field(...)

    def stable_identity_hash(self) -> str:
        stable = {
            "overlay_file_hash": self.overlay_file_hash,
            "contract_requirements_hash": self.contract_requirements_hash,
            "resolved_config_hash": self.resolved_config_hash,
            "resolved_transports": sorted(self.resolved_transports),
        }
        return f"sha256:{hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()}"

    def model_dump_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent)
