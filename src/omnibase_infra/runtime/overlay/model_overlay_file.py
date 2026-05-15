# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Stub ModelOverlayFile — replaced by omnibase_core.models.overlay.model_overlay_file once Task 1 merges."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field, field_validator

_SUPPORTED_VERSIONS = frozenset({"1.0.0"})


class ModelOverlayFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    overlay_version: str = Field(..., min_length=1)
    environment: str = Field(..., min_length=1)
    scope: str = Field(...)
    transports: dict[str, dict[str, str]] = Field(default_factory=dict)
    secrets: dict[str, str] = Field(default_factory=dict)
    services: dict[str, dict[str, str]] = Field(default_factory=dict)
    llm: dict[str, dict[str, str]] = Field(default_factory=dict)

    @field_validator("overlay_version")
    @classmethod
    def _check_version(cls, v: str) -> str:
        if v not in _SUPPORTED_VERSIONS:
            msg = f"overlay_version '{v}' not supported; supported: {sorted(_SUPPORTED_VERSIONS)}"
            raise ValueError(msg)
        return v

    def content_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"

    def all_env_pairs(self) -> dict[str, str]:
        pairs: dict[str, str] = {}
        for section_vals in self.transports.values():
            pairs.update(section_vals)
        pairs.update(self.secrets)
        for section_vals in self.services.values():
            pairs.update(section_vals)
        for section_vals in self.llm.values():
            pairs.update(section_vals)
        return pairs
