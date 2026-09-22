# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Canonical provider/escalation policy identity."""

from __future__ import annotations

import hashlib
import json
import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ModelDelegationProviderPolicy(BaseModel):
    """Canonical provider/escalation policy identity from terminal provenance."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    routing_tiers_sha256: str = Field(min_length=64, max_length=64)
    escalation_config_sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _validate_source_hashes(self) -> ModelDelegationProviderPolicy:
        for name in ("routing_tiers_sha256", "escalation_config_sha256"):
            if not _SHA256_PATTERN.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be lowercase SHA-256 hex")
        return self

    @property
    def sha256(self) -> str:
        """Hash the named source hashes in a stable, explicit serialization."""
        canonical = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
