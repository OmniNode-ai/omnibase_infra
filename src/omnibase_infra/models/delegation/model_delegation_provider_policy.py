# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Provider and escalation policy identity of one delegation consumer (OMN-18930)."""

from __future__ import annotations

import hashlib
import json
import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ModelDelegationProviderPolicy(BaseModel):
    """Content hashes of the policy sources the consumer resolved routes from.

    Each field is the SHA-256 of one file's bytes as the consumer read it:

    * ``routing_tiers_sha256`` -- the routing tiers (``routing_tiers.yaml``);
    * ``backend_config_sha256`` -- the backend and retry declarations
      (``bifrost_delegation.yaml``);
    * ``task_class_contracts_sha256`` -- the per-task-class escalation policy
      (``task_class_contracts.v1.yaml``);
    * ``overlay_sha256`` -- the bound lane overlay, or an explicit ``None``
      when the consumer bound no overlay.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    routing_tiers_sha256: str = Field(min_length=64, max_length=64)
    backend_config_sha256: str = Field(min_length=64, max_length=64)
    task_class_contracts_sha256: str = Field(min_length=64, max_length=64)
    overlay_sha256: str | None = Field()

    @model_validator(mode="after")
    def _validate_source_hashes(self) -> ModelDelegationProviderPolicy:
        for name in (
            "routing_tiers_sha256",
            "backend_config_sha256",
            "task_class_contracts_sha256",
        ):
            if not _SHA256_PATTERN.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be lowercase SHA-256 hex")
        if self.overlay_sha256 is not None and not _SHA256_PATTERN.fullmatch(
            self.overlay_sha256
        ):
            raise ValueError("overlay_sha256 must be lowercase SHA-256 hex or None")
        return self

    @property
    def sha256(self) -> str:
        """Hash the named source hashes in a stable, explicit serialization."""
        canonical = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
