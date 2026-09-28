# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed comparison key for delegation evidence cohorts (OMN-18930).

First written by a Codex session in omnibase_infra#3951; carried forward here
with a typed build identity and explicit provider-policy sources.
"""

from __future__ import annotations

import hashlib
import json
import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.models.delegation.model_delegation_build_identity import (
    ModelDelegationBuildIdentity,
)
from omnibase_infra.models.delegation.model_delegation_first_inference_identity import (
    ModelDelegationFirstInferenceIdentity,
)
from omnibase_infra.models.delegation.model_delegation_provider_policy import (
    ModelDelegationProviderPolicy,
)
from omnibase_infra.models.delegation.model_delegation_retry_bounds import (
    ModelDelegationRetryBounds,
)
from omnibase_infra.models.model_node_identity import ModelNodeIdentity

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ModelDelegationCohortKey(BaseModel):
    """Every dimension that must be equal before two run outcomes are compared.

    No field has a default. ``response_contract_sha256`` must be present, and an
    explicit ``None`` records that no response contract was conveyed.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    prompt_sha256: str = Field(min_length=64, max_length=64)
    resolved_task_type: str = Field(min_length=1)
    response_contract_sha256: str | None = Field()
    lane: str = Field(min_length=1)
    build_identity: ModelDelegationBuildIdentity
    consumer_identity: ModelNodeIdentity
    first_hop_identity: ModelDelegationFirstInferenceIdentity
    provider_policy: ModelDelegationProviderPolicy
    deadline_seconds: float = Field(gt=0)
    retry_bounds: ModelDelegationRetryBounds

    @model_validator(mode="after")
    def _validate_identity_hashes_and_dimensions(self) -> ModelDelegationCohortKey:
        if not _SHA256_PATTERN.fullmatch(self.prompt_sha256):
            raise ValueError("prompt_sha256 must be lowercase SHA-256 hex")
        if self.response_contract_sha256 is not None and not _SHA256_PATTERN.fullmatch(
            self.response_contract_sha256
        ):
            raise ValueError(
                "response_contract_sha256 must be lowercase SHA-256 hex or None"
            )
        for field_name in ("resolved_task_type", "lane"):
            value = getattr(self, field_name)
            if value != value.strip():
                raise ValueError(f"{field_name} must not have surrounding whitespace")
        return self

    @property
    def key_sha256(self) -> str:
        """Return a deterministic digest of the complete validated key."""
        canonical = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def changed_dimensions(self, other: ModelDelegationCohortKey) -> tuple[str, ...]:
        """Return the named dimensions that differ from another complete key."""
        return tuple(
            name
            for name in type(self).model_fields
            if getattr(self, name) != getattr(other, name)
        )

    def require_same_cohort(self, other: ModelDelegationCohortKey) -> None:
        """Refuse a comparison across any differing cohort dimension."""
        changed = self.changed_dimensions(other)
        if changed:
            raise ValueError(f"incomparable delegation cohorts: {', '.join(changed)}")
