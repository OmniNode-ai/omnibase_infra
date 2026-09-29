# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed comparison key for delegation evidence cohorts (OMN-18930)."""

from __future__ import annotations

import hashlib
import json
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_infra.models.model_node_identity import ModelNodeIdentity

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ModelDelegationTierRetryBound(BaseModel):
    """One immutable retry ceiling declared for a routing tier."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    tier: str = Field(min_length=1)
    max_retries: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_tier(self) -> ModelDelegationTierRetryBound:
        if not self.tier.strip() or self.tier != self.tier.strip():
            raise ValueError("retry tier name must be nonblank and trimmed")
        return self


class ModelDelegationRetryBounds(BaseModel):
    """The distinct retry ceilings declared by routing and task contracts."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    per_tier: tuple[ModelDelegationTierRetryBound, ...] = Field(min_length=1)
    max_escalations: int = Field(ge=0)

    @field_validator("per_tier", mode="before")
    @classmethod
    def _sort_tier_bounds(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            return tuple(
                sorted(
                    value,
                    key=lambda item: (
                        item.tier
                        if isinstance(item, ModelDelegationTierRetryBound)
                        else item.get("tier", "")
                        if isinstance(item, dict)
                        else ""
                    ),
                )
            )
        return value

    @model_validator(mode="after")
    def _validate_tier_retry_limits(self) -> ModelDelegationRetryBounds:
        tiers = [item.tier for item in self.per_tier]
        if len(tiers) != len(set(tiers)):
            raise ValueError("retry tier names must be unique")
        return self


class ModelDelegationFirstInferenceIdentity(BaseModel):
    """Identity of the actual first inference rung, from its attempt record.

    This is deliberately distinct from the deployed consumer's node identity.
    The canonical caller-visible attempt row carries backend, model, and tier;
    provider is nullable but required so absence is explicit rather than silently
    filled from a top-level final provider or a later routing configuration.
    With multiple attempts, use the first attempt's own provider only; if it is
    not captured, preserve ``None`` and record the provenance limitation.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    backend_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    tier: str = Field(min_length=1)
    provider: str | None = Field()

    @model_validator(mode="after")
    def _validate_identity_values(
        self,
    ) -> ModelDelegationFirstInferenceIdentity:
        for name in ("backend_id", "model_id", "tier"):
            value = getattr(self, name)
            if not value.strip() or value != value.strip():
                raise ValueError(f"{name} must be nonblank and trimmed")
        if self.provider is not None and (
            not self.provider.strip() or self.provider != self.provider.strip()
        ):
            raise ValueError("provider must be nonblank and trimmed when present")
        return self


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


class ModelDelegationCohortKey(BaseModel):
    """All declared dimensions that must match before comparing run outcomes.

    This model records identifiers supplied by request, runtime, and route
    evidence. It does not assert that a caller-provided value is authoritative;
    callers must populate ``consumer_identity`` from observed consumer evidence.
    ``response_contract_sha256`` is required even when its value is ``None`` so
    an absent contract cannot be confused with missing evidence.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    prompt_sha256: str = Field(min_length=64, max_length=64)
    resolved_task_type: str = Field(min_length=1)
    response_contract_sha256: str | None = Field()
    lane: str = Field(min_length=1)
    build_identity: str = Field(min_length=1)
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
        for field_name in (
            "resolved_task_type",
            "lane",
            "build_identity",
        ):
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
        """Reject comparisons across any differing cohort dimension."""
        changed = self.changed_dimensions(other)
        if changed:
            raise ValueError(f"incomparable delegation cohorts: {', '.join(changed)}")


__all__ = [
    "ModelDelegationCohortKey",
    "ModelDelegationFirstInferenceIdentity",
    "ModelDelegationProviderPolicy",
    "ModelDelegationRetryBounds",
    "ModelDelegationTierRetryBound",
]
