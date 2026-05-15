# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime-profile contract ownership filtering for auto-wiring.

Contracts may declare ``runtime_profiles`` to assign their Kafka subscriptions
to a specific runtime process. Unscoped legacy contracts default to ``main`` so
the effects/worker runtimes do not join general compute groups and steal work
from the primary runtime.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelDiscoveredContract,
)


def _normalize_runtime_profile(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("runtime_profile must be a string")
    profile = value.strip().lower()
    if not profile:
        raise ValueError("runtime_profile cannot be blank")
    return profile


class ModelRuntimeProfileOwnershipResult(BaseModel):
    """Result of filtering an auto-wiring manifest by runtime profile."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    runtime_profile: str = Field(..., min_length=1)
    manifest: ModelAutoWiringManifest
    skipped_contracts: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("runtime_profile", mode="before")
    @classmethod
    def normalize_runtime_profile(cls, value: object) -> str:
        return _normalize_runtime_profile(value)


def filter_manifest_for_runtime_profile(
    manifest: ModelAutoWiringManifest,
    runtime_profile: str,
) -> ModelRuntimeProfileOwnershipResult:
    """Return a manifest containing only contracts owned by runtime_profile.

    Contracts without ``runtime_profiles`` default to ``main`` ownership.
    Contracts with an explicit list are wired only by profiles named in that
    list.
    """
    normalized_profile = _normalize_runtime_profile(runtime_profile)
    owned_contracts: list[ModelDiscoveredContract] = []
    skipped_contracts: list[str] = []

    for contract in manifest.contracts:
        if contract.runtime_profiles:
            if normalized_profile in contract.runtime_profiles:
                owned_contracts.append(contract)
            else:
                skipped_contracts.append(contract.name)
            continue

        if normalized_profile != "main":
            skipped_contracts.append(contract.name)
            continue
        owned_contracts.append(contract)

    return ModelRuntimeProfileOwnershipResult(
        runtime_profile=normalized_profile,
        manifest=ModelAutoWiringManifest(
            contracts=tuple(owned_contracts),
            errors=manifest.errors,
        ),
        skipped_contracts=tuple(skipped_contracts),
    )


__all__ = [
    "ModelRuntimeProfileOwnershipResult",
    "filter_manifest_for_runtime_profile",
]
