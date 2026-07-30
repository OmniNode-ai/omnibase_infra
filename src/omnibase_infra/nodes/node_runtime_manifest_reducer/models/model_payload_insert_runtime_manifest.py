# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Payload model for runtime manifest INSERT intent (OMN-11197 / OMN-15512)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.event_bus.model_runtime_attach_readiness import (
    ModelRuntimeAttachReadiness,
)
from omnibase_infra.utils.util_pydantic_validators import (
    validate_timezone_aware_datetime,
)


class ModelPayloadInsertRuntimeManifest(BaseModel):
    """Typed payload for inserting a runtime manifest row into PostgreSQL.

    This intent is emitted by the runtime manifest reducer when it receives
    a runtime-manifest-published event. The handler performs an INSERT only —
    the unique index on (runtime_profile, topology_hash, started_at) handles
    deduplication at the database layer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent_type: Literal["postgres.insert_runtime_manifest"] = Field(
        default="postgres.insert_runtime_manifest"
    )
    runtime_profile: str = Field(..., min_length=1)
    contract_hash: str = Field(..., min_length=1)
    topology_hash: str = Field(..., min_length=1)
    manifest_hash: str = Field(..., min_length=1)
    contracts: list[dict[str, object]] = Field(default_factory=list)
    owned_command_topics: list[str] = Field(default_factory=list)
    subscribed_event_topics: list[str] = Field(default_factory=list)
    handlers: list[dict[str, object]] = Field(default_factory=list)
    skipped_contracts: list[dict[str, object]] = Field(default_factory=list)
    failed_contracts: list[dict[str, object]] = Field(default_factory=list)
    ownership_violations: list[dict[str, object]] = Field(default_factory=list)
    image_digest: str | None = Field(default=None)
    started_at: datetime = Field(...)
    # OMN-15512: boot attach-readiness aggregate, carried on the SAME
    # runtime-manifest-published envelope. The producer narrows `results` to
    # the blocker set (ModelRuntimeAttachReadiness.blockers_only), so this is
    # bounded by the not-attached count, not by the 475+ contracts walked at
    # boot. The handler flattens it across four runtime_manifests columns —
    # this model is `extra="forbid"`, so the producer key name and this field
    # name are a matched seam and drift fails loudly at coercion.
    attach_readiness: ModelRuntimeAttachReadiness | None = Field(default=None)

    @field_validator("started_at")
    @classmethod
    def _validate_started_at_tz(cls, v: datetime) -> datetime:
        return validate_timezone_aware_datetime(v)


__all__ = ["ModelPayloadInsertRuntimeManifest"]
