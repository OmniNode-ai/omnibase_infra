# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Payload model for runtime manifest INSERT intent (OMN-11197 / OMN-15512)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.event_bus.model_runtime_attach_readiness import (
    ModelRuntimeAttachReadiness,
)
from omnibase_infra.utils.util_pydantic_validators import (
    validate_timezone_aware_datetime,
)

if TYPE_CHECKING:
    from omnibase_infra.runtime.models.model_runtime_manifest_published import (
        ModelRuntimeManifestPublished,
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

    @classmethod
    def from_manifest_event(cls, event: ModelRuntimeManifestPublished) -> Self:
        """Fold a published runtime manifest into this INSERT payload.

        This is the step OMN-17296 section 2 found missing. The contract's
        ``intent_emission`` block declared the fold but has zero runtime
        consumers — no generic ``ModelReducerInput``/``ModelReducerOutput``
        adapter exists — so the event never became an intent payload and the
        handler was dispatched with an event it could not read.

        ``contract_hash`` and ``topology_hash`` are inherited computed fields on
        ``ModelRuntimeManifest`` and are copied, never recomputed here, so the
        row's dedup key matches the publisher's own values exactly.

        Args:
            event: The manifest snapshot published at boot on
                ``onex.evt.omnibase-infra.runtime-manifest-published.v1``.

        Returns:
            The INSERT payload for the ``runtime_manifests`` projection.
        """
        return cls(
            runtime_profile=event.runtime_profile,
            contract_hash=event.contract_hash,
            topology_hash=event.topology_hash,
            manifest_hash=cls.manifest_hash_for(event),
            contracts=[c.model_dump(mode="json") for c in event.contracts],
            # Sorted because the wire type is a frozenset: iteration order is not
            # stable across processes and the column would otherwise differ
            # between two rows describing the identical topology.
            owned_command_topics=sorted(event.owned_command_topics),
            subscribed_event_topics=sorted(event.subscribed_event_topics),
            handlers=[h.model_dump(mode="json") for h in event.handlers],
            skipped_contracts=[
                c.model_dump(mode="json") for c in event.skipped_contracts
            ],
            failed_contracts=[
                c.model_dump(mode="json") for c in event.failed_contracts
            ],
            ownership_violations=[
                v.model_dump(mode="json") for v in event.ownership_violations
            ],
            image_digest=event.image_digest,
            started_at=event.started_at,
            attach_readiness=event.attach_readiness,
        )

    @staticmethod
    def manifest_hash_for(event: ModelRuntimeManifestPublished) -> str:
        """Content hash of the WHOLE published manifest.

        Distinct from the two inherited hashes by scope, deliberately:
        ``contract_hash`` covers the contract set, ``topology_hash`` covers
        profile + contracts + topics + handler names, and this covers every
        published field including ``started_at``, ``image_digest`` and the
        attach-readiness aggregate. It is therefore unique per boot and is NOT
        part of the dedup key ``(runtime_profile, topology_hash, started_at)``;
        it exists so a stored row can be compared byte-for-byte against the
        payload that produced it.

        ``sort_keys`` makes it order-independent, so two runtimes publishing the
        same manifest produce the same hash.
        """
        return hashlib.sha256(
            json.dumps(event.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest()


__all__ = ["ModelPayloadInsertRuntimeManifest"]
