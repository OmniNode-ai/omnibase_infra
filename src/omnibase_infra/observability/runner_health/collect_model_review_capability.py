# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runner-local model-review capability probe and attestation boundary."""

from __future__ import annotations

from collections.abc import Callable, Collection
from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from omnibase_infra.observability.runner_health.model_model_review_capability_config import (
    ModelModelReviewCapabilityConfig,
)
from omnibase_infra.observability.runner_health.model_model_review_capability_observation import (
    MODEL_REVIEW_OBSERVATION_PROVENANCE,
    ModelModelReviewCapabilityObservation,
)


class ModelModelReviewReferenceProbe(BaseModel):
    """Result of probing one opaque reference in the local control plane."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    present: bool
    healthy: bool


def collect_model_review_capability_observation(
    config: ModelModelReviewCapabilityConfig,
    *,
    runner_labels: Collection[str],
    runner_groups: Collection[str],
    probe_reference: Callable[[UUID], ModelModelReviewReferenceProbe],
    probe_reviewer_cli: Callable[[], bool],
    now: datetime | None = None,
) -> ModelModelReviewCapabilityObservation:
    """Collect facts and derive an attestation from runner-local probes.

    Production callers bind ``probe_reference`` to the sanctioned private
    control-plane health probe and ``probe_reviewer_cli`` to the installed
    reviewer CLI. They return only presence/health booleans; no secret,
    endpoint, or runner identity is accepted by this boundary. The fixture
    canary exercises this collection shape, but preflight still rejects it
    because this slice has no live attestation verifier.
    """
    observed_at = now or datetime.now(UTC)
    if observed_at.tzinfo is None:
        raise ValueError("capability observation time must be timezone-aware")

    required_reference_ids = (
        config.credential_reference_id,
        config.endpoint_reference_id,
        config.healthcheck_reference_id,
    )
    results = {
        reference_id: probe_reference(reference_id)
        for reference_id in required_reference_ids
    }
    present_reference_ids = frozenset(
        reference_id for reference_id, result in results.items() if result.present
    )
    healthy_reference_ids = frozenset(
        reference_id
        for reference_id, result in results.items()
        if result.present and result.healthy
    )
    observation = ModelModelReviewCapabilityObservation(
        runner_labels=frozenset(runner_labels),
        runner_groups=frozenset(runner_groups),
        present_reference_ids=present_reference_ids,
        healthy_reference_ids=healthy_reference_ids,
        reviewer_cli_available=probe_reviewer_cli(),
        observed_at=observed_at,
        provenance=MODEL_REVIEW_OBSERVATION_PROVENANCE,
    )
    observation = observation.model_copy(
        update={"attestation_id": observation.derived_attestation_id()}
    )
    return observation


__all__ = [
    "ModelModelReviewReferenceProbe",
    "collect_model_review_capability_observation",
]
