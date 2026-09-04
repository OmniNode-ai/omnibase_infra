# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure preflight for the model-review runner-overlay contract."""

from __future__ import annotations

from datetime import UTC, datetime

from omnibase_infra.observability.runner_health.enum_model_review_capability_failure import (
    EnumModelReviewCapabilityFailure,
)
from omnibase_infra.observability.runner_health.model_model_review_capability_config import (
    ModelModelReviewCapabilityConfig,
)
from omnibase_infra.observability.runner_health.model_model_review_capability_observation import (
    MODEL_REVIEW_OBSERVATION_PROVENANCE,
    ModelModelReviewCapabilityObservation,
)
from omnibase_infra.observability.runner_health.model_model_review_capability_preflight import (
    ModelModelReviewCapabilityPreflight,
)


def preflight_model_review_capability(
    config: ModelModelReviewCapabilityConfig | None,
    observation: ModelModelReviewCapabilityObservation,
    *,
    now: datetime | None = None,
) -> ModelModelReviewCapabilityPreflight:
    """Return eligibility without resolving or exposing any sensitive value.

    Inactive or absent config, a missing capability label, an absent required
    reference, or a missing health assertion each produce a non-ready result.
    No caller may infer readiness from one of the other facts.
    """
    if config is None:
        return ModelModelReviewCapabilityPreflight(
            ready=False,
            failures=(EnumModelReviewCapabilityFailure.CONFIG_ABSENT,),
        )
    if not config.active:
        return ModelModelReviewCapabilityPreflight(
            ready=False,
            failures=(EnumModelReviewCapabilityFailure.CONFIG_INACTIVE,),
        )

    required_reference_ids = frozenset(
        (
            config.credential_reference_id,
            config.endpoint_reference_id,
            config.healthcheck_reference_id,
        )
    )
    missing_reference_ids = tuple(
        sorted(required_reference_ids - observation.present_reference_ids)
    )
    unexpected_reference_ids = (
        observation.present_reference_ids - required_reference_ids
    )
    failures: list[EnumModelReviewCapabilityFailure] = []
    if config.runner_label not in observation.runner_labels:
        failures.append(EnumModelReviewCapabilityFailure.REQUIRED_LABEL_MISSING)
    if config.runner_group not in observation.runner_groups:
        failures.append(EnumModelReviewCapabilityFailure.REQUIRED_GROUP_MISSING)
    if missing_reference_ids:
        failures.append(EnumModelReviewCapabilityFailure.REQUIRED_REFERENCE_MISSING)
    if unexpected_reference_ids:
        failures.append(EnumModelReviewCapabilityFailure.UNEXPECTED_REFERENCE)
    unhealthy_reference_ids = required_reference_ids - observation.healthy_reference_ids
    if unhealthy_reference_ids:
        failures.append(EnumModelReviewCapabilityFailure.HEALTH_ASSERTION_MISSING)
    if (
        observation.provenance != MODEL_REVIEW_OBSERVATION_PROVENANCE
        or observation.attestation_id is None
    ):
        failures.append(EnumModelReviewCapabilityFailure.PROVENANCE_MISSING)
    elif observation.attestation_id != observation.derived_attestation_id():
        failures.append(EnumModelReviewCapabilityFailure.ATTESTATION_INVALID)
    # OMN-17876 deliberately ships no verifier implementation. UUIDv5 is only
    # a correlation identity; a future sanctioned verifier/receipt contract
    # must be added by the operator rollout before this can ever be ready.
    failures.append(EnumModelReviewCapabilityFailure.LIVE_ATTESTATION_UNAVAILABLE)
    if observation.observed_at is None:
        failures.append(EnumModelReviewCapabilityFailure.OBSERVATION_STALE)
    else:
        observed_at = observation.observed_at
        current_time = now or datetime.now(UTC)
        if observed_at.tzinfo is None or current_time.tzinfo is None:
            failures.append(EnumModelReviewCapabilityFailure.OBSERVATION_STALE)
        else:
            age_seconds = (current_time - observed_at).total_seconds()
            if age_seconds < 0 or age_seconds > config.max_observation_age_seconds:
                failures.append(EnumModelReviewCapabilityFailure.OBSERVATION_STALE)
    if not observation.reviewer_cli_available:
        failures.append(EnumModelReviewCapabilityFailure.REVIEWER_CLI_UNAVAILABLE)

    return ModelModelReviewCapabilityPreflight(
        ready=not failures,
        failures=tuple(failures),
        missing_reference_ids=missing_reference_ids,
    )


__all__ = ["preflight_model_review_capability"]
