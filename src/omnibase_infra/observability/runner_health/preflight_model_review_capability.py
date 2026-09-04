# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure preflight for the model-review runner-overlay contract."""

from __future__ import annotations

from omnibase_infra.observability.runner_health.enum_model_review_capability_failure import (
    EnumModelReviewCapabilityFailure,
)
from omnibase_infra.observability.runner_health.model_model_review_capability_config import (
    ModelModelReviewCapabilityConfig,
)
from omnibase_infra.observability.runner_health.model_model_review_capability_observation import (
    ModelModelReviewCapabilityObservation,
)
from omnibase_infra.observability.runner_health.model_model_review_capability_preflight import (
    ModelModelReviewCapabilityPreflight,
)


def preflight_model_review_capability(
    config: ModelModelReviewCapabilityConfig | None,
    observation: ModelModelReviewCapabilityObservation,
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
    failures: list[EnumModelReviewCapabilityFailure] = []
    if config.runner_label not in observation.runner_labels:
        failures.append(EnumModelReviewCapabilityFailure.REQUIRED_LABEL_MISSING)
    if missing_reference_ids:
        failures.append(EnumModelReviewCapabilityFailure.REQUIRED_REFERENCE_MISSING)
    if (
        config.healthcheck_reference_id in observation.present_reference_ids
        and config.healthcheck_reference_id not in observation.healthy_reference_ids
    ):
        failures.append(EnumModelReviewCapabilityFailure.HEALTH_ASSERTION_MISSING)

    return ModelModelReviewCapabilityPreflight(
        ready=not failures,
        failures=tuple(failures),
        missing_reference_ids=missing_reference_ids,
    )


__all__ = ["preflight_model_review_capability"]
