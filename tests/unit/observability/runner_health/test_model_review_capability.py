# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the fail-closed model-review runner capability preflight."""

from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from omnibase_infra.observability.runner_health.enum_model_review_capability_failure import (
    EnumModelReviewCapabilityFailure,
)
from omnibase_infra.observability.runner_health.model_model_review_capability_config import (
    ModelModelReviewCapabilityConfig,
)
from omnibase_infra.observability.runner_health.model_model_review_capability_observation import (
    ModelModelReviewCapabilityObservation,
)
from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    load_runner_fleet_config,
)
from omnibase_infra.observability.runner_health.preflight_model_review_capability import (
    preflight_model_review_capability,
)


def _active_config() -> ModelModelReviewCapabilityConfig:
    return ModelModelReviewCapabilityConfig(
        active=True,
        credential_reference_id="dc9565c8-7f13-46dc-bd89-9694c13e1d2f",
        endpoint_reference_id="b2a8287b-0a9f-4cbc-b2e8-cf954f9a71f7",
        healthcheck_reference_id="2672472a-bac9-4344-8c8c-79da6cb604ae",
    )


def test_repo_config_records_an_inert_model_review_capability() -> None:
    config = load_runner_fleet_config()

    assert config.model_review is not None
    assert config.model_review.active is False
    assert config.model_review.runner_label == "model-review"


def test_preflight_fails_closed_when_config_is_inactive() -> None:
    config = _active_config().model_copy(update={"active": False})
    observation = ModelModelReviewCapabilityObservation(
        runner_labels=frozenset({"model-review"}),
        present_reference_ids=frozenset(
            {
                config.credential_reference_id,
                config.endpoint_reference_id,
                config.healthcheck_reference_id,
            }
        ),
        healthy_reference_ids=frozenset({config.healthcheck_reference_id}),
    )

    result = preflight_model_review_capability(config, observation)

    assert result.ready is False
    assert result.failures == (EnumModelReviewCapabilityFailure.CONFIG_INACTIVE,)


def test_preflight_fails_closed_when_config_is_absent() -> None:
    result = preflight_model_review_capability(
        None,
        ModelModelReviewCapabilityObservation(),
    )

    assert result.ready is False
    assert result.failures == (EnumModelReviewCapabilityFailure.CONFIG_ABSENT,)


def test_preflight_requires_label_references_and_health_assertion() -> None:
    config = _active_config()

    result = preflight_model_review_capability(
        config,
        ModelModelReviewCapabilityObservation(),
    )

    assert result.ready is False
    assert result.failures == (
        EnumModelReviewCapabilityFailure.REQUIRED_LABEL_MISSING,
        EnumModelReviewCapabilityFailure.REQUIRED_REFERENCE_MISSING,
    )
    assert result.missing_reference_ids == tuple(
        sorted(
            (
                config.credential_reference_id,
                config.endpoint_reference_id,
                config.healthcheck_reference_id,
            )
        )
    )


def test_preflight_rejects_unhealthy_health_assertion() -> None:
    config = _active_config()
    present_reference_ids = frozenset(
        (
            config.credential_reference_id,
            config.endpoint_reference_id,
            config.healthcheck_reference_id,
        )
    )

    result = preflight_model_review_capability(
        config,
        ModelModelReviewCapabilityObservation(
            runner_labels=frozenset({"model-review"}),
            present_reference_ids=present_reference_ids,
        ),
    )

    assert result.ready is False
    assert result.failures == (
        EnumModelReviewCapabilityFailure.HEALTH_ASSERTION_MISSING,
    )


def test_preflight_accepts_only_complete_healthy_observation() -> None:
    config = _active_config()
    present_reference_ids = frozenset(
        (
            config.credential_reference_id,
            config.endpoint_reference_id,
            config.healthcheck_reference_id,
        )
    )

    result = preflight_model_review_capability(
        config,
        ModelModelReviewCapabilityObservation(
            runner_labels=frozenset({"model-review"}),
            present_reference_ids=present_reference_ids,
            healthy_reference_ids=frozenset({config.healthcheck_reference_id}),
        ),
    )

    assert result.ready is True
    assert result.failures == ()
    assert result.missing_reference_ids == ()


def test_observation_rejects_health_assertion_for_absent_reference() -> None:
    with pytest.raises(ValidationError, match="must be a subset"):
        ModelModelReviewCapabilityObservation(
            healthy_reference_ids=frozenset(
                {UUID("2672472a-bac9-4344-8c8c-79da6cb604ae")}
            ),
        )


def test_config_rejects_a_non_opaque_reference_identifier() -> None:
    with pytest.raises(ValidationError):
        ModelModelReviewCapabilityConfig(
            credential_reference_id="not-a-uuid",
            endpoint_reference_id="b2a8287b-0a9f-4cbc-b2e8-cf954f9a71f7",
            healthcheck_reference_id="2672472a-bac9-4344-8c8c-79da6cb604ae",
        )


def test_config_requires_distinct_credential_endpoint_and_health_references() -> None:
    with pytest.raises(ValidationError, match="must differ"):
        ModelModelReviewCapabilityConfig(
            credential_reference_id="dc9565c8-7f13-46dc-bd89-9694c13e1d2f",
            endpoint_reference_id="dc9565c8-7f13-46dc-bd89-9694c13e1d2f",
            healthcheck_reference_id="2672472a-bac9-4344-8c8c-79da6cb604ae",
        )
