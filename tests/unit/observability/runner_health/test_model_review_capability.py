# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the fail-closed model-review runner capability preflight."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from pydantic import ValidationError

from omnibase_infra.observability.runner_health.collect_model_review_capability import (
    ModelModelReviewReferenceProbe,
    collect_model_review_capability_observation,
)
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

OBSERVED_AT = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)


def _active_config() -> ModelModelReviewCapabilityConfig:
    return ModelModelReviewCapabilityConfig(
        active=True,
        credential_reference_id="dc9565c8-7f13-46dc-bd89-9694c13e1d2f",
        endpoint_reference_id="b2a8287b-0a9f-4cbc-b2e8-cf954f9a71f7",
        healthcheck_reference_id="2672472a-bac9-4344-8c8c-79da6cb604ae",
    )


def _healthy_observation(
    config: ModelModelReviewCapabilityConfig,
    *,
    observed_at: datetime = OBSERVED_AT,
    reviewer_cli_available: bool = True,
) -> ModelModelReviewCapabilityObservation:
    return collect_model_review_capability_observation(
        config,
        runner_labels={config.runner_label},
        runner_groups={config.runner_group},
        probe_reference=lambda _reference_id: ModelModelReviewReferenceProbe(
            present=True,
            healthy=True,
        ),
        probe_reviewer_cli=lambda: reviewer_cli_available,
        now=observed_at,
    )


def test_repo_config_records_an_inert_model_review_capability() -> None:
    config = load_runner_fleet_config()

    assert config.model_review is not None
    assert config.model_review.active is False
    assert config.model_review.runner_label == "model-review"
    assert config.model_review.runner_group == "omnibase-ci"
    assert config.model_review.max_observation_age_seconds == 300


def test_preflight_fails_closed_when_config_is_inactive() -> None:
    config = _active_config().model_copy(update={"active": False})
    result = preflight_model_review_capability(
        config,
        ModelModelReviewCapabilityObservation(),
    )

    assert result.ready is False
    assert result.failures == (EnumModelReviewCapabilityFailure.CONFIG_INACTIVE,)


def test_preflight_fails_closed_when_config_is_absent() -> None:
    result = preflight_model_review_capability(
        None,
        ModelModelReviewCapabilityObservation(),
    )

    assert result.ready is False
    assert result.failures == (EnumModelReviewCapabilityFailure.CONFIG_ABSENT,)


def test_preflight_requires_all_authoritative_observation_facts() -> None:
    config = _active_config()

    result = preflight_model_review_capability(
        config,
        ModelModelReviewCapabilityObservation(),
        now=OBSERVED_AT,
    )

    assert result.ready is False
    assert result.failures == (
        EnumModelReviewCapabilityFailure.REQUIRED_LABEL_MISSING,
        EnumModelReviewCapabilityFailure.REQUIRED_GROUP_MISSING,
        EnumModelReviewCapabilityFailure.REQUIRED_REFERENCE_MISSING,
        EnumModelReviewCapabilityFailure.HEALTH_ASSERTION_MISSING,
        EnumModelReviewCapabilityFailure.PROVENANCE_MISSING,
        EnumModelReviewCapabilityFailure.LIVE_ATTESTATION_UNAVAILABLE,
        EnumModelReviewCapabilityFailure.OBSERVATION_STALE,
        EnumModelReviewCapabilityFailure.REVIEWER_CLI_UNAVAILABLE,
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
    result = preflight_model_review_capability(
        config,
        collect_model_review_capability_observation(
            config,
            runner_labels={"model-review"},
            runner_groups={"omnibase-ci"},
            probe_reference=lambda reference_id: ModelModelReviewReferenceProbe(
                present=True,
                healthy=reference_id != config.healthcheck_reference_id,
            ),
            probe_reviewer_cli=lambda: True,
            now=OBSERVED_AT,
        ),
        now=OBSERVED_AT,
    )

    assert result.ready is False
    assert result.failures == (
        EnumModelReviewCapabilityFailure.HEALTH_ASSERTION_MISSING,
        EnumModelReviewCapabilityFailure.LIVE_ATTESTATION_UNAVAILABLE,
    )


def test_preflight_keeps_complete_observation_not_ready_without_live_verifier() -> None:
    config = _active_config()

    result = preflight_model_review_capability(
        config,
        _healthy_observation(config),
        now=OBSERVED_AT,
    )

    assert result.ready is False
    assert result.failures == (
        EnumModelReviewCapabilityFailure.LIVE_ATTESTATION_UNAVAILABLE,
    )
    assert result.missing_reference_ids == ()


def test_preflight_rejects_stale_attestation() -> None:
    config = _active_config()
    result = preflight_model_review_capability(
        config,
        _healthy_observation(
            config,
            observed_at=OBSERVED_AT
            - timedelta(seconds=config.max_observation_age_seconds + 1),
        ),
        now=OBSERVED_AT,
    )

    assert result.ready is False
    assert result.failures == (
        EnumModelReviewCapabilityFailure.LIVE_ATTESTATION_UNAVAILABLE,
        EnumModelReviewCapabilityFailure.OBSERVATION_STALE,
    )


def test_preflight_rejects_missing_cli_capability() -> None:
    config = _active_config()
    result = preflight_model_review_capability(
        config,
        _healthy_observation(config, reviewer_cli_available=False),
        now=OBSERVED_AT,
    )

    assert result.ready is False
    assert result.failures == (
        EnumModelReviewCapabilityFailure.LIVE_ATTESTATION_UNAVAILABLE,
        EnumModelReviewCapabilityFailure.REVIEWER_CLI_UNAVAILABLE,
    )


def test_preflight_rejects_detached_attestation() -> None:
    config = _active_config()
    observation = _healthy_observation(config).model_copy(
        update={"attestation_id": UUID("f3df2b7f-e8d5-41f0-8d6b-df65ea5c8ae4")}
    )

    result = preflight_model_review_capability(
        config,
        observation,
        now=OBSERVED_AT,
    )

    assert result.ready is False
    assert result.failures == (
        EnumModelReviewCapabilityFailure.ATTESTATION_INVALID,
        EnumModelReviewCapabilityFailure.LIVE_ATTESTATION_UNAVAILABLE,
    )


def test_collector_requires_each_reference_probe_to_be_healthy() -> None:
    config = _active_config()

    observation = collect_model_review_capability_observation(
        config,
        runner_labels={config.runner_label},
        runner_groups={config.runner_group},
        probe_reference=lambda reference_id: ModelModelReviewReferenceProbe(
            present=reference_id != config.endpoint_reference_id,
            healthy=True,
        ),
        probe_reviewer_cli=lambda: True,
        now=OBSERVED_AT,
    )

    assert observation.present_reference_ids == frozenset(
        {config.credential_reference_id, config.healthcheck_reference_id}
    )
    assert observation.healthy_reference_ids == observation.present_reference_ids


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
