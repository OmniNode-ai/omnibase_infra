# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Contract tests for stable delegation cohort keys (OMN-18930)."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from omnibase_infra.models.delegation import (
    ModelDelegationCohortKey,
    ModelDelegationFirstInferenceIdentity,
    ModelDelegationProviderPolicy,
    ModelDelegationRetryBounds,
    ModelDelegationTierRetryBound,
)
from omnibase_infra.models.model_node_identity import ModelNodeIdentity

_PROMPT_SHA256 = "a" * 64
_ROUTING_SHA256 = "b" * 64
_ESCALATION_SHA256 = "d" * 64
_RESPONSE_CONTRACT_SHA256 = "c" * 64


def _synthetic_complete_key() -> ModelDelegationCohortKey:
    """Build a synthetic complete key; this is not live-lane evidence."""
    return ModelDelegationCohortKey(
        prompt_sha256=_PROMPT_SHA256,
        resolved_task_type="document",
        response_contract_sha256=_RESPONSE_CONTRACT_SHA256,
        lane="isolated-lab",
        build_identity="git:0123456789abcdef0123456789abcdef01234567",
        consumer_identity=ModelNodeIdentity(
            env="isolated-lab",
            service="omnibase-infra",
            node_name="runtime_effects",
            version="v0.38.36",
        ),
        first_hop_identity=ModelDelegationFirstInferenceIdentity(
            backend_id="test-backend",
            model_id="test-model",
            tier="local",
            provider="test-provider",
        ),
        provider_policy=ModelDelegationProviderPolicy(
            routing_tiers_sha256=_ROUTING_SHA256,
            escalation_config_sha256=_ESCALATION_SHA256,
        ),
        deadline_seconds=240.0,
        retry_bounds=ModelDelegationRetryBounds(
            per_tier=(
                ModelDelegationTierRetryBound(tier="local", max_retries=1),
                ModelDelegationTierRetryBound(tier="remote", max_retries=2),
            ),
            max_escalations=3,
        ),
    )


def test_complete_synthetic_key_is_stable_and_matches_itself() -> None:
    key = _synthetic_complete_key()

    key.require_same_cohort(_synthetic_complete_key())

    assert key.key_sha256 == _synthetic_complete_key().key_sha256
    assert key.changed_dimensions(_synthetic_complete_key()) == ()


def test_every_required_dimension_must_be_present() -> None:
    complete = _synthetic_complete_key().model_dump(mode="python")

    for field_name in ModelDelegationCohortKey.model_fields:
        incomplete = deepcopy(complete)
        incomplete.pop(field_name)
        with pytest.raises(ValidationError):
            ModelDelegationCohortKey.model_validate(incomplete)


def test_absent_response_contract_is_explicit_not_omitted() -> None:
    fields = _synthetic_complete_key().model_dump(mode="python")
    fields["response_contract_sha256"] = None
    key = ModelDelegationCohortKey.model_validate(fields)

    assert key.response_contract_sha256 is None
    fields.pop("response_contract_sha256")
    with pytest.raises(ValidationError):
        ModelDelegationCohortKey.model_validate(fields)


def test_offset489_capture_is_an_incomplete_key_negative_control() -> None:
    fixture_path = (
        Path(__file__).parents[1]
        / "fixtures"
        / "delegation"
        / "offset489_incomplete_cohort_key.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    with pytest.raises(ValidationError) as error:
        ModelDelegationCohortKey.model_validate(fixture["observed_key_fields"])

    rendered_error = str(error.value)
    for field_name in fixture["unproven_key_dimensions"]:
        assert field_name in rendered_error


def test_captured_key_cli_rejects_cross_cohort(tmp_path: Path) -> None:
    script = (
        Path(__file__).parents[2] / "scripts" / "validate_delegation_cohort_keys.py"
    )
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline = _synthetic_complete_key()
    changed_policy = baseline.provider_policy.model_copy(
        update={"routing_tiers_sha256": "e" * 64}
    )
    candidate = baseline.model_copy(update={"provider_policy": changed_policy})
    baseline_path.write_text(baseline.model_dump_json(), encoding="utf-8")
    candidate_path.write_text(candidate.model_dump_json(), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(script), str(baseline_path), str(candidate_path)],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 1
    assert result.stdout == "CROSS_COHORT: provider_policy\n"


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("build_identity", "git:fedcba9876543210fedcba9876543210fedcba98"),
        (
            "provider_policy",
            ModelDelegationProviderPolicy(
                routing_tiers_sha256="e" * 64,
                escalation_config_sha256=_ESCALATION_SHA256,
            ),
        ),
    ],
)
def test_build_or_provider_policy_change_rejects_comparison(
    field_name: str, changed_value: str
) -> None:
    baseline = _synthetic_complete_key()
    changed = baseline.model_copy(update={field_name: changed_value})

    with pytest.raises(ValueError, match=field_name):
        baseline.require_same_cohort(changed)


def test_lane_label_cannot_substitute_for_typed_consumer_identity() -> None:
    fields = _synthetic_complete_key().model_dump(mode="python")
    fields["consumer_identity"] = fields["lane"]

    with pytest.raises(ValidationError):
        ModelDelegationCohortKey.model_validate(fields)


def test_first_inference_identity_is_provider_attempt_not_node_identity() -> None:
    fields = _synthetic_complete_key().model_dump(mode="python")
    fields["first_hop_identity"] = {
        "backend_id": "backend-first",
        "model_id": "model-first",
        "tier": "local",
        "provider": "local-provider",
    }
    key = ModelDelegationCohortKey.model_validate(fields)
    changed = key.model_copy(
        update={
            "first_hop_identity": key.first_hop_identity.model_copy(
                update={"backend_id": "backend-second"}
            )
        }
    )

    assert key.first_hop_identity.backend_id == "backend-first"
    assert key.changed_dimensions(changed) == ("first_hop_identity",)
    with pytest.raises(ValidationError):
        ModelDelegationFirstInferenceIdentity(
            backend_id="backend-first",
            model_id="model-first",
            tier="local",
        )


def test_invalid_digest_and_unbounded_retry_values_are_rejected() -> None:
    fields = _synthetic_complete_key().model_dump(mode="python")
    fields["prompt_sha256"] = "not-a-digest"

    with pytest.raises(ValidationError, match="prompt_sha256"):
        ModelDelegationCohortKey.model_validate(fields)

    with pytest.raises(ValidationError):
        ModelDelegationRetryBounds(
            per_tier=(ModelDelegationTierRetryBound(tier="local", max_retries=-1),),
            max_escalations=0,
        )
    with pytest.raises(ValidationError):
        ModelDelegationRetryBounds(
            per_tier=(ModelDelegationTierRetryBound(tier="local", max_retries=0),),
            max_escalations=-1,
        )
    with pytest.raises(ValidationError, match="unique"):
        ModelDelegationRetryBounds(
            per_tier=(
                ModelDelegationTierRetryBound(tier="local", max_retries=0),
                ModelDelegationTierRetryBound(tier="local", max_retries=1),
            ),
            max_escalations=0,
        )


def test_policy_hash_is_canonical_and_retry_bounds_keep_scopes_distinct() -> None:
    policy = ModelDelegationProviderPolicy(
        routing_tiers_sha256=_ROUTING_SHA256,
        escalation_config_sha256=_ESCALATION_SHA256,
    )
    reordered = ModelDelegationProviderPolicy.model_validate(
        {
            "escalation_config_sha256": _ESCALATION_SHA256,
            "routing_tiers_sha256": _ROUTING_SHA256,
        }
    )

    assert policy.sha256 == reordered.sha256
    bounds = ModelDelegationRetryBounds(
        per_tier=(
            ModelDelegationTierRetryBound(tier="local", max_retries=1),
            ModelDelegationTierRetryBound(tier="remote", max_retries=2),
        ),
        max_escalations=3,
    )
    changed = bounds.model_copy(update={"max_escalations": 4})
    assert bounds.per_tier == changed.per_tier
    assert bounds.max_escalations != changed.max_escalations
    reversed_key = _synthetic_complete_key().model_copy(
        update={
            "retry_bounds": ModelDelegationRetryBounds(
                per_tier=tuple(reversed(bounds.per_tier)), max_escalations=3
            )
        }
    )
    assert _synthetic_complete_key().key_sha256 == reversed_key.key_sha256
    with pytest.raises(ValidationError):
        ModelDelegationProviderPolicy(
            routing_tiers_sha256="Z" * 64,
            escalation_config_sha256=_ESCALATION_SHA256,
        )
