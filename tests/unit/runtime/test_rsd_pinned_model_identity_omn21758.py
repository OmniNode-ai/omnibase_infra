# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""RSD model ids come from the pinned contract, not from the model classes (OMN-21758)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.models.rsd.model_rsd_pinned_model_identity import (
    ModelRsdPinnedModelIdentity,
)
from omnibase_infra.runtime.models.model_rsd_artifact_source_relation import (
    ModelRsdArtifactSourceRelation,
)
from omnibase_infra.runtime.models.model_rsd_live_delegation_observation import (
    ModelRsdLiveDelegationObservation,
)

pytestmark = pytest.mark.unit

_MODEL_CARD = "57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641"
_CONFIG = "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
_ARTIFACT_CARD = "3704987ff0e2206ab934af6d71cd0a9b5140536ee8d305aa7ba6e7665f135058"


def _relation(base_model_id: str) -> ModelRsdArtifactSourceRelation:
    return ModelRsdArtifactSourceRelation.model_validate(
        {
            "source_metadata_authority": "huggingface-model-revision-api-v1",
            "source_model_card_sha256": _MODEL_CARD,
            "source_config_sha256": _CONFIG,
            "artifact_metadata_authority": "huggingface-model-revision-api-v1",
            "artifact_model_card_sha256": _ARTIFACT_CARD,
            "artifact_embedded_source_model_card_sha256": _MODEL_CARD,
            "artifact_declared_base_model_id": base_model_id,
            "relation_status": "publisher-declared-unverified",
            "approval_status": "not-approved",
        }
    )


def test_contract_declares_every_pinned_identity() -> None:
    pinned = ModelRsdPinnedModelIdentity.from_contract()
    assert pinned.served_model
    assert pinned.observed_model
    assert pinned.observed_model != pinned.served_model


def test_require_refuses_a_value_that_differs_from_the_pin() -> None:
    pinned = ModelRsdPinnedModelIdentity.from_contract()
    assert pinned.require("served_model", pinned.served_model) == (pinned.served_model)
    with pytest.raises(ValueError, match="differs from the contract pin"):
        pinned.require("served_model", pinned.observed_model)
    with pytest.raises(ValueError, match="not a pinned model identity field"):
        pinned.require("missing", "x")


def test_source_relation_takes_its_base_model_from_the_contract() -> None:
    pinned = ModelRsdPinnedModelIdentity.from_contract()
    relation = _relation(pinned.base_model)
    assert relation.artifact_declared_base_model_id == pinned.base_model
    with pytest.raises(ValidationError, match="contract pin"):
        _relation(pinned.observed_model)


def test_observation_takes_its_observed_model_from_the_contract() -> None:
    pinned = ModelRsdPinnedModelIdentity.from_contract()
    revision = "0" * 40
    ok = ModelRsdLiveDelegationObservation(
        installed_public_rsd_revision_sha=revision,
        observed_model_id=pinned.observed_model,
    )
    assert ok.observed_model_id == pinned.observed_model
    assert (
        ModelRsdLiveDelegationObservation(
            installed_public_rsd_revision_sha=revision
        ).observed_model_id
        is None
    )
    with pytest.raises(ValidationError, match="contract pin"):
        ModelRsdLiveDelegationObservation(
            installed_public_rsd_revision_sha=revision,
            observed_model_id=pinned.served_model,
        )
