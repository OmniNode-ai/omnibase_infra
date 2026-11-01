# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""RSD model ids come from the pinned contract, not from the model classes (OMN-21758)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.rsd import model_rsd_pinned_model_identity as pin
from omnibase_infra.models.rsd.model_rsd_pinned_model_identity import (
    ModelRsdPinnedModelIdentity,
)
from omnibase_infra.runtime.models.model_rsd_artifact_source_relation import (
    ModelRsdArtifactSourceRelation,
)
from omnibase_infra.runtime.models.model_rsd_live_delegation_authority_envelope import (
    ModelRsdLiveDelegationAuthorityEnvelope,
)
from omnibase_infra.runtime.models.model_rsd_live_delegation_observation import (
    ModelRsdLiveDelegationObservation,
)
from omnibase_infra.runtime.models.model_rsd_live_delegation_overlay import (
    ModelRsdLiveDelegationOverlay,
)
from omnibase_infra.runtime.models.model_rsd_model_artifact_provenance import (
    ModelRsdModelArtifactProvenance,
)
from tests.unit.runtime.test_rsd_live_delegation_overlay import _OVERLAY_TEXT
from tests.unit.runtime.test_rsd_model_artifact_provenance import _EXAMPLE_TEXT

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


@pytest.mark.parametrize(
    ("model", "field", "pin_field"),
    [
        (ModelRsdLiveDelegationAuthorityEnvelope, "model_id", "served_model"),
        (
            ModelRsdLiveDelegationAuthorityEnvelope,
            "observed_model_id",
            "observed_model",
        ),
        (ModelRsdLiveDelegationOverlay, "model_id", "served_model"),
        (ModelRsdLiveDelegationOverlay, "observed_model_id", "observed_model"),
        (ModelRsdModelArtifactProvenance, "model_id", "artifact_model"),
        (ModelRsdModelArtifactProvenance, "base_model_id", "base_model"),
        (ModelRsdModelArtifactProvenance, "artifact_id", "artifact_ref"),
        (ModelRsdModelArtifactProvenance, "architecture", "architecture"),
        (ModelRsdModelArtifactProvenance, "served_model_id", "served_model"),
        (ModelRsdModelArtifactProvenance, "launch_profile_id", "launch_profile"),
    ],
)
def test_authority_overlay_and_provenance_refuse_each_substituted_pin(
    model: type[BaseModel], field: str, pin_field: str
) -> None:
    if model is ModelRsdModelArtifactProvenance:
        payload = yaml.safe_load(_EXAMPLE_TEXT)
    else:
        payload = yaml.safe_load(_OVERLAY_TEXT)
        if model is ModelRsdLiveDelegationAuthorityEnvelope:
            payload = payload["authority_envelope"]

    expected = getattr(ModelRsdPinnedModelIdentity.from_contract(), pin_field)
    accepted = model.model_validate(payload)
    assert getattr(accepted, field) == expected
    payload[field] = expected + "-substituted"
    with pytest.raises(ValidationError) as error:
        model.model_validate(payload)
    assert any(
        detail["loc"] == (field,) and "contract pin" in detail["msg"]
        for detail in error.value.errors()
    )


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "[",
        "",
        "[]",
        "{}",
        "pinned_model_identity: {}",
        "pinned_model_identity: 1",
        "pinned_model_identity: []",
        "pinned_model_identity: null",
        "pinned_model_identity: {served_model: 123}",
    ],
)
def test_contract_load_failures_are_configuration_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: str | None
) -> None:
    contracts = tmp_path / "contracts"
    contracts.mkdir()
    contract = contracts / "rsd_pinned_model_identity.yaml"
    if payload is not None:
        contract.write_text(payload, encoding="utf-8")
    monkeypatch.setattr(pin, "files", lambda _package: tmp_path)
    ModelRsdPinnedModelIdentity.from_contract.cache_clear()
    try:
        with pytest.raises(
            ProtocolConfigurationError, match="repair or reinstall"
        ) as error:
            _relation("substituted")
        assert error.value.__cause__ is not None
        assert not isinstance(error.value, ValidationError)
    finally:
        ModelRsdPinnedModelIdentity.from_contract.cache_clear()


def test_model_imports_do_not_read_the_contract() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.resources; "
            "importlib.resources.files = lambda *args: "
            "(_ for _ in ()).throw(AssertionError('import must not read contract')); "
            "import omnibase_infra.runtime.models.model_rsd_artifact_source_relation; "
            "import omnibase_infra.runtime.models.model_rsd_live_delegation_authority_envelope; "
            "import omnibase_infra.runtime.models.model_rsd_live_delegation_observation; "
            "import omnibase_infra.runtime.models.model_rsd_live_delegation_overlay; "
            "import omnibase_infra.runtime.models.model_rsd_model_artifact_provenance",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0


def test_payload_validation_reuses_the_cached_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pinned = ModelRsdPinnedModelIdentity.from_contract()

    def unexpected_load(_package: str) -> None:
        raise AssertionError("cached payload validation must not read a contract")

    monkeypatch.setattr(pin, "files", unexpected_load)
    assert (
        _relation(pinned.base_model).artifact_declared_base_model_id
        == pinned.base_model
    )
    with pytest.raises(ValidationError, match="contract pin"):
        _relation(pinned.base_model + "-substituted")
