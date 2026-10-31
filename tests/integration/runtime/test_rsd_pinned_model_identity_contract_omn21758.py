# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The packaged RSD pin contract on disk drives the real RSD runtime models (OMN-21758)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import omnibase_infra
from omnibase_infra.models.rsd.model_rsd_pinned_model_identity import (
    ModelRsdPinnedModelIdentity,
)
from omnibase_infra.runtime.models.model_rsd_live_delegation_observation import (
    ModelRsdLiveDelegationObservation,
)

pytestmark = pytest.mark.integration

_CONTRACT = (
    Path(omnibase_infra.__file__).parent
    / "contracts"
    / "rsd_pinned_model_identity.yaml"
)


def test_pin_comes_from_the_contract_file_on_disk() -> None:
    document = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    declared = document["pinned_model_identity"]
    pinned = ModelRsdPinnedModelIdentity.from_contract()
    assert pinned.model_dump() == declared


def test_real_runtime_model_accepts_the_pin_and_refuses_a_substitute() -> None:
    pinned = ModelRsdPinnedModelIdentity.from_contract()
    revision = "0" * 40
    accepted = ModelRsdLiveDelegationObservation(
        installed_public_rsd_revision_sha=revision,
        observed_model_id=pinned.observed_model,
    )
    assert accepted.observed_model_id == pinned.observed_model
    with pytest.raises(ValidationError, match="contract pin"):
        ModelRsdLiveDelegationObservation(
            installed_public_rsd_revision_sha=revision,
            observed_model_id=pinned.served_model,
        )
