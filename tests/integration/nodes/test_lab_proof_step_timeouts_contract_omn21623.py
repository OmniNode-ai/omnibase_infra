# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""The shipped plan contract loads into a bounded step-timeout policy.

Ticket: OMN-21623
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.models.lab_proof.model_lab_proof_bundle_policy import (
    ModelLabProofBundlePolicy,
)
from omnibase_infra.models.lab_proof.model_lab_proof_step_timeouts import (
    MAX_STEP_SECONDS,
    ModelLabProofStepTimeouts,
)

_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_lab_proof_plan_compute/contract.yaml"
)


def _bundle_block() -> dict[str, object]:
    contract = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    block: dict[str, object] = contract["config"]["local_bundle"]
    return block


@pytest.mark.integration
def test_shipped_contract_loads_with_every_timeout_within_the_bound() -> None:
    policy = ModelLabProofBundlePolicy.model_validate(_bundle_block())
    values = policy.step_timeouts.model_dump().values()
    assert all(1 <= value <= MAX_STEP_SECONDS for value in values)


@pytest.mark.integration
@pytest.mark.parametrize("name", list(ModelLabProofStepTimeouts.model_fields))
def test_a_timeout_above_the_bound_is_refused_at_load(name: str) -> None:
    block = _bundle_block()
    timeouts = dict(
        ModelLabProofBundlePolicy.model_validate(block).step_timeouts.model_dump()
    )
    timeouts[name] = MAX_STEP_SECONDS + 1
    with pytest.raises(ValidationError, match=name):
        ModelLabProofBundlePolicy.model_validate({**block, "step_timeouts": timeouts})
