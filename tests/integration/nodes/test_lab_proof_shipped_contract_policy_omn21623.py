# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""The shipped lab-proof plan contract on disk loads into a bounded policy.

Ticket: OMN-21623
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.models.lab_proof.model_lab_proof_bundle_policy import (
    ModelLabProofBundlePolicy,
)
from omnibase_infra.models.lab_proof.model_lab_proof_step_timeouts import (
    MAX_STEP_SECONDS,
)

_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_lab_proof_plan_compute/contract.yaml"
)


@pytest.mark.integration
def test_shipped_contract_on_disk_loads_with_every_timeout_within_the_bound() -> None:
    contract = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    policy = ModelLabProofBundlePolicy.model_validate(
        contract["config"]["local_bundle"]
    )
    values = policy.step_timeouts.model_dump().values()
    assert all(1 <= value <= MAX_STEP_SECONDS for value in values)
