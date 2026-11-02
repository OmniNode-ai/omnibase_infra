# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""The plan built from the shipped contract's bundle policy carries the contract's numbers.

The unit suite proves the handler against synthetic policy values. This test
reads the contract on disk and asks the built plan for the numbers that file
declares, so a contract value that never reaches a step fails here.

Ticket: OMN-21623
"""

from __future__ import annotations

import pytest
import yaml

from omnibase_infra.lab_proof.enum_lab_proof_step_id import EnumLabProofStepId
from omnibase_infra.nodes.node_lab_proof_plan_compute.handlers.handler_lab_proof_plan import (
    HandlerLabProofPlan,
)
from tests.unit.nodes.node_lab_proof_plan_compute.lab_proof_fixtures import (
    CONTRACT,
    core_request,
)


@pytest.mark.integration
def test_plan_steps_carry_the_timeouts_and_bus_address_the_contract_declares() -> None:
    declared = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))["config"][
        "local_bundle"
    ]
    plan = HandlerLabProofPlan().handle(core_request())
    by_id = {step.step_id: step for step in plan.steps}
    step_timeouts = declared["step_timeouts"]
    assert (
        by_id[EnumLabProofStepId.FOCUSED_TESTS].timeout_seconds
        == step_timeouts["focused_tests_seconds"]
    )
    assert by_id[EnumLabProofStepId.UP].timeout_seconds == step_timeouts["up_seconds"]
    delegation = by_id[EnumLabProofStepId.GOLDEN_CHAIN_DELEGATION]
    assert delegation.timeout_seconds == declared["delegation_timeout_seconds"]
    argv = list(delegation.argv)
    assert (
        argv[argv.index("--kafka-bootstrap") + 1] == declared["bus_bootstrap_servers"]
    )
