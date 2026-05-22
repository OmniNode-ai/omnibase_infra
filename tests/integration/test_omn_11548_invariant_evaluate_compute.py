# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration coverage for the OMN-11548 invariant evaluation compute node."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_core.enums import EnumInvariantType, EnumSeverity
from omnibase_core.models.invariant import ModelInvariant
from omnibase_infra.nodes.node_invariant_evaluate_compute.handlers import (
    handle_invariant_evaluate,
)
from omnibase_infra.nodes.node_invariant_evaluate_compute.models import (
    ModelInvariantEvaluateInput,
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_invariant_evaluate_compute_contract_routes_to_handler() -> None:
    contract_path = (
        Path("src")
        / "omnibase_infra"
        / "nodes"
        / "node_invariant_evaluate_compute"
        / "contract.yaml"
    )
    contract = yaml.safe_load(contract_path.read_text())
    handler = contract["handler_routing"]["handlers"][0]["handler"]

    assert handler["module"].endswith("handler_invariant_evaluate")
    assert handler["function"] == "handle_invariant_evaluate"

    result = await handle_invariant_evaluate(
        ModelInvariantEvaluateInput(
            invariant=ModelInvariant(
                name="Status Present",
                type=EnumInvariantType.FIELD_PRESENCE,
                severity=EnumSeverity.CRITICAL,
                config={"fields": ["status"]},
            ),
            output={"status": "ok"},
        )
    )

    assert result.passed is True
    assert result.invariant_name == "Status Present"
