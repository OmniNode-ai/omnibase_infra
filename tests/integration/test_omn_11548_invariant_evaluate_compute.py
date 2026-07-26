# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration coverage for the OMN-11548 invariant evaluation compute node.

OMN-14822 flipped the node to canonical def-B: contract routing now binds each
operation to a def-B handler class (``name`` + ``module``, no ``function``), and
the behavior lives on ``HandlerInvariantEvaluate.handle(request) -> response``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_core.enums import EnumInvariantType, EnumSeverity
from omnibase_core.models.invariant import ModelInvariant
from omnibase_infra.nodes.node_invariant_evaluate_compute.handlers import (
    HandlerInvariantEvaluate,
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
    assert handler["name"] == "HandlerInvariantEvaluate"
    # Canonical def-B routing binds by name+module only; no legacy function field.
    assert "function" not in handler

    result = await HandlerInvariantEvaluate().handle(
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
