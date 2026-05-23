# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration coverage for the OMN-11550 contract validation compute node."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.nodes.node_contract_validate_compute.handlers import (
    handle_contract_validate,
)
from omnibase_infra.nodes.node_contract_validate_compute.models import (
    ModelContractValidateInput,
)

_VALID_EFFECT_CONTRACT = """
name: DatabaseWriterEffect
contract_version:
  major: 1
  minor: 0
  patch: 0
description: Effect node for writing data to PostgreSQL database
node_type: effect_generic
input_model: omnibase_core.models.ModelDatabaseWriteInput
output_model: omnibase_core.models.ModelDatabaseWriteOutput
io_operations:
  - operation_type: WRITE
    operation_target: DATABASE
    atomic: true
    validation_enabled: true
    error_handling_strategy: RETRY
"""


@pytest.mark.integration
@pytest.mark.asyncio
async def test_contract_validate_compute_contract_routes_to_handler() -> None:
    contract_path = (
        Path("src")
        / "omnibase_infra"
        / "nodes"
        / "node_contract_validate_compute"
        / "contract.yaml"
    )
    contract = yaml.safe_load(contract_path.read_text())

    handler = contract["handler_routing"]["handlers"][0]["handler"]
    assert handler["module"].endswith("handler_contract_validate")
    assert handler["function"] == "handle_contract_validate"

    result = await handle_contract_validate(
        ModelContractValidateInput(
            contract_content=_VALID_EFFECT_CONTRACT,
            contract_type="effect",
        )
    )

    assert result.is_valid is True
    assert result.score >= 0.8
    assert result.violations == []
