# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for node_contract_validate_compute."""

from __future__ import annotations

import pytest

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.models.validation.model_contract_validation_result import (
    ModelContractValidationResult,
)
from omnibase_infra.nodes.node_contract_validate_compute.handlers import (
    handle_contract_validate,
)
from omnibase_infra.nodes.node_contract_validate_compute.models import (
    ModelContractValidateInput,
)
from omnibase_infra.nodes.node_contract_validate_compute.node import (
    NodeContractValidateCompute,
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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_contract_validate_validates_yaml_content() -> None:
    result = await handle_contract_validate(
        ModelContractValidateInput(
            contract_content=_VALID_EFFECT_CONTRACT,
            contract_type="effect",
        )
    )

    assert isinstance(result, ModelContractValidationResult)
    assert result.is_valid is True
    assert result.score >= 0.8
    assert result.violations == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_contract_validate_scores_model_compliance() -> None:
    model_code = """
from pydantic import BaseModel

class ModelDatabaseWriteInput(BaseModel):
    table_name: str

class ModelDatabaseWriteOutput(BaseModel):
    success: bool
"""

    result = await handle_contract_validate(
        ModelContractValidateInput(
            contract_content=_VALID_EFFECT_CONTRACT,
            contract_type="effect",
            model_code=model_code,
        )
    )

    assert result.is_valid is True
    assert not [item for item in result.violations if "not found" in item]


@pytest.mark.unit
def test_node_contract_validate_compute_is_declarative_shell() -> None:
    node = NodeContractValidateCompute(
        ModelONEXContainer(enable_service_registry=False)
    )

    assert isinstance(node, NodeContractValidateCompute)
