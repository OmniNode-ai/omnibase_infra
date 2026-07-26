# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for node_contract_validate_compute (canonical definition B).

The handler is driven through the REAL auto-wiring dispatch entrypoint
(``_make_dispatch_callback``) — the same bind the runtime uses — so the tests
prove the canonical ``handle(request) -> response`` shape is dispatch-reachable,
not merely importable. Before the def-B flip the class exposed no ``handle`` and
this path raised ``ModelOnexError`` (missing dispatch entrypoint); after the flip
the same path returns a ``ModelDispatchResult`` carrying the validation result.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.models.validation.model_contract_validation_result import (
    ModelContractValidationResult,
)
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.nodes.node_contract_validate_compute.handlers import (
    HandlerContractValidate,
)
from omnibase_infra.nodes.node_contract_validate_compute.models import (
    ModelContractValidateInput,
)
from omnibase_infra.nodes.node_contract_validate_compute.node import (
    NodeContractValidateCompute,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import _make_dispatch_callback

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


def _wire_dispatch() -> object:
    """Bind the canonical handler through the real auto-wiring dispatch path."""
    # event_model=None mirrors an operation_match def-B handler: the engine hands
    # the dispatcher the raw materialized wire dict, and the adapter coerces it
    # into the handler's declared input model from the handle() signature.
    return _make_dispatch_callback(HandlerContractValidate())


def _wire_envelope(payload: dict[str, object]) -> dict[str, object]:
    """A raw materialized wire envelope; correlation_id triggers payload unwrap."""
    return {"payload": payload, "correlation_id": str(uuid4())}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_validates_yaml_content_via_real_dispatch() -> None:
    callback = _wire_dispatch()
    result = await callback(  # type: ignore[operator]
        _wire_envelope(
            {
                "contract_content": _VALID_EFFECT_CONTRACT,
                "contract_type": "effect",
            }
        )
    )

    assert isinstance(result, ModelDispatchResult)
    assert result.status is EnumDispatchStatus.SUCCESS
    assert result.output_count == 1
    (validation_result,) = result.output_events
    assert isinstance(validation_result, ModelContractValidationResult)
    assert validation_result.is_valid is True
    assert validation_result.score >= 0.8
    assert validation_result.violations == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_scores_model_compliance_via_real_dispatch() -> None:
    model_code = """
from pydantic import BaseModel

class ModelDatabaseWriteInput(BaseModel):
    table_name: str

class ModelDatabaseWriteOutput(BaseModel):
    success: bool
"""

    callback = _wire_dispatch()
    result = await callback(  # type: ignore[operator]
        _wire_envelope(
            {
                "contract_content": _VALID_EFFECT_CONTRACT,
                "contract_type": "effect",
                "model_code": model_code,
            }
        )
    )

    assert isinstance(result, ModelDispatchResult)
    (validation_result,) = result.output_events
    assert isinstance(validation_result, ModelContractValidationResult)
    assert validation_result.is_valid is True
    assert not [item for item in validation_result.violations if "not found" in item]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_owns_behavior_direct_call() -> None:
    """handle() itself owns the validation behavior — no retained op-method shim."""
    result = await HandlerContractValidate().handle(
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
def test_node_contract_validate_compute_is_declarative_shell() -> None:
    node = NodeContractValidateCompute(
        ModelONEXContainer(enable_service_registry=False)
    )

    assert isinstance(node, NodeContractValidateCompute)
