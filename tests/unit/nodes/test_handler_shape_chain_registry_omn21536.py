# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler-shape regression for node_chain_orchestrator and node_registry_effect.

Every handler of the two nodes takes one typed request, ``handle(self, request:
ModelX)``, which is the shape the auto-wiring runtime resolves and dispatches
(``_resolve_def_b_input_model_type``); the correlation ID travels inside the request
model instead of as a second positional parameter (OMN-21536).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.primitives import ModelSemVer
from omnibase_core.validation.validator_handler_shape import ValidatorHandlerShape
from omnibase_infra.enums import EnumBackendType
from omnibase_infra.nodes.node_chain_orchestrator.handlers.handler_chain_learn_initiate import (
    HandlerChainLearnInitiate,
)
from omnibase_infra.nodes.node_chain_orchestrator.handlers.handler_chain_retrieval_complete import (
    HandlerChainRetrievalComplete,
)
from omnibase_infra.nodes.node_chain_orchestrator.handlers.handler_chain_store_complete import (
    HandlerChainStoreComplete,
)
from omnibase_infra.nodes.node_chain_orchestrator.models import (
    ModelChainLearnCommand,
    ModelChainRetrievalResult,
    ModelChainStoreResult,
)
from omnibase_infra.nodes.node_registry_effect.handlers import (
    HandlerPartialRetry,
    HandlerPostgresDeactivate,
    HandlerPostgresUpsert,
)
from omnibase_infra.nodes.node_registry_effect.models import (
    ModelBackendResult,
    ModelRegistryRequest,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _resolve_def_b_input_model_type,
)

pytestmark = pytest.mark.unit


@dataclass
class _PartialRetry:
    """A ProtocolPartialRetryRequest the handler accepts."""

    node_id: UUID
    node_type: EnumNodeKind
    node_version: ModelSemVer
    correlation_id: UUID
    target_backend: EnumBackendType
    idempotency_key: str | None = None
    endpoints: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)


@pytest.mark.parametrize(
    ("handle_method", "request_model"),
    [
        (HandlerChainLearnInitiate.handle, ModelChainLearnCommand),
        (HandlerChainRetrievalComplete.handle, ModelChainRetrievalResult),
        (HandlerChainStoreComplete.handle, ModelChainStoreResult),
        (HandlerPostgresUpsert.handle, ModelRegistryRequest),
        (HandlerPostgresDeactivate.handle, ModelRegistryRequest),
    ],
)
def test_runtime_resolves_the_single_typed_request(
    handle_method: object, request_model: type[BaseModel]
) -> None:
    assert _resolve_def_b_input_model_type(handle_method) is request_model


def test_partial_retry_handle_takes_one_request_parameter() -> None:
    # The request is typed by ProtocolPartialRetryRequest, a Protocol rather than a
    # BaseModel, so the runtime resolver yields no model for it; the shape is one
    # positional parameter.
    import inspect

    positional = [
        name
        for name in inspect.signature(HandlerPartialRetry.handle).parameters
        if name != "self"
    ]
    assert positional == ["request"]


@pytest.mark.asyncio
async def test_partial_retry_reports_the_request_correlation_id() -> None:
    adapter = AsyncMock()
    adapter.upsert.return_value = ModelBackendResult(
        success=True, duration_ms=1.0, backend_id="postgres"
    )
    correlation_id = uuid4()
    request = _PartialRetry(
        node_id=uuid4(),
        node_type=EnumNodeKind.EFFECT,
        node_version=ModelSemVer(major=1, minor=0, patch=0),
        correlation_id=correlation_id,
        target_backend=EnumBackendType.POSTGRES,
    )

    result = await HandlerPartialRetry(adapter).handle(request)

    assert result.success is True
    assert result.correlation_id == correlation_id


@pytest.mark.asyncio
async def test_registry_handlers_report_the_request_correlation_id() -> None:
    adapter = AsyncMock()
    adapter.upsert.return_value = ModelBackendResult(
        success=True, duration_ms=1.0, backend_id="postgres"
    )
    adapter.deactivate.return_value = ModelBackendResult(
        success=True, duration_ms=1.0, backend_id="postgres"
    )
    correlation_id = uuid4()
    request = ModelRegistryRequest.model_validate(
        {
            "node_id": uuid4(),
            "node_type": EnumNodeKind.EFFECT,
            "node_version": {"major": 1, "minor": 0, "patch": 0},
            "correlation_id": correlation_id,
            "timestamp": datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
        }
    )

    upserted = await HandlerPostgresUpsert(adapter).handle(request)
    deactivated = await HandlerPostgresDeactivate(adapter).handle(request)

    assert upserted.correlation_id == correlation_id
    assert deactivated.correlation_id == correlation_id


_SLICE_HANDLERS = (
    "nodes/node_chain_orchestrator/handlers/handler_chain_learn_initiate.py",
    "nodes/node_chain_orchestrator/handlers/handler_chain_retrieval_complete.py",
    "nodes/node_chain_orchestrator/handlers/handler_chain_store_complete.py",
    "nodes/node_registry_effect/handlers/handler_partial_retry.py",
    "nodes/node_registry_effect/handlers/handler_postgres_deactivate.py",
    "nodes/node_registry_effect/handlers/handler_postgres_upsert.py",
)

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "omnibase_infra"


def test_core_handler_shape_validator_flags_a_multi_parameter_handle(
    tmp_path: Path,
) -> None:
    """Positive control: the validator does report the shape this change removed."""
    offender = tmp_path / "handler_offender.py"
    offender.write_text(
        "class HandlerOffender:\n"
        "    async def handle(self, request: object, correlation_id: object) -> None:\n"
        "        return None\n",
        encoding="utf-8",
    )

    result = ValidatorHandlerShape().validate_file(offender)

    assert [issue.rule_name for issue in result.issues] == ["noncanonical_handle"]


@pytest.mark.parametrize("relative_path", _SLICE_HANDLERS)
def test_core_handler_shape_validator_reports_nothing_for_the_slice(
    relative_path: str,
) -> None:
    result = ValidatorHandlerShape().validate_file(_SRC_ROOT / relative_path)

    assert not result.issues, (
        f"handler_shape reports {[issue.rule_name for issue in result.issues]}"
        f" on {relative_path}"
    )
