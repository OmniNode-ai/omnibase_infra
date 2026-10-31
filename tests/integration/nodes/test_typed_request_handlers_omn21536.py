# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21536: handlers of two effect nodes take the one request model their contract names.

Loads the real ``contract.yaml`` of node_contract_persistence_effect and
node_session_state_effect from disk, resolves each routed handler and its
declared ``input_model``, and requires ``handle`` to take that model as its one
argument. The session-state handlers then run a write, read and garbage-collect
round trip on a real directory through their typed requests.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from omnibase_infra.nodes import (
    node_contract_persistence_effect,
    node_session_state_effect,
)
from omnibase_infra.nodes.node_session_state_effect.handlers import (
    HandlerRunContextRead,
    HandlerRunContextWrite,
    HandlerSessionIndexRead,
    HandlerSessionIndexWrite,
    HandlerStaleRunGC,
)
from omnibase_infra.nodes.node_session_state_effect.models import (
    ModelRunContext,
    ModelRunContextReadRequest,
    ModelRunContextWriteRequest,
    ModelSessionIndex,
    ModelSessionIndexReadRequest,
    ModelSessionIndexWriteRequest,
    ModelStaleRunGcRequest,
)

pytestmark = pytest.mark.integration

_NODE_PACKAGES = (node_contract_persistence_effect, node_session_state_effect)


def _routed_handlers() -> list[tuple[str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str]] = []
    for package in _NODE_PACKAGES:
        contract_path = Path(str(package.__file__)).parent / "contract.yaml"
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        for entry in contract["handler_routing"]["handlers"]:
            handler = entry["handler"]
            model = entry["input_model"]
            rows.append(
                (
                    package.__name__,
                    handler["module"],
                    handler["name"],
                    model["module"],
                    model["name"],
                )
            )
    return rows


def _imported_from(module_name: str, name: str) -> str | None:
    source = Path(str(importlib.import_module(module_name).__file__)).read_text(
        encoding="utf-8"
    )
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if any(alias.name == name for alias in node.names):
                return node.module
    return None


@pytest.mark.parametrize(
    ("node", "handler_module", "handler_name", "model_module", "model_name"),
    _routed_handlers(),
)
def test_routed_handler_takes_its_contract_input_model(
    node: str,
    handler_module: str,
    handler_name: str,
    model_module: str,
    model_name: str,
) -> None:
    handler_cls = getattr(importlib.import_module(handler_module), handler_name)
    model_cls = getattr(importlib.import_module(model_module), model_name)

    signature = inspect.signature(handler_cls.handle)
    params = list(signature.parameters)
    assert params == ["self", "request"], (
        f"{node}: {handler_name}.handle takes {params}"
    )
    # Handlers import payload types under TYPE_CHECKING, so the annotation is a string
    # that cannot be resolved at runtime. Pin the name, then pin the module identity
    # by resolving the handler's own import of that name.
    assert signature.parameters["request"].annotation == model_name
    source_module = _imported_from(handler_module, model_name)
    assert source_module is not None
    assert getattr(importlib.import_module(source_module), model_name) is model_cls


@pytest.mark.asyncio
async def test_session_state_round_trip_through_typed_requests(tmp_path: Path) -> None:
    correlation_id = uuid4()
    context = ModelRunContext(run_id="run-omn21536")
    index = ModelSessionIndex()

    wrote_run = await HandlerRunContextWrite(tmp_path).handle(
        ModelRunContextWriteRequest(context=context, correlation_id=correlation_id)
    )
    wrote_index = await HandlerSessionIndexWrite(tmp_path).handle(
        ModelSessionIndexWriteRequest(index=index, correlation_id=correlation_id)
    )
    read_run, read_run_result = await HandlerRunContextRead(tmp_path).handle(
        ModelRunContextReadRequest(run_id=context.run_id, correlation_id=correlation_id)
    )
    read_index, read_index_result = await HandlerSessionIndexRead(tmp_path).handle(
        ModelSessionIndexReadRequest(correlation_id=correlation_id)
    )
    deleted, gc_result = await HandlerStaleRunGC(tmp_path).handle(
        ModelStaleRunGcRequest(correlation_id=correlation_id)
    )

    for result in (
        wrote_run,
        wrote_index,
        read_run_result,
        read_index_result,
        gc_result,
    ):
        assert result.success is True
        assert result.correlation_id == correlation_id
    assert read_run is not None
    assert read_run.run_id == context.run_id
    assert read_index == index
    assert deleted == []
