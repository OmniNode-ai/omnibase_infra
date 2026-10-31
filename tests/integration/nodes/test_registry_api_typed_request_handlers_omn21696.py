# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21696: the registry API handlers take the one request model their contract names.

Loads the real ``contract.yaml`` of node_registry_api_effect from disk, resolves
each routed handler and its declared ``input_model``, and requires ``handle`` to
take that model as its one argument and to answer with the response model.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from typing import get_type_hints

import pytest
import yaml

from omnibase_infra.nodes import node_registry_api_effect
from omnibase_infra.nodes.node_registry_api_effect.models import (
    ModelRegistryApiResponse,
)

pytestmark = pytest.mark.integration


def _routed_handlers() -> list[tuple[str, str, str, str]]:
    contract_path = (
        Path(str(node_registry_api_effect.__file__)).parent / "contract.yaml"
    )
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    rows: list[tuple[str, str, str, str]] = []
    for entry in contract["handler_routing"]["handlers"]:
        handler = entry["handler"]
        model = entry["input_model"]
        rows.append(
            (
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


def test_the_contract_routes_nine_handlers() -> None:
    assert len(_routed_handlers()) == 9


@pytest.mark.parametrize(
    ("handler_module", "handler_name", "model_module", "model_name"),
    _routed_handlers(),
)
def test_routed_handler_takes_its_contract_input_model(
    handler_module: str,
    handler_name: str,
    model_module: str,
    model_name: str,
) -> None:
    handler_cls = getattr(importlib.import_module(handler_module), handler_name)
    model_cls = getattr(importlib.import_module(model_module), model_name)

    signature = inspect.signature(handler_cls.handle)
    params = list(signature.parameters)
    assert params == ["self", "request"], f"{handler_name}.handle takes {params}"
    # Handlers annotate with strings (``from __future__ import annotations``), so pin
    # the raw string to the model name, evaluate it against the contract's own model
    # class and require that class, then pin the module identity through the
    # handler's own import of that name.
    assert signature.parameters["request"].annotation == model_name
    hints = get_type_hints(
        handler_cls.handle,
        localns={
            model_name: model_cls,
            "ModelRegistryApiResponse": ModelRegistryApiResponse,
        },
    )
    assert hints["request"] is model_cls
    assert hints["return"] is ModelRegistryApiResponse
    source_module = _imported_from(handler_module, model_name)
    assert source_module is not None
    assert getattr(importlib.import_module(source_module), model_name) is model_cls
