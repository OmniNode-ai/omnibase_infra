# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""node_runtime_start_first_effect's contract on disk resolves to its handler (OMN-22151).

Loads the node contract from the installed package tree, resolves every routed
operation to the declared handler class and checks the handler's classification.
No Docker daemon is touched.
"""

from __future__ import annotations

import importlib
import types
from pathlib import Path

import pytest
import yaml

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.handlers import handler_runtime_start_first as effect_mod

pytestmark = pytest.mark.integration

CONTRACT = (
    Path(effect_mod.__file__).resolve().parents[1]
    / "nodes"
    / "node_runtime_start_first_effect"
    / "contract.yaml"
)


def test_every_routed_operation_resolves_to_the_infra_effect_handler() -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    routes = contract["handler_routing"]["handlers"]
    assert {entry["operation"] for entry in routes} == {"replace"}
    for entry in routes:
        module = importlib.import_module(entry["handler"]["module"])
        handler_cls = getattr(module, entry["handler"]["name"])
        assert handler_cls is effect_mod.HandlerRuntimeStartFirst
        handler = handler_cls(execution=types.ModuleType("execution"))
        assert handler.handler_type is EnumHandlerType.INFRA_HANDLER
        assert handler.handler_category is EnumHandlerTypeCategory.EFFECT
