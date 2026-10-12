# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-to-runtime dispatch entrypoint check for the topic-migration executor (OMN-21050).

Loads the node's real contract.yaml from disk, imports the handler class the
contract routes to, and resolves its dispatch entrypoint and request model with
the resolver auto-wiring uses at runtime.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

from omnibase_infra.nodes import node_topic_migration_executor_effect as node_pkg
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _resolve_effective_handle_method,
    _typed_def_b_input_model,
)

pytestmark = pytest.mark.integration


def test_contract_routed_handler_exposes_runtime_dispatch_entrypoint() -> None:
    contract_path = Path(node_pkg.__file__).parent / "contract.yaml"
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    routed = contract["handler_routing"]["handlers"]
    assert routed, "contract routes no handlers"

    for entry in routed:
        module = importlib.import_module(entry["handler"]["module"])
        handler_cls = getattr(module, entry["handler"]["name"])
        handler = handler_cls.__new__(handler_cls)
        assert _resolve_effective_handle_method(handler) is not None, (
            f"{entry['handler']['name']} exposes no handle()/handle_async() entrypoint"
        )
        input_model = contract["input_model"]
        assert _typed_def_b_input_model(handler).__name__ == input_model["name"]
