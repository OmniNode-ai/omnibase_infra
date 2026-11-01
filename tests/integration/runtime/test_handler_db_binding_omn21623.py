# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Wire the six DB-composing handlers from their real contracts (OMN-21623)."""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.handlers.container.container_handler_db import bind_handler_db
from omnibase_infra.runtime.auto_wiring import (
    discover_contracts_from_paths,
    wire_from_manifest,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = pytest.mark.integration

_NODES = Path(__file__).resolve().parents[3] / "src/omnibase_infra/nodes"
_HANDLERS = (
    ("node_build_loop_write_effect", ("handler_build_loop_append",)),
    ("node_delegation_chain_ledger_effect", ("handler_delegation_chain_ledger",)),
    ("node_gateway_link_health_write_effect", ("handler_gateway_link_health_upsert",)),
    ("node_ledger_write_effect", ("handler_ledger_append", "handler_ledger_query")),
    ("node_validation_ledger_write_effect", ("handler_validation_ledger_append",)),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(("node", "modules"), _HANDLERS)
async def test_contract_wiring_uses_db_binding(
    node: str, modules: tuple[str, ...]
) -> None:
    contract = _NODES / node / "contract.yaml"
    assert contract.is_file()
    manifest = discover_contracts_from_paths([contract])
    assert manifest.total_errors == 0, manifest.errors
    assert manifest.total_discovered == 1
    container = ModelONEXContainer()
    report = await wire_from_manifest(
        manifest=manifest,
        dispatch_engine=MessageDispatchEngine(),
        event_bus=None,
        container=container,
        subscribe_immediately=False,
    )

    assert report.total_failed == 0, report.results
    assert report.total_wired == 1, report.results
    assert report.results[0].dispatchers_registered


@pytest.mark.parametrize(("node", "modules"), _HANDLERS)
def test_contract_handlers_bind_the_supplied_container(
    node: str, modules: tuple[str, ...]
) -> None:
    """Exercise optional-container handlers with the runtime container supplied."""
    contract = yaml.safe_load((_NODES / node / "contract.yaml").read_text())
    references = {
        entry["handler"]["module"]: entry["handler"]["name"]
        for entry in contract["handler_routing"]["handlers"]
    }
    container = ModelONEXContainer()
    for module in modules:
        module_path = f"omnibase_infra.nodes.{node}.handlers.{module}"
        handler_cls = getattr(
            importlib.import_module(module_path), references[module_path]
        )
        with patch(f"{module_path}.bind_handler_db", wraps=bind_handler_db) as binding:
            handler = handler_cls(container=container)
        binding.assert_called_once_with(container)
        assert handler._db_handler._container is container
        assert handler._db_handler._initialized is False
        assert handler._db_handler._pool is None
