# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handlers and nodes are built at a binding site, never by hand (OMN-21623).

Runs core's contracts-and-DI validator over the modules that used to build a
``HandlerDb`` by hand, with a planted construction as the positive control, and
checks that the node packages whose registry factories nothing called no longer
hold one.
"""

from __future__ import annotations

import ast
import importlib
from collections.abc import Callable
from importlib.util import resolve_name
from pathlib import Path

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_core.validation.validator_contracts_and_di import (
    ValidatorContractsAndDi,
)
from omnibase_infra.handlers.handler_db import HandlerDb

pytestmark = pytest.mark.unit

_SRC = Path(__file__).resolve().parents[4] / "src" / "omnibase_infra"

_RULE = "handler_hand_construction"

_DB_COMPOSING_HANDLERS: tuple[str, ...] = (
    "nodes/node_build_loop_write_effect/handlers/handler_build_loop_append.py",
    "nodes/node_delegation_chain_ledger_effect/handlers/handler_delegation_chain_ledger.py",
    "nodes/node_gateway_link_health_write_effect/handlers/handler_gateway_link_health_upsert.py",
    "nodes/node_ledger_write_effect/handlers/handler_ledger_append.py",
    "nodes/node_ledger_write_effect/handlers/handler_ledger_query.py",
    "nodes/node_validation_ledger_write_effect/handlers/handler_validation_ledger_append.py",
)

_BINDING_NODES: tuple[str, ...] = (
    "node_build_loop_write_effect",
    "node_delegation_chain_ledger_effect",
    "node_gateway_link_health_write_effect",
    "node_ledger_write_effect",
    "node_validation_ledger_write_effect",
)

_NODES_WITHOUT_REGISTRY_FACTORY: tuple[str, ...] = (
    "node_architecture_validator",
    "node_artifact_reconciliation_orchestrator",
    "node_baseline_capture",
    "node_baseline_comparison_compute",
    "node_baselines_batch_compute",
    "node_build_loop_projection_compute",
    "node_build_loop_write_effect",
    "node_contract_registry_reducer",
    "node_decision_store_effect",
    "node_decision_store_query_compute",
    "node_delta_bundle_effect",
    "node_delta_metrics_effect",
    "node_gateway_link_health_projection_compute",
    "node_gateway_link_health_write_effect",
    "node_impact_analyzer_compute",
    "node_merge_gate_effect",
    "node_registry_effect",
    "node_release_identity_compute",
    "node_rrh_storage_effect",
    "node_rrh_validate_compute",
    "node_savings_estimation_compute",
    "node_update_plan_reducer",
    "node_validation_ledger_projection_compute",
    "node_validation_orchestrator",
)


def _rules_reported(path: Path) -> list[str]:
    assert path.is_file(), f"validator input missing: {path}"
    result = ValidatorContractsAndDi().validate([path])
    return [issue.rule_name or "" for issue in result.issues]


@pytest.mark.parametrize("method", ["__init__", "handle"])
def test_the_check_reports_a_planted_construction(tmp_path: Path, method: str) -> None:
    planted = tmp_path / "planted_module.py"
    planted.write_text(
        "from omnibase_infra.handlers.handler_db import HandlerDb\n\n"
        "class HandlerPlanted:\n"
        f"    def {method}(self, container):\n"
        "        self._db_handler = HandlerDb(container)\n",
        encoding="utf-8",
    )
    assert _RULE in _rules_reported(planted)


def _binder(node: str) -> Callable[[ModelONEXContainer], HandlerDb]:
    module = importlib.import_module(
        f"omnibase_infra.nodes.{node}.container.container_handler_db"
    )
    binder: Callable[[ModelONEXContainer], HandlerDb] = module.bind_handler_db
    return binder


@pytest.mark.parametrize("node", _BINDING_NODES)
def test_the_binding_site_builds_the_handler(node: str) -> None:
    container = ModelONEXContainer()
    handler = _binder(node)(container)
    assert isinstance(handler, HandlerDb)
    assert handler._container is container


@pytest.mark.parametrize("node", _BINDING_NODES)
def test_each_binding_is_owned_by_its_caller(node: str) -> None:
    container = ModelONEXContainer()
    first = _binder(node)(container)
    second = _binder(node)(container)
    assert first is not second
    assert first._container is second._container is container


@pytest.mark.parametrize("node", _BINDING_NODES)
def test_the_binding_site_is_not_reported(node: str) -> None:
    site = _SRC / "nodes" / node / "container" / "container_handler_db.py"
    rules = _rules_reported(site)
    assert _RULE not in rules


@pytest.mark.parametrize("module", _DB_COMPOSING_HANDLERS)
def test_db_composing_handler_builds_nothing_by_hand(module: str) -> None:
    path = _SRC / module
    assert _RULE not in _rules_reported(path)


@pytest.mark.parametrize("node", _NODES_WITHOUT_REGISTRY_FACTORY)
def test_node_package_has_no_dead_registry_factory(node: str) -> None:
    package = _SRC / "nodes" / node
    assert package.is_dir(), f"node package missing: {package}"
    assert (package / "__init__.py").is_file()
    assert not (package / "registry").exists()


def test_dead_registry_factory_has_no_importers() -> None:
    removed = tuple(
        f"omnibase_infra.nodes.{node}.registry"
        for node in _NODES_WITHOUT_REGISTRY_FACTORY
    )
    sources = list(_SRC.rglob("*.py"))
    assert sources
    stale: list[str] = []
    for source in sources:
        package = ".".join(source.relative_to(_SRC.parent).parts[:-1])
        for statement in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if not isinstance(statement, (ast.Import, ast.ImportFrom)):
                continue
            names: list[str] = []
            if isinstance(statement, ast.Import):
                names = [alias.name for alias in statement.names]
            elif isinstance(statement, ast.ImportFrom):
                module = "." * statement.level + (statement.module or "")
                if statement.level:
                    module = resolve_name(module, package)
                names = [
                    module,
                    *(f"{module}.{alias.name}" for alias in statement.names),
                ]
            for name in names:
                if any(
                    name == prefix or name.startswith(prefix + ".")
                    for prefix in removed
                ):
                    stale.append(
                        f"{source.relative_to(_SRC)}:{statement.lineno}: {name}"
                    )
    assert not stale, "\n".join(stale)
