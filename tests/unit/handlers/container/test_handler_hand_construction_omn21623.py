# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handlers and nodes are built at a binding site, never by hand (OMN-21623).

Runs core's contracts-and-DI validator over the modules that used to build a
``HandlerDb`` by hand, with a planted construction as the positive control, and
checks that the node packages whose registry factories nothing called no longer
hold one.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from omnibase_core.validation.validator_contracts_and_di import (
    ValidatorContractsAndDi,
)
from omnibase_infra.handlers.container.container_handler_db import bind_handler_db
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
    result = ValidatorContractsAndDi().validate([path])
    return [issue.rule_name or "" for issue in result.issues]


def test_the_check_reports_a_planted_construction(tmp_path: Path) -> None:
    planted = tmp_path / "planted_module.py"
    planted.write_text(
        "from somewhere import HandlerPlanted\n\nplanted = HandlerPlanted()\n",
        encoding="utf-8",
    )
    assert _RULE in _rules_reported(planted)


def test_the_binding_site_builds_the_handler() -> None:
    assert isinstance(bind_handler_db(MagicMock()), HandlerDb)


def test_the_binding_site_is_not_reported() -> None:
    site = _SRC / "handlers" / "container" / "container_handler_db.py"
    assert _RULE not in _rules_reported(site)


@pytest.mark.parametrize("module", _DB_COMPOSING_HANDLERS)
def test_db_composing_handler_builds_nothing_by_hand(module: str) -> None:
    assert _RULE not in _rules_reported(_SRC / module)


@pytest.mark.parametrize("node", _NODES_WITHOUT_REGISTRY_FACTORY)
def test_node_package_has_no_dead_registry_factory(node: str) -> None:
    assert not (_SRC / "nodes" / node / "registry").exists()
