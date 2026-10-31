# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The modules that build intent targets hold no URL literal (OMN-21792).

Runs core's contracts-and-DI validator over each module that used to spell a
``postgres://`` target, and over a planted URL as the positive control.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_core.validation.validator_contracts_and_di import (
    ValidatorContractsAndDi,
)

pytestmark = pytest.mark.unit

_SRC = Path(__file__).resolve().parents[4] / "src" / "omnibase_infra"

_CONSUMERS: tuple[str, ...] = (
    "nodes/node_build_loop_projection_compute/handlers/handler_build_loop_projection.py",
    "nodes/node_contract_registry_reducer/reducer.py",
    "nodes/node_gateway_link_health_projection_compute/handlers/handler_gateway_link_health_projection.py",
    "nodes/node_ledger_projection_compute/handlers/handler_ledger_projection.py",
    "nodes/node_registration_orchestrator/services/registration_reducer_service.py",
    "nodes/node_registration_reducer/registration_reducer.py",
    "nodes/node_validation_ledger_projection_compute/handlers/handler_validation_ledger_projection.py",
    "models/intent_target_uris/model_intent_target_uris.py",
)


def _rules_reported(path: Path) -> list[str]:
    result = ValidatorContractsAndDi().validate([path])
    return [issue.rule_name or "" for issue in result.issues]


def test_the_check_reports_a_planted_url(tmp_path: Path) -> None:
    planted = tmp_path / "planted_module.py"
    planted.write_text('TARGET = "postgres://planted/1"\n', encoding="utf-8")
    assert "literal_url" in _rules_reported(planted)


@pytest.mark.parametrize("module", _CONSUMERS)
def test_consumer_holds_no_url_literal(module: str) -> None:
    assert "literal_url" not in _rules_reported(_SRC / module)
