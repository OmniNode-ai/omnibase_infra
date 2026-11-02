# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21683: the slice paths hold no handler_hand_construction finding.

Runs core's contracts-and-DI validator, the same reference the burn-down
measures with, over each file of this slice and asserts the
``handler_hand_construction`` rule reports nothing there. A file that builds a
handler or node by hand again turns this test red.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_core.validation.validator_contracts_and_di import (
    ValidatorContractsAndDi,
)

_SRC = Path(__file__).resolve().parents[3] / "src" / "omnibase_infra"
_RULE = "handler_hand_construction"

SLICE_PATHS: tuple[str, ...] = (
    "nodes/node_registration_orchestrator/wiring.py",
    "nodes/node_llm_completion_effect/registry/registry_infra_llm_completion_effect.py",
    "nodes/node_auth_gate_compute/registry/registry_infra_auth_gate_compute.py",
    "nodes/node_contract_resolver_bridge/registry/registry_infra_contract_resolver_bridge.py",
    "nodes/node_ledger_projection_compute/registry/registry_infra_ledger_projection.py",
    "nodes/node_registration_reducer/registry/registry_infra_node_registration_reducer.py",
    "nodes/node_registry_api_effect/registry/registry_infra_registry_api_effect.py",
    "nodes/node_session_lifecycle_reducer/registry/registry_infra_session_lifecycle.py",
    "nodes/node_session_state_effect/registry/registry_infra_session_state.py",
    "nodes/node_validation_adjudicator/registry/registry_infra_validation_adjudicator.py",
    "nodes/node_validation_executor/registry/registry_infra_validation_executor.py",
)


def _hand_construction_findings(path: Path) -> list[str]:
    validator = ValidatorContractsAndDi()
    result = validator.validate_file(path)
    return [
        f"{path}:{issue.line_number}: {issue.message}"
        for issue in result.issues
        if issue.rule_name == _RULE
    ]


@pytest.mark.unit
@pytest.mark.parametrize("relative", SLICE_PATHS)
def test_slice_path_has_no_hand_construction(relative: str) -> None:
    path = _SRC / relative
    assert path.is_file(), f"slice path missing: {path}"
    assert _hand_construction_findings(path) == []


@pytest.mark.unit
def test_rule_still_fires_on_hand_construction(tmp_path: Path) -> None:
    """Positive control: the rule reports a hand-built handler."""
    probe = tmp_path / "probe_module.py"
    probe.write_text(
        "class HandlerProbe:\n    pass\n\n\nvalue = HandlerProbe()\n",
        encoding="utf-8",
    )
    assert len(_hand_construction_findings(probe)) == 1
