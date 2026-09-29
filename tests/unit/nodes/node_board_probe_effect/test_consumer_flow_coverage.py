# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Coverage is enforced against executable contract declarations."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from omnibase_infra.nodes.node_board_probe_effect.coverage_validator import (
    validate_coverage,
)

pytestmark = pytest.mark.unit
NODE = (
    Path(__file__).resolve().parents[4]
    / "src/omnibase_infra/nodes/node_board_probe_effect"
)


@pytest.mark.parametrize(
    ("fixture", "message"),
    [("phantom", "no handler"), ("missing", "no coverage"), ("reason", "reason")],
)
def test_refusal_fixtures(fixture: str, message: str) -> None:
    body = yaml.safe_load(
        (Path(__file__).parent / "fixtures" / f"coverage_{fixture}.yaml").read_text()
    )
    with pytest.raises(ValueError, match=message):
        validate_coverage(body["board_checks"], body["coverage"])


def test_empty_coverage_refused() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_coverage([], [])


def test_real_files_and_cli_and_hook() -> None:
    contract = yaml.safe_load((NODE / "contract.yaml").read_text())
    coverage = yaml.safe_load((NODE / "board_check_coverage.yaml").read_text())[
        "coverage"
    ]
    validate_coverage(contract["board_checks"], coverage)
    assert {r["check_id"] for r in coverage} == {
        "forwarder_refused_topic",
        "consumer_flow",
    }
    routes = {r["operation"] for r in contract["handler_routing"]["handlers"]}
    assert all(r["operation"] in routes for r in contract["board_checks"])
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "omnibase_infra.nodes.node_board_probe_effect.coverage_validator",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    hooks = yaml.safe_load((NODE.parents[3] / ".pre-commit-config.yaml").read_text())
    assert any(
        "node_board_probe_effect.coverage_validator" in h.get("entry", "")
        and h.get("always_run")
        for repo in hooks["repos"]
        for h in repo["hooks"]
    )


def test_node_yaml_passes_existing_contract_validators() -> None:
    from omnibase_core.validation import validate_contracts
    from omnibase_infra.validation.linter_contract import lint_contracts_in_directory

    yaml_result = validate_contracts(str(NODE))
    assert yaml_result.is_valid, yaml_result.errors
    lint_result = lint_contracts_in_directory(
        str(NODE), check_imports=True, strict_mode=False, check_dependencies=False
    )
    assert lint_result.is_valid, lint_result.violations
