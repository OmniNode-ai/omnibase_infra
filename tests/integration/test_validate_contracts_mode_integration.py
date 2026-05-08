# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for omni-infra validate contracts mode wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from omnibase_infra.cli.commands import cli

pytestmark = pytest.mark.integration


def _effect_contract(name: str = "node_cli_mode_effect") -> dict[str, Any]:
    return {
        "name": name,
        "contract_version": {"major": 1, "minor": 0, "patch": 0},
        "description": f"{name} integration fixture.",
        "node_type": "EFFECT_GENERIC",
        "input_model": "foo.bar.models.ModelFooRequest",
        "output_model": "foo.bar.models.ModelFooResult",
        "handler_routing": {
            "version": {"major": 1, "minor": 0, "patch": 0},
            "routing_strategy": "operation_match",
            "handlers": [
                {
                    "routing_key": "foo.run",
                    "handler_key": "handle_foo_run",
                    "priority": 0,
                }
            ],
        },
        "io_operations": [
            {
                "operation_type": "read",
                "resource_type": "external_api",
                "resource_identifier": "foo.api/run",
            }
        ],
    }


def test_validate_contracts_cli_runs_batch_validator_with_migration_audit_mode(
    tmp_path: Path,
) -> None:
    """Exercise the click command through the real batch validator integration."""
    contract_path = tmp_path / "nodes" / "node_cli_mode_effect" / "contract.yaml"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text(yaml.safe_dump(_effect_contract()), encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        ["validate", "contracts", str(tmp_path), "--mode", "migration_audit"],
    )

    assert result.exit_code == 0, result.output
    assert "mode=migration_audit" in result.output
    assert "1/1 passed" in result.output
