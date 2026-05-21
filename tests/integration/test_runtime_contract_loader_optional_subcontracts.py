# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for optional runtime contract subcontracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.runtime.runtime_contract_config_loader import (
    RuntimeContractConfigLoader,
)

pytestmark = pytest.mark.integration


def test_runtime_loader_skips_contract_without_handler_routing(tmp_path: Path) -> None:
    """Startup loader accepts contracts that omit optional handler_routing."""
    contract_dir = tmp_path / "nodes" / "no_routing_node"
    contract_dir.mkdir(parents=True)
    (contract_dir / "contract.yaml").write_text(
        "\n".join(
            [
                'name: "no_routing_node"',
                'version: "1.0.0"',
                'node_type: "EFFECT_GENERIC"',
                'description: "Contract intentionally omits handler_routing"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = RuntimeContractConfigLoader().load_all_contracts([tmp_path / "nodes"])

    assert config.total_contracts_found == 1
    assert config.total_contracts_loaded == 1
    assert config.total_errors == 0
    assert config.contract_results[0].success is True
    assert config.contract_results[0].handler_routing is None
