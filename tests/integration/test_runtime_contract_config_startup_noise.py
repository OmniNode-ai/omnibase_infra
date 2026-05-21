# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for runtime contract startup noise."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from omnibase_infra.runtime.runtime_contract_config_loader import (
    RuntimeContractConfigLoader,
)


@pytest.mark.integration
def test_runtime_contract_loader_omits_missing_handler_routing_error(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Startup loader skips optional missing handler_routing without error logs."""
    node_dir = tmp_path / "node_without_routing"
    node_dir.mkdir()
    (node_dir / "contract.yaml").write_text(
        """
name: "node_without_routing"
version: "1.0.0"
node_type: "COMPUTE_GENERIC"
description: "Valid node contract with no optional handler_routing subcontract"
""",
        encoding="utf-8",
    )

    loader = RuntimeContractConfigLoader()

    with caplog.at_level(logging.ERROR):
        config = loader.load_all_contracts(search_paths=[tmp_path])

    assert config.total_contracts_loaded == 1
    assert config.total_errors == 0
    assert config.contract_results[0].handler_routing is None
    assert not any(
        "MISSING_HANDLER_ROUTING" in record.message
        or "Missing 'handler_routing' section" in record.message
        for record in caplog.records
    )
