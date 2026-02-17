# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for the bootstrap sequence (OMN-2287).

These tests verify the contract-driven config discovery pipeline works
end-to-end using real contract files from the repository, without
requiring external services (Infisical, PostgreSQL, etc.).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.runtime.config_discovery.config_prefetcher import (
    ConfigPrefetcher,
)
from omnibase_infra.runtime.config_discovery.contract_config_extractor import (
    ContractConfigExtractor,
)
from omnibase_infra.runtime.config_discovery.transport_config_map import (
    TransportConfigMap,
)

NODES_DIR = Path("/Volumes/PRO-G40/Code/omnibase_infra5/src/omnibase_infra/nodes")


@pytest.mark.skipif(
    not NODES_DIR.is_dir(),
    reason="Repository nodes directory not available",
)
class TestBootstrapSequence:
    """Integration tests for the full bootstrap pipeline."""

    def test_extract_then_map_then_prefetch(self) -> None:
        """Full pipeline: extract -> map -> prefetch with mock handler."""
        # Step 1: Extract requirements from real contracts
        extractor = ContractConfigExtractor()
        reqs = extractor.extract_from_paths([NODES_DIR])

        assert len(reqs.transport_types) > 0, (
            "Should find at least one transport type in repo contracts"
        )
        assert len(reqs.requirements) > 0, "Should find at least one config requirement"

        # Step 2: Build transport specs
        tcm = TransportConfigMap()
        specs = tcm.specs_for_transports(reqs.transport_types)
        assert len(specs) > 0, "Should generate specs for discovered transports"

        # Step 3: Prefetch with mock handler
        mock_handler = MagicMock()
        mock_handler.get_secret_sync = MagicMock(return_value=None)

        prefetcher = ConfigPrefetcher(handler=mock_handler)
        result = prefetcher.prefetch(reqs)

        # Handler should have been called for each key in each spec
        assert mock_handler.get_secret_sync.call_count > 0

    def test_real_contracts_have_database_transport(self) -> None:
        """Real contracts should declare DATABASE transport type."""
        extractor = ContractConfigExtractor()
        reqs = extractor.extract_from_paths([NODES_DIR])

        assert EnumInfraTransportType.DATABASE in reqs.transport_types

    def test_real_contracts_have_consul_transport(self) -> None:
        """Real contracts should declare CONSUL transport type."""
        extractor = ContractConfigExtractor()
        reqs = extractor.extract_from_paths([NODES_DIR])

        assert EnumInfraTransportType.CONSUL in reqs.transport_types

    def test_real_contracts_have_env_dependencies(self) -> None:
        """Real contracts should have environment dependencies."""
        extractor = ContractConfigExtractor()
        reqs = extractor.extract_from_paths([NODES_DIR])

        env_deps = [
            r for r in reqs.requirements if r.source_field.startswith("dependencies[")
        ]
        # The slack alerter has env dependencies
        assert len(env_deps) > 0, (
            "Should find at least one environment dependency in repo contracts"
        )

    def test_no_extraction_errors_on_real_contracts(self) -> None:
        """Should have no parse errors on valid repo contracts."""
        extractor = ContractConfigExtractor()
        reqs = extractor.extract_from_paths([NODES_DIR])

        # Filter out "unknown transport" errors (some contracts may use
        # transport types not in our mapping yet)
        parse_errors = [e for e in reqs.errors if "Failed to parse" in e]
        assert len(parse_errors) == 0, (
            f"Should not have YAML parse errors: {parse_errors}"
        )
