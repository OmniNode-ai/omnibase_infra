# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ContractConfigExtractor (OMN-2287)."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.runtime.config_discovery.contract_config_extractor import (
    ContractConfigExtractor,
    _resolve_transport,
)
from tests.helpers.path_utils import find_project_root


class TestResolveTransport:
    """Tests for the _resolve_transport helper."""

    def test_resolve_db(self) -> None:
        assert _resolve_transport("db") == EnumInfraTransportType.DATABASE

    def test_resolve_database_alias(self) -> None:
        assert _resolve_transport("database") == EnumInfraTransportType.DATABASE

    def test_resolve_postgres_alias(self) -> None:
        assert _resolve_transport("postgres") == EnumInfraTransportType.DATABASE

    def test_resolve_kafka(self) -> None:
        assert _resolve_transport("kafka") == EnumInfraTransportType.KAFKA

    def test_resolve_consul(self) -> None:
        assert _resolve_transport("consul") == EnumInfraTransportType.CONSUL

    def test_resolve_infisical(self) -> None:
        assert _resolve_transport("infisical") == EnumInfraTransportType.INFISICAL

    def test_resolve_unknown_returns_none(self) -> None:
        assert _resolve_transport("nonexistent") is None

    def test_resolve_case_insensitive(self) -> None:
        assert _resolve_transport("DATABASE") == EnumInfraTransportType.DATABASE

    def test_resolve_strips_whitespace(self) -> None:
        assert _resolve_transport("  kafka  ") == EnumInfraTransportType.KAFKA


class TestContractConfigExtractor:
    """Tests for ContractConfigExtractor."""

    def setup_method(self) -> None:
        self.extractor = ContractConfigExtractor()

    def test_extract_from_db_contract(self, tmp_path: Path) -> None:
        """Should extract DATABASE transport from db handler contract."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            """
name: "handler_db"
node_type: "EFFECT_GENERIC"
description: "Test"
handler_routing:
  routing_strategy: "operation_match"
  handlers:
    - handler_type: "db"
      handler:
        name: "HandlerDb"
        module: "omnibase_infra.handlers.handler_db"
metadata:
  handler_id: "db-handler"
  transport_type: "database"
"""
        )
        reqs = self.extractor.extract_from_yaml(contract)

        assert EnumInfraTransportType.DATABASE in reqs.transport_types
        assert any(r.key == "POSTGRES_HOST" for r in reqs.requirements)
        assert contract in reqs.contract_paths
        assert len(reqs.errors) == 0

    def test_extract_from_consul_contract(self, tmp_path: Path) -> None:
        """Should extract CONSUL transport from consul handler contract."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            """
name: "handler_consul"
metadata:
  transport_type: "consul"
handler_routing:
  routing_strategy: "operation_match"
  handlers:
    - handler_type: "consul"
      handler:
        name: "HandlerConsul"
        module: "omnibase_infra.handlers.handler_consul"
"""
        )
        reqs = self.extractor.extract_from_yaml(contract)

        assert EnumInfraTransportType.CONSUL in reqs.transport_types
        assert any(r.key == "CONSUL_HOST" for r in reqs.requirements)

    def test_extract_env_dependencies(self, tmp_path: Path) -> None:
        """Should extract environment-type dependencies."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            """
name: "slack_alerter"
dependencies:
  - name: "slack_webhook_url"
    type: "environment"
    env_var: "SLACK_WEBHOOK_URL"
    required: false
  - name: "slack_bot_token"
    type: "environment"
    env_var: "SLACK_BOT_TOKEN"
    required: false
  - name: "http_client"
    type: "library"
    library: "aiohttp"
"""
        )
        reqs = self.extractor.extract_from_yaml(contract)

        # Should have env dependency requirements
        env_reqs = [
            r for r in reqs.requirements if r.source_field.startswith("dependencies[")
        ]
        assert len(env_reqs) == 2
        assert env_reqs[0].key == "SLACK_WEBHOOK_URL"
        assert env_reqs[0].required is False
        assert env_reqs[1].key == "SLACK_BOT_TOKEN"

    def test_extract_unknown_transport_type(self, tmp_path: Path) -> None:
        """Should log error for unknown transport types."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            """
name: "test"
metadata:
  transport_type: "nonexistent_transport"
"""
        )
        reqs = self.extractor.extract_from_yaml(contract)

        assert len(reqs.errors) == 1
        assert "nonexistent_transport" in reqs.errors[0]

    def test_extract_invalid_yaml(self, tmp_path: Path) -> None:
        """Should handle invalid YAML gracefully."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(":::invalid yaml::: [\n")

        reqs = self.extractor.extract_from_yaml(contract)
        assert len(reqs.errors) == 1
        assert "Failed to parse" in reqs.errors[0]

    def test_extract_empty_contract(self, tmp_path: Path) -> None:
        """Should handle empty contract gracefully."""
        contract = tmp_path / "contract.yaml"
        contract.write_text("")

        reqs = self.extractor.extract_from_yaml(contract)
        assert len(reqs.requirements) == 0
        assert len(reqs.errors) == 0

    def test_extract_from_directory(self, tmp_path: Path) -> None:
        """Should recursively find contract.yaml files in directories."""
        # Create nested structure
        (tmp_path / "handlers" / "db").mkdir(parents=True)
        (tmp_path / "handlers" / "consul").mkdir(parents=True)

        (tmp_path / "handlers" / "db" / "contract.yaml").write_text(
            """
name: "handler_db"
metadata:
  transport_type: "database"
"""
        )
        (tmp_path / "handlers" / "consul" / "contract.yaml").write_text(
            """
name: "handler_consul"
metadata:
  transport_type: "consul"
"""
        )

        reqs = self.extractor.extract_from_paths([tmp_path])

        assert EnumInfraTransportType.DATABASE in reqs.transport_types
        assert EnumInfraTransportType.CONSUL in reqs.transport_types
        assert len(reqs.contract_paths) == 2

    def test_extract_named_contract_variant(self, tmp_path: Path) -> None:
        """Should find contract_*.yaml files in addition to contract.yaml (OMN-2995).

        Repos that adopt the named-contract convention (e.g.
        ``contract_omniclaude_runtime.yaml``) must be visible to the seeder
        without renaming or moving the file.
        """
        (tmp_path / "contracts").mkdir()
        (tmp_path / "contracts" / "contract_omniclaude_runtime.yaml").write_text(
            """
name: "omniclaude_runtime"
dependencies:
  - name: "contracts_root"
    type: "environment"
    env_var: "OMNICLAUDE_CONTRACTS_ROOT"
    required: true
"""
        )
        reqs = self.extractor.extract_from_paths([tmp_path])

        assert len(reqs.contract_paths) == 1
        dep_keys = [r.key for r in reqs.requirements]
        assert "OMNICLAUDE_CONTRACTS_ROOT" in dep_keys

    def test_extract_named_and_canonical_contracts_combined(
        self, tmp_path: Path
    ) -> None:
        """Should find both contract.yaml and contract_*.yaml in the same tree (OMN-2995).

        Verifies that both naming conventions co-exist correctly and that
        deduplication prevents any path from appearing twice.
        """
        (tmp_path / "node_a").mkdir()
        (tmp_path / "node_b").mkdir()

        (tmp_path / "node_a" / "contract.yaml").write_text(
            """
name: "node_a"
metadata:
  transport_type: "database"
"""
        )
        (tmp_path / "node_b" / "contract_node_b_runtime.yaml").write_text(
            """
name: "node_b_runtime"
dependencies:
  - name: "some_key"
    type: "environment"
    env_var: "NODE_B_RUNTIME_KEY"
    required: false
"""
        )
        reqs = self.extractor.extract_from_paths([tmp_path])

        assert len(reqs.contract_paths) == 2
        assert EnumInfraTransportType.DATABASE in reqs.transport_types
        dep_keys = [r.key for r in reqs.requirements]
        assert "NODE_B_RUNTIME_KEY" in dep_keys

    def test_extract_named_contract_glob_does_not_match_canonical(
        self, tmp_path: Path
    ) -> None:
        """contract_*.yaml glob must NOT match contract.yaml (OMN-2995).

        The ``contract_*.yaml`` pattern requires an underscore before the
        suffix, so ``contract.yaml`` is NOT a match. This test confirms that
        a directory containing only ``contract.yaml`` yields exactly one
        scanned file (not two from overlapping glob results).
        """
        (tmp_path / "contract.yaml").write_text(
            """
name: "canonical"
metadata:
  transport_type: "consul"
"""
        )
        reqs = self.extractor.extract_from_paths([tmp_path])

        # Only one file -- contract_paths must have exactly one entry.
        assert len(reqs.contract_paths) == 1
        assert EnumInfraTransportType.CONSUL in reqs.transport_types

    def test_extract_from_nonexistent_path(self) -> None:
        """Should handle nonexistent paths gracefully."""
        reqs = self.extractor.extract_from_paths([Path("/nonexistent/path")])
        assert len(reqs.errors) == 1
        assert "does not exist" in reqs.errors[0]

    def test_extract_deduplicates_transport_types(self, tmp_path: Path) -> None:
        """Should deduplicate transport types across contracts."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            """
name: "handler_db"
metadata:
  transport_type: "database"
handler_routing:
  handlers:
    - handler_type: "db"
      handler:
        name: "H"
        module: "m"
"""
        )
        reqs = self.extractor.extract_from_yaml(contract)

        # Both metadata.transport_type and handler_routing point to DATABASE
        # but transport_types should be deduplicated
        assert reqs.transport_types.count(EnumInfraTransportType.DATABASE) == 1

    def test_extract_from_real_contracts(self) -> None:
        """Should extract requirements from actual repo contracts."""
        repo_root = find_project_root(Path(__file__).resolve().parent)
        nodes_dir = repo_root / "src" / "omnibase_infra" / "nodes"
        if not nodes_dir.is_dir():
            pytest.skip("Repo nodes directory not available")

        reqs = self.extractor.extract_from_paths([nodes_dir])

        # Should find multiple transport types from real contracts
        assert len(reqs.transport_types) > 0
        assert len(reqs.requirements) > 0
        assert len(reqs.contract_paths) > 0

    def test_merge_requirements(self, tmp_path: Path) -> None:
        """Should merge requirements from multiple extractions."""
        c1 = tmp_path / "c1.yaml"
        c1.write_text('name: "a"\nmetadata:\n  transport_type: "database"\n')

        c2 = tmp_path / "c2.yaml"
        c2.write_text('name: "b"\nmetadata:\n  transport_type: "consul"\n')

        reqs1 = self.extractor.extract_from_yaml(c1)
        reqs2 = self.extractor.extract_from_yaml(c2)
        merged = reqs1.merge(reqs2)

        assert EnumInfraTransportType.DATABASE in merged.transport_types
        assert EnumInfraTransportType.CONSUL in merged.transport_types
        assert len(merged.contract_paths) == 2


class TestTransportAliasesNewEntries:
    """Parametrized tests for the 12 new aliases added to _TRANSPORT_ALIASES."""

    @pytest.mark.parametrize(
        ("alias", "expected_transport"),
        [
            ("architecture_validation", EnumInfraTransportType.RUNTIME),
            ("auth_gate", EnumInfraTransportType.RUNTIME),
            ("ledger_projection", EnumInfraTransportType.RUNTIME),
            ("validation_ledger_projection", EnumInfraTransportType.RUNTIME),
            ("rrh_validate", EnumInfraTransportType.RUNTIME),
            ("runtime_target", EnumInfraTransportType.RUNTIME),
            ("toolchain", EnumInfraTransportType.RUNTIME),
            ("mock", EnumInfraTransportType.INMEMORY),
            ("intent", EnumInfraTransportType.GRAPH),
            ("memgraph", EnumInfraTransportType.GRAPH),
            ("repo_state", EnumInfraTransportType.FILESYSTEM),
            ("rrh_storage", EnumInfraTransportType.FILESYSTEM),
        ],
    )
    def test_alias_resolves_to_expected_transport(
        self, alias: str, expected_transport: EnumInfraTransportType
    ) -> None:
        """Each new alias should resolve to its expected transport type."""
        assert _resolve_transport(alias) == expected_transport

    def test_routing_state_not_in_aliases(self) -> None:
        """routing_state should NOT be in aliases (intentionally excluded)."""
        assert _resolve_transport("routing_state") is None
