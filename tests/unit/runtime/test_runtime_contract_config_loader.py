# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for RuntimeContractConfigLoader.

This module provides comprehensive unit tests for the RuntimeContractConfigLoader class
that scans directories for contract.yaml files and loads handler_routing and
operation_bindings subcontracts.

Part of OMN-1519: Runtime contract config loader.

Test Categories:
    - TestScanDirectories: Tests for directory scanning functionality
    - TestLoadHandlerRouting: Tests for handler_routing section loading
    - TestLoadOperationBindings: Tests for operation_bindings section loading
    - TestErrorHandling: Tests for graceful error handling
    - TestConsolidatedResults: Tests for result aggregation and metrics
    - TestCorrelationId: Tests for correlation ID handling
    - TestModelRuntimeContractConfig: Tests for the config model
    - TestModelContractLoadResult: Tests for the result model

Running Tests:
    # Run all tests in this module:
    pytest tests/unit/runtime/test_runtime_contract_config_loader.py -v

    # Run specific test class:
    pytest tests/unit/runtime/test_runtime_contract_config_loader.py::TestScanDirectories -v
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.models import (
    ModelContractLoadResult,
    ModelRuntimeContractConfig,
)
from omnibase_infra.runtime.runtime_contract_config_loader import (
    CONTRACT_YAML_FILENAME,
    RuntimeContractConfigLoader,
)

# =============================================================================
# Test Contract YAML Constants
# =============================================================================

VALID_HANDLER_ROUTING_CONTRACT = """
name: "test_node"
version: "1.0.0"
node_type: "EFFECT_GENERIC"
handler_routing:
  routing_strategy: "payload_type_match"
  handlers:
    - event_model:
        name: "TestEventModel"
        module: "test.models.test_event"
      handler:
        name: "HandlerTest"
        module: "test.handlers.handler_test"
"""

VALID_OPERATION_BINDINGS_CONTRACT = """
name: "test_handler"
version: "1.0.0"
node_type: "EFFECT_GENERIC"
operation_bindings:
  version: { major: 1, minor: 0, patch: 0 }
  bindings:
    "db.query":
      - parameter_name: "sql"
        expression: "${payload.sql}"
"""

VALID_COMBINED_CONTRACT = """
name: "combined_node"
version: "1.0.0"
node_type: "ORCHESTRATOR_GENERIC"
handler_routing:
  routing_strategy: "payload_type_match"
  handlers:
    - event_model:
        name: "ModelTestEvent"
        module: "test.models.test_event"
      handler:
        name: "HandlerTestEvent"
        module: "test.handlers.handler_test"
operation_bindings:
  version: { major: 1, minor: 0, patch: 0 }
  global_bindings:
    - parameter_name: "correlation_id"
      expression: "${envelope.correlation_id}"
  bindings:
    "db.execute":
      - parameter_name: "query"
        expression: "${payload.query}"
"""

CONTRACT_WITHOUT_ROUTING_OR_BINDINGS = """
name: "minimal_node"
version: "1.0.0"
node_type: "COMPUTE_GENERIC"
description: "A contract without handler_routing or operation_bindings"
"""

INVALID_YAML_SYNTAX = """
name: "broken_contract"
handler_routing:
  routing_strategy: "payload_type_match"
  handlers: [
    unclosed bracket
"""


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def loader() -> RuntimeContractConfigLoader:
    """Create a RuntimeContractConfigLoader instance."""
    return RuntimeContractConfigLoader()


@pytest.fixture
def valid_handler_routing_contract(tmp_path: Path) -> Path:
    """Create a contract with valid handler_routing."""
    contract_dir = tmp_path / "test_node"
    contract_dir.mkdir()
    contract_file = contract_dir / CONTRACT_YAML_FILENAME
    contract_file.write_text(VALID_HANDLER_ROUTING_CONTRACT)
    return contract_file


@pytest.fixture
def valid_operation_bindings_contract(tmp_path: Path) -> Path:
    """Create a contract with valid operation_bindings."""
    contract_dir = tmp_path / "test_handler"
    contract_dir.mkdir()
    contract_file = contract_dir / CONTRACT_YAML_FILENAME
    contract_file.write_text(VALID_OPERATION_BINDINGS_CONTRACT)
    return contract_file


@pytest.fixture
def valid_combined_contract(tmp_path: Path) -> Path:
    """Create a contract with both handler_routing and operation_bindings."""
    contract_dir = tmp_path / "combined_node"
    contract_dir.mkdir()
    contract_file = contract_dir / CONTRACT_YAML_FILENAME
    contract_file.write_text(VALID_COMBINED_CONTRACT)
    return contract_file


@pytest.fixture
def contract_without_sections(tmp_path: Path) -> Path:
    """Create a contract without handler_routing or operation_bindings."""
    contract_dir = tmp_path / "minimal_node"
    contract_dir.mkdir()
    contract_file = contract_dir / CONTRACT_YAML_FILENAME
    contract_file.write_text(CONTRACT_WITHOUT_ROUTING_OR_BINDINGS)
    return contract_file


@pytest.fixture
def invalid_yaml_contract(tmp_path: Path) -> Path:
    """Create a contract with invalid YAML syntax."""
    contract_dir = tmp_path / "broken_node"
    contract_dir.mkdir()
    contract_file = contract_dir / CONTRACT_YAML_FILENAME
    contract_file.write_text(INVALID_YAML_SYNTAX)
    return contract_file


@pytest.fixture
def empty_search_path(tmp_path: Path) -> Path:
    """Create an empty directory for search."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    return empty_dir


@pytest.fixture
def nested_contracts_path(tmp_path: Path) -> Path:
    """Create a directory with nested contract.yaml files."""
    # Level 1 contract
    level1 = tmp_path / "nodes" / "level1"
    level1.mkdir(parents=True)
    (level1 / CONTRACT_YAML_FILENAME).write_text(VALID_HANDLER_ROUTING_CONTRACT)

    # Level 2 contract (nested)
    level2 = tmp_path / "nodes" / "level1" / "level2"
    level2.mkdir(parents=True)
    (level2 / CONTRACT_YAML_FILENAME).write_text(VALID_OPERATION_BINDINGS_CONTRACT)

    # Sibling contract
    sibling = tmp_path / "nodes" / "sibling"
    sibling.mkdir(parents=True)
    (sibling / CONTRACT_YAML_FILENAME).write_text(VALID_COMBINED_CONTRACT)

    return tmp_path / "nodes"


# =============================================================================
# TestScanDirectories
# =============================================================================


class TestScanDirectories:
    """Tests for directory scanning functionality."""

    def test_empty_search_paths_returns_empty_config(
        self,
        loader: RuntimeContractConfigLoader,
        empty_search_path: Path,
    ) -> None:
        """Empty directories should return config with no contracts."""
        config = loader.load_all_contracts(search_paths=[empty_search_path])

        assert config.total_contracts_found == 0
        assert config.total_contracts_loaded == 0
        assert config.total_errors == 0
        assert len(config.contract_results) == 0

    def test_nonexistent_path_logs_warning_and_continues(
        self,
        loader: RuntimeContractConfigLoader,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Nonexistent search paths should log warning but not fail."""
        import logging

        nonexistent = tmp_path / "nonexistent"

        with caplog.at_level(logging.WARNING):
            config = loader.load_all_contracts(search_paths=[nonexistent])

        assert config.total_contracts_found == 0
        assert any("does not exist" in record.message for record in caplog.records)

    def test_file_path_logs_warning_and_continues(
        self,
        loader: RuntimeContractConfigLoader,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """File paths (not directories) should log warning but not fail."""
        import logging

        file_path = tmp_path / "not_a_directory.txt"
        file_path.write_text("just a file")

        with caplog.at_level(logging.WARNING):
            config = loader.load_all_contracts(search_paths=[file_path])

        assert config.total_contracts_found == 0
        assert any("not a directory" in record.message for record in caplog.records)

    def test_finds_single_contract_yaml(
        self,
        loader: RuntimeContractConfigLoader,
        valid_handler_routing_contract: Path,
    ) -> None:
        """Should find a single contract.yaml file."""
        search_path = valid_handler_routing_contract.parent.parent

        config = loader.load_all_contracts(search_paths=[search_path])

        assert config.total_contracts_found == 1
        assert config.total_contracts_loaded == 1

    def test_finds_multiple_contract_yaml_files(
        self,
        loader: RuntimeContractConfigLoader,
        nested_contracts_path: Path,
    ) -> None:
        """Should find all contract.yaml files in nested directories."""
        config = loader.load_all_contracts(search_paths=[nested_contracts_path])

        # Should find 3 contracts: level1, level2, and sibling
        assert config.total_contracts_found == 3
        assert config.total_contracts_loaded == 3

    def test_nested_directories_are_scanned(
        self,
        loader: RuntimeContractConfigLoader,
        nested_contracts_path: Path,
    ) -> None:
        """Should scan nested directories recursively."""
        config = loader.load_all_contracts(search_paths=[nested_contracts_path])

        # Verify we found contracts at different nesting levels
        paths = [r.contract_path for r in config.contract_results]

        # Check that we have paths at different depths
        path_str = [str(p) for p in paths]
        assert any(
            "level1" in p and "level2" not in p.split("level1/")[1]
            for p in path_str
            if "level1" in p
        )
        assert any("level2" in p for p in path_str)
        assert any("sibling" in p for p in path_str)

    def test_multiple_search_paths_combined(
        self,
        loader: RuntimeContractConfigLoader,
        tmp_path: Path,
    ) -> None:
        """Should combine results from multiple search paths."""
        # Create two separate search paths
        path1 = tmp_path / "path1" / "node1"
        path1.mkdir(parents=True)
        (path1 / CONTRACT_YAML_FILENAME).write_text(VALID_HANDLER_ROUTING_CONTRACT)

        path2 = tmp_path / "path2" / "node2"
        path2.mkdir(parents=True)
        (path2 / CONTRACT_YAML_FILENAME).write_text(VALID_OPERATION_BINDINGS_CONTRACT)

        config = loader.load_all_contracts(
            search_paths=[tmp_path / "path1", tmp_path / "path2"]
        )

        assert config.total_contracts_found == 2
        assert config.total_contracts_loaded == 2

    def test_results_are_sorted_by_path(
        self,
        loader: RuntimeContractConfigLoader,
        tmp_path: Path,
    ) -> None:
        """Contract paths should be sorted for deterministic ordering."""
        # Create contracts in non-alphabetical order
        for name in ["zebra", "alpha", "middle"]:
            node_dir = tmp_path / name
            node_dir.mkdir()
            (node_dir / CONTRACT_YAML_FILENAME).write_text(
                CONTRACT_WITHOUT_ROUTING_OR_BINDINGS
            )

        config = loader.load_all_contracts(search_paths=[tmp_path])

        paths = [r.contract_path for r in config.contract_results]
        assert paths == sorted(paths)


# =============================================================================
# TestLoadHandlerRouting
# =============================================================================


class TestLoadHandlerRouting:
    """Tests for handler_routing section loading."""

    def test_loads_valid_handler_routing(
        self,
        loader: RuntimeContractConfigLoader,
        valid_handler_routing_contract: Path,
    ) -> None:
        """Contract with valid handler_routing should be loaded."""
        search_path = valid_handler_routing_contract.parent.parent

        config = loader.load_all_contracts(search_paths=[search_path])

        assert config.total_contracts_loaded == 1
        routing_configs = config.handler_routing_configs
        assert len(routing_configs) == 1

        # Verify handler routing content
        routing = next(iter(routing_configs.values()))
        assert routing.routing_strategy == "payload_type_match"
        assert len(routing.handlers) == 1
        assert routing.handlers[0].routing_key == "TestEventModel"

    def test_contract_without_handler_routing_returns_none(
        self,
        loader: RuntimeContractConfigLoader,
        contract_without_sections: Path,
    ) -> None:
        """Contract without handler_routing should have None for that field."""
        search_path = contract_without_sections.parent.parent

        config = loader.load_all_contracts(search_paths=[search_path])

        assert config.total_contracts_loaded == 1
        assert len(config.handler_routing_configs) == 0

        # Verify result has no handler_routing
        result = config.contract_results[0]
        assert result.handler_routing is None

    def test_handler_routing_with_multiple_handlers(
        self,
        loader: RuntimeContractConfigLoader,
        tmp_path: Path,
    ) -> None:
        """Contract with multiple handlers should load all handlers."""
        contract_content = """
name: "multi_handler_node"
version: "1.0.0"
node_type: "ORCHESTRATOR_GENERIC"
handler_routing:
  routing_strategy: "payload_type_match"
  handlers:
    - event_model:
        name: "EventOne"
        module: "test.models.one"
      handler:
        name: "HandlerOne"
        module: "test.handlers.one"
    - event_model:
        name: "EventTwo"
        module: "test.models.two"
      handler:
        name: "HandlerTwo"
        module: "test.handlers.two"
    - event_model:
        name: "EventThree"
        module: "test.models.three"
      handler:
        name: "HandlerThree"
        module: "test.handlers.three"
"""
        node_dir = tmp_path / "multi_handler"
        node_dir.mkdir()
        (node_dir / CONTRACT_YAML_FILENAME).write_text(contract_content)

        config = loader.load_all_contracts(search_paths=[tmp_path])

        routing = next(iter(config.handler_routing_configs.values()))
        assert len(routing.handlers) == 3


# =============================================================================
# TestLoadOperationBindings
# =============================================================================


class TestLoadOperationBindings:
    """Tests for operation_bindings section loading."""

    def test_loads_valid_operation_bindings(
        self,
        loader: RuntimeContractConfigLoader,
        valid_operation_bindings_contract: Path,
    ) -> None:
        """Contract with valid operation_bindings should be loaded."""
        search_path = valid_operation_bindings_contract.parent.parent

        config = loader.load_all_contracts(search_paths=[search_path])

        assert config.total_contracts_loaded == 1
        bindings_configs = config.operation_bindings_configs
        assert len(bindings_configs) == 1

        # Verify operation bindings content
        bindings = next(iter(bindings_configs.values()))
        assert "db.query" in bindings.bindings
        assert len(bindings.bindings["db.query"]) == 1
        assert bindings.bindings["db.query"][0].parameter_name == "sql"

    def test_contract_without_operation_bindings_returns_none(
        self,
        loader: RuntimeContractConfigLoader,
        contract_without_sections: Path,
    ) -> None:
        """Contract without operation_bindings should have None for that field."""
        search_path = contract_without_sections.parent.parent

        config = loader.load_all_contracts(search_paths=[search_path])

        assert config.total_contracts_loaded == 1
        assert len(config.operation_bindings_configs) == 0

        # Verify result has no operation_bindings
        result = config.contract_results[0]
        assert result.operation_bindings is None

    def test_loads_global_bindings(
        self,
        loader: RuntimeContractConfigLoader,
        valid_combined_contract: Path,
    ) -> None:
        """Contract with global_bindings should load them."""
        search_path = valid_combined_contract.parent.parent

        config = loader.load_all_contracts(search_paths=[search_path])

        bindings = next(iter(config.operation_bindings_configs.values()))
        assert bindings.global_bindings is not None
        assert len(bindings.global_bindings) == 1
        assert bindings.global_bindings[0].parameter_name == "correlation_id"


# =============================================================================
# TestErrorHandling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_yaml_handles_gracefully(
        self,
        loader: RuntimeContractConfigLoader,
        invalid_yaml_contract: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Invalid YAML syntax should be logged but not crash loading."""
        import logging

        search_path = invalid_yaml_contract.parent.parent

        with caplog.at_level(logging.WARNING):
            config = loader.load_all_contracts(search_paths=[search_path])

        assert config.total_contracts_found == 1
        assert config.total_contracts_loaded == 0
        assert config.total_errors == 1

        # Verify error is recorded
        assert len(config.failed_results) == 1
        assert config.failed_results[0].error != ""

    def test_missing_sections_handled_gracefully(
        self,
        loader: RuntimeContractConfigLoader,
        contract_without_sections: Path,
    ) -> None:
        """Missing sections should return None, not error."""
        search_path = contract_without_sections.parent.parent

        config = loader.load_all_contracts(search_paths=[search_path])

        # Should succeed even without handler_routing or operation_bindings
        assert config.total_contracts_loaded == 1
        assert config.total_errors == 0

        result = config.contract_results[0]
        assert result.success
        assert result.handler_routing is None
        assert result.operation_bindings is None

    def test_errors_collected_in_load_errors(
        self,
        loader: RuntimeContractConfigLoader,
        invalid_yaml_contract: Path,
    ) -> None:
        """Errors should be collected in the error_messages property."""
        search_path = invalid_yaml_contract.parent.parent

        config = loader.load_all_contracts(search_paths=[search_path])

        error_messages = config.error_messages
        assert len(error_messages) == 1
        assert invalid_yaml_contract in error_messages

    def test_one_bad_contract_does_not_affect_others(
        self,
        loader: RuntimeContractConfigLoader,
        tmp_path: Path,
    ) -> None:
        """A single bad contract should not stop loading of other contracts."""
        # Create a valid contract
        valid_dir = tmp_path / "valid_node"
        valid_dir.mkdir()
        (valid_dir / CONTRACT_YAML_FILENAME).write_text(VALID_HANDLER_ROUTING_CONTRACT)

        # Create an invalid contract
        invalid_dir = tmp_path / "invalid_node"
        invalid_dir.mkdir()
        (invalid_dir / CONTRACT_YAML_FILENAME).write_text(INVALID_YAML_SYNTAX)

        config = loader.load_all_contracts(search_paths=[tmp_path])

        assert config.total_contracts_found == 2
        assert config.total_contracts_loaded == 1
        assert config.total_errors == 1

    def test_load_single_contract_missing_file_raises_error(
        self,
        loader: RuntimeContractConfigLoader,
        tmp_path: Path,
    ) -> None:
        """load_single_contract should raise error for missing file."""
        nonexistent = tmp_path / "nonexistent" / CONTRACT_YAML_FILENAME

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_single_contract(nonexistent)

        assert "not found" in str(exc_info.value).lower()


# =============================================================================
# TestConsolidatedResults
# =============================================================================


class TestConsolidatedResults:
    """Tests for consolidated result aggregation."""

    def test_total_contracts_found_is_correct(
        self,
        loader: RuntimeContractConfigLoader,
        nested_contracts_path: Path,
    ) -> None:
        """total_contracts_found should count all discovered contracts."""
        config = loader.load_all_contracts(search_paths=[nested_contracts_path])

        assert config.total_contracts_found == 3

    def test_total_contracts_loaded_is_correct(
        self,
        loader: RuntimeContractConfigLoader,
        tmp_path: Path,
    ) -> None:
        """total_contracts_loaded should count only successful loads."""
        # Create 2 valid and 1 invalid contract
        for i, content in enumerate(
            [VALID_HANDLER_ROUTING_CONTRACT, VALID_OPERATION_BINDINGS_CONTRACT]
        ):
            node_dir = tmp_path / f"valid_{i}"
            node_dir.mkdir()
            (node_dir / CONTRACT_YAML_FILENAME).write_text(content)

        invalid_dir = tmp_path / "invalid"
        invalid_dir.mkdir()
        (invalid_dir / CONTRACT_YAML_FILENAME).write_text(INVALID_YAML_SYNTAX)

        config = loader.load_all_contracts(search_paths=[tmp_path])

        assert config.total_contracts_found == 3
        assert config.total_contracts_loaded == 2
        assert config.total_errors == 1

    def test_success_rate_calculated_correctly(
        self,
        loader: RuntimeContractConfigLoader,
        tmp_path: Path,
    ) -> None:
        """success_rate should be calculated correctly."""
        # Create 3 valid and 1 invalid contract (75% success)
        for i in range(3):
            node_dir = tmp_path / f"valid_{i}"
            node_dir.mkdir()
            (node_dir / CONTRACT_YAML_FILENAME).write_text(
                CONTRACT_WITHOUT_ROUTING_OR_BINDINGS
            )

        invalid_dir = tmp_path / "invalid"
        invalid_dir.mkdir()
        (invalid_dir / CONTRACT_YAML_FILENAME).write_text(INVALID_YAML_SYNTAX)

        config = loader.load_all_contracts(search_paths=[tmp_path])

        assert config.success_rate == 0.75

    def test_success_rate_is_1_when_no_contracts(
        self,
        loader: RuntimeContractConfigLoader,
        empty_search_path: Path,
    ) -> None:
        """success_rate should be 1.0 when no contracts are found."""
        config = loader.load_all_contracts(search_paths=[empty_search_path])

        assert config.success_rate == 1.0

    def test_handler_routing_configs_dict_has_correct_paths(
        self,
        loader: RuntimeContractConfigLoader,
        valid_handler_routing_contract: Path,
    ) -> None:
        """handler_routing_configs should map paths to routing subcontracts."""
        search_path = valid_handler_routing_contract.parent.parent

        config = loader.load_all_contracts(search_paths=[search_path])

        assert valid_handler_routing_contract in config.handler_routing_configs

    def test_operation_bindings_configs_dict_has_correct_paths(
        self,
        loader: RuntimeContractConfigLoader,
        valid_operation_bindings_contract: Path,
    ) -> None:
        """operation_bindings_configs should map paths to bindings subcontracts."""
        search_path = valid_operation_bindings_contract.parent.parent

        config = loader.load_all_contracts(search_paths=[search_path])

        assert valid_operation_bindings_contract in config.operation_bindings_configs

    def test_all_successful_property(
        self,
        loader: RuntimeContractConfigLoader,
        tmp_path: Path,
    ) -> None:
        """all_successful should be True only when no errors."""
        # All successful
        for i in range(2):
            node_dir = tmp_path / f"valid_{i}"
            node_dir.mkdir()
            (node_dir / CONTRACT_YAML_FILENAME).write_text(
                CONTRACT_WITHOUT_ROUTING_OR_BINDINGS
            )

        config = loader.load_all_contracts(search_paths=[tmp_path])
        assert config.all_successful

        # Now add an invalid contract
        invalid_dir = tmp_path / "invalid"
        invalid_dir.mkdir()
        (invalid_dir / CONTRACT_YAML_FILENAME).write_text(INVALID_YAML_SYNTAX)

        config2 = loader.load_all_contracts(search_paths=[tmp_path])
        assert not config2.all_successful


# =============================================================================
# TestCorrelationId
# =============================================================================


class TestCorrelationId:
    """Tests for correlation ID handling."""

    def test_provided_correlation_id_is_used(
        self,
        loader: RuntimeContractConfigLoader,
        valid_handler_routing_contract: Path,
    ) -> None:
        """Provided correlation_id should be used in result."""
        search_path = valid_handler_routing_contract.parent.parent
        correlation_id = uuid4()

        config = loader.load_all_contracts(
            search_paths=[search_path],
            correlation_id=correlation_id,
        )

        assert config.correlation_id == correlation_id

    def test_auto_generates_correlation_id(
        self,
        loader: RuntimeContractConfigLoader,
        valid_handler_routing_contract: Path,
    ) -> None:
        """Should auto-generate correlation_id if not provided."""
        search_path = valid_handler_routing_contract.parent.parent

        config = loader.load_all_contracts(search_paths=[search_path])

        assert config.correlation_id is not None
        assert isinstance(config.correlation_id, UUID)

    def test_correlation_id_propagated_to_results(
        self,
        loader: RuntimeContractConfigLoader,
        valid_handler_routing_contract: Path,
    ) -> None:
        """Correlation ID should be propagated to individual results."""
        search_path = valid_handler_routing_contract.parent.parent
        correlation_id = uuid4()

        config = loader.load_all_contracts(
            search_paths=[search_path],
            correlation_id=correlation_id,
        )

        for result in config.contract_results:
            assert result.correlation_id == correlation_id

    def test_load_single_contract_uses_correlation_id(
        self,
        loader: RuntimeContractConfigLoader,
        valid_handler_routing_contract: Path,
    ) -> None:
        """load_single_contract should use provided correlation_id."""
        correlation_id = uuid4()

        result = loader.load_single_contract(
            valid_handler_routing_contract,
            correlation_id=correlation_id,
        )

        assert result.correlation_id == correlation_id

    def test_load_single_contract_auto_generates_correlation_id(
        self,
        loader: RuntimeContractConfigLoader,
        valid_handler_routing_contract: Path,
    ) -> None:
        """load_single_contract should auto-generate correlation_id."""
        result = loader.load_single_contract(valid_handler_routing_contract)

        assert result.correlation_id is not None
        assert isinstance(result.correlation_id, UUID)


# =============================================================================
# TestModelRuntimeContractConfig
# =============================================================================


class TestModelRuntimeContractConfig:
    """Tests for ModelRuntimeContractConfig model."""

    def test_bool_returns_true_when_all_successful(self) -> None:
        """__bool__ should return True when all contracts loaded successfully."""
        config = ModelRuntimeContractConfig(
            contract_results=[
                ModelContractLoadResult.succeeded(
                    contract_path=Path("/test/contract.yaml")
                )
            ],
            total_contracts_found=1,
            total_contracts_loaded=1,
            total_errors=0,
        )

        assert bool(config) is True

    def test_bool_returns_false_when_errors(self) -> None:
        """__bool__ should return False when there are errors."""
        config = ModelRuntimeContractConfig(
            contract_results=[
                ModelContractLoadResult.failed(
                    contract_path=Path("/test/contract.yaml"),
                    error="Test error",
                )
            ],
            total_contracts_found=1,
            total_contracts_loaded=0,
            total_errors=1,
        )

        assert bool(config) is False

    def test_bool_returns_false_when_no_contracts_found(self) -> None:
        """__bool__ should return False when no contracts found."""
        config = ModelRuntimeContractConfig(
            contract_results=[],
            total_contracts_found=0,
            total_contracts_loaded=0,
            total_errors=0,
        )

        assert bool(config) is False

    def test_successful_results_property(self) -> None:
        """successful_results should filter to only successful loads."""
        config = ModelRuntimeContractConfig(
            contract_results=[
                ModelContractLoadResult.succeeded(
                    contract_path=Path("/test/success.yaml")
                ),
                ModelContractLoadResult.failed(
                    contract_path=Path("/test/fail.yaml"),
                    error="Test error",
                ),
            ],
            total_contracts_found=2,
            total_contracts_loaded=1,
            total_errors=1,
        )

        successful = config.successful_results
        assert len(successful) == 1
        assert successful[0].contract_path == Path("/test/success.yaml")

    def test_failed_results_property(self) -> None:
        """failed_results should filter to only failed loads."""
        config = ModelRuntimeContractConfig(
            contract_results=[
                ModelContractLoadResult.succeeded(
                    contract_path=Path("/test/success.yaml")
                ),
                ModelContractLoadResult.failed(
                    contract_path=Path("/test/fail.yaml"),
                    error="Test error",
                ),
            ],
            total_contracts_found=2,
            total_contracts_loaded=1,
            total_errors=1,
        )

        failed = config.failed_results
        assert len(failed) == 1
        assert failed[0].contract_path == Path("/test/fail.yaml")

    def test_get_routing_for_contract(
        self,
        loader: RuntimeContractConfigLoader,
        valid_handler_routing_contract: Path,
    ) -> None:
        """get_routing_for_contract should return routing for specific path."""
        search_path = valid_handler_routing_contract.parent.parent

        config = loader.load_all_contracts(search_paths=[search_path])

        routing = config.get_routing_for_contract(valid_handler_routing_contract)
        assert routing is not None
        assert routing.routing_strategy == "payload_type_match"

    def test_get_routing_for_contract_returns_none_if_not_found(
        self,
        loader: RuntimeContractConfigLoader,
        valid_handler_routing_contract: Path,
    ) -> None:
        """get_routing_for_contract should return None for unknown path."""
        search_path = valid_handler_routing_contract.parent.parent

        config = loader.load_all_contracts(search_paths=[search_path])

        routing = config.get_routing_for_contract(Path("/nonexistent/contract.yaml"))
        assert routing is None

    def test_get_bindings_for_contract(
        self,
        loader: RuntimeContractConfigLoader,
        valid_operation_bindings_contract: Path,
    ) -> None:
        """get_bindings_for_contract should return bindings for specific path."""
        search_path = valid_operation_bindings_contract.parent.parent

        config = loader.load_all_contracts(search_paths=[search_path])

        bindings = config.get_bindings_for_contract(valid_operation_bindings_contract)
        assert bindings is not None
        assert "db.query" in bindings.bindings

    def test_str_representation(self) -> None:
        """__str__ should return human-readable summary."""
        config = ModelRuntimeContractConfig(
            contract_results=[],
            total_contracts_found=5,
            total_contracts_loaded=4,
            total_errors=1,
        )

        str_repr = str(config)
        assert "found=5" in str_repr
        assert "loaded=4" in str_repr
        assert "errors=1" in str_repr


# =============================================================================
# TestModelContractLoadResult
# =============================================================================


class TestModelContractLoadResult:
    """Tests for ModelContractLoadResult model."""

    def test_succeeded_factory_method(self) -> None:
        """succeeded() should create a successful result."""
        result = ModelContractLoadResult.succeeded(
            contract_path=Path("/test/contract.yaml"),
        )

        assert result.success
        assert result.error == ""
        assert result.contract_path == Path("/test/contract.yaml")

    def test_failed_factory_method(self) -> None:
        """failed() should create a failed result."""
        result = ModelContractLoadResult.failed(
            contract_path=Path("/test/contract.yaml"),
            error="Test error message",
        )

        assert not result.success
        assert result.error == "Test error message"
        assert result.handler_routing is None
        assert result.operation_bindings is None

    def test_bool_returns_success_value(self) -> None:
        """__bool__ should return the success value."""
        success_result = ModelContractLoadResult.succeeded(
            contract_path=Path("/test/contract.yaml")
        )
        failed_result = ModelContractLoadResult.failed(
            contract_path=Path("/test/contract.yaml"),
            error="Error",
        )

        assert bool(success_result) is True
        assert bool(failed_result) is False

    def test_has_error_property(self) -> None:
        """has_error should check if error message exists."""
        success_result = ModelContractLoadResult.succeeded(
            contract_path=Path("/test/contract.yaml")
        )
        failed_result = ModelContractLoadResult.failed(
            contract_path=Path("/test/contract.yaml"),
            error="Error",
        )

        assert not success_result.has_error
        assert failed_result.has_error

    def test_has_handler_routing_property(
        self,
        loader: RuntimeContractConfigLoader,
        valid_handler_routing_contract: Path,
    ) -> None:
        """has_handler_routing should check if routing was loaded."""
        result = loader.load_single_contract(valid_handler_routing_contract)

        assert result.has_handler_routing

    def test_has_operation_bindings_property(
        self,
        loader: RuntimeContractConfigLoader,
        valid_operation_bindings_contract: Path,
    ) -> None:
        """has_operation_bindings should check if bindings were loaded."""
        result = loader.load_single_contract(valid_operation_bindings_contract)

        assert result.has_operation_bindings

    def test_str_representation_success(self) -> None:
        """__str__ should show loaded subcontracts for success."""
        result = ModelContractLoadResult.succeeded(
            contract_path=Path("/test/contract.yaml"),
        )

        str_repr = str(result)
        assert "/test/contract.yaml" in str_repr
        assert "no subcontracts" in str_repr

    def test_str_representation_failed(self) -> None:
        """__str__ should show error for failed result."""
        result = ModelContractLoadResult.failed(
            contract_path=Path("/test/contract.yaml"),
            error="Test error",
        )

        str_repr = str(result)
        assert "/test/contract.yaml" in str_repr
        assert "FAILED" in str_repr
        assert "Test error" in str_repr


# =============================================================================
# TestContractYamlFilename
# =============================================================================


class TestContractYamlFilename:
    """Tests for CONTRACT_YAML_FILENAME constant."""

    def test_constant_value(self) -> None:
        """CONTRACT_YAML_FILENAME should be 'contract.yaml'."""
        assert CONTRACT_YAML_FILENAME == "contract.yaml"

    def test_constant_is_exported(self) -> None:
        """CONTRACT_YAML_FILENAME should be in __all__."""
        from omnibase_infra.runtime import runtime_contract_config_loader

        assert "CONTRACT_YAML_FILENAME" in runtime_contract_config_loader.__all__


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "TestConsolidatedResults",
    "TestContractYamlFilename",
    "TestCorrelationId",
    "TestErrorHandling",
    "TestLoadHandlerRouting",
    "TestLoadOperationBindings",
    "TestModelContractLoadResult",
    "TestModelRuntimeContractConfig",
    "TestScanDirectories",
]
