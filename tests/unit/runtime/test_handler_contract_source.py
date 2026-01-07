# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Unit tests for HandlerContractSource filesystem discovery.

TDD RED Phase - Tests written before implementation.

Tests the HandlerContractSource functionality including:
- Recursive discovery of handler_contract.yaml files in nested directories
- Transformation of contracts to ModelHandlerDescriptor instances
- Contract validation during discovery
- Error handling for malformed contracts

Related:
    - OMN-1097: HandlerContractSource + Filesystem Discovery
    - src/omnibase_infra/runtime/handler_contract_source.py (to be created)
    - docs/architecture/HANDLER_PROTOCOL_DRIVEN_ARCHITECTURE.md

Expected Behavior:
    HandlerContractSource implements ProtocolHandlerSource from omnibase_spi.
    It discovers handler contracts from the filesystem by recursively scanning
    configured paths for handler_contract.yaml files, parsing them, and
    transforming them into ProtocolHandlerDescriptor instances.

    The source_type property returns "CONTRACT" as per the protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import pytest

# Protocol imports with fallback for compatibility
# The protocols may be in different locations depending on omnibase_spi version
try:
    from omnibase_spi.protocols.handlers.protocol_handler_source import (
        ProtocolHandlerSource,
    )
except ImportError:
    # Fallback: define minimal protocol stub for testing
    # This allows the test to run and fail on HandlerContractSource import

    @runtime_checkable
    class ProtocolHandlerSource(Protocol):
        """Fallback protocol definition for testing."""

        @property
        def source_type(self) -> str:
            """The type of handler source."""
            ...

        async def discover_handlers(self) -> list:
            """Discover and return all handlers from this source."""
            ...


try:
    from omnibase_spi.protocols.handlers.types import ProtocolHandlerDescriptor
except ImportError:
    # Fallback: define minimal protocol stub for testing

    @runtime_checkable
    class ProtocolHandlerDescriptor(Protocol):
        """Fallback protocol definition for testing."""

        @property
        def handler_id(self) -> str:
            """Unique identifier for the handler."""
            ...

        @property
        def name(self) -> str:
            """Human-readable name for the handler."""
            ...

        @property
        def version(self) -> str:
            """Semantic version of the handler."""
            ...


# =============================================================================
# Constants for Test Contracts
# =============================================================================

MINIMAL_HANDLER_CONTRACT_YAML = """
handler_id: "{handler_id}"
name: "{name}"
version: "1.0.0"
descriptor:
  handler_kind: "compute"
input_model: "test.models.Input"
output_model: "test.models.Output"
"""

HANDLER_CONTRACT_WITH_METADATA_YAML = """
handler_id: "{handler_id}"
name: "{name}"
version: "{version}"
descriptor:
  handler_kind: "{handler_kind}"
  description: "{description}"
input_model: "{input_model}"
output_model: "{output_model}"
metadata:
  category: "{category}"
  priority: {priority}
"""


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def nested_contract_structure(tmp_path: Path) -> dict[str, Path]:
    """Create a nested directory structure with handler_contract.yaml files.

    Structure:
        tmp_path/
        |-- level1/
        |   |-- handler_contract.yaml  (handler: level1.handler)
        |   |-- level2/
        |   |   |-- handler_contract.yaml  (handler: level1.level2.handler)
        |   |   |-- level3/
        |   |   |   |-- handler_contract.yaml  (handler: level1.level2.level3.handler)

    Returns:
        Dictionary mapping handler_id to contract file path
    """
    contracts: dict[str, Path] = {}

    # Level 1 contract
    level1_dir = tmp_path / "level1"
    level1_dir.mkdir(parents=True)
    level1_contract = level1_dir / "handler_contract.yaml"
    level1_contract.write_text(
        MINIMAL_HANDLER_CONTRACT_YAML.format(
            handler_id="level1.handler",
            name="Level 1 Handler",
        )
    )
    contracts["level1.handler"] = level1_contract

    # Level 2 contract (nested in level1)
    level2_dir = level1_dir / "level2"
    level2_dir.mkdir(parents=True)
    level2_contract = level2_dir / "handler_contract.yaml"
    level2_contract.write_text(
        MINIMAL_HANDLER_CONTRACT_YAML.format(
            handler_id="level1.level2.handler",
            name="Level 2 Handler",
        )
    )
    contracts["level1.level2.handler"] = level2_contract

    # Level 3 contract (nested in level1/level2)
    level3_dir = level2_dir / "level3"
    level3_dir.mkdir(parents=True)
    level3_contract = level3_dir / "handler_contract.yaml"
    level3_contract.write_text(
        MINIMAL_HANDLER_CONTRACT_YAML.format(
            handler_id="level1.level2.level3.handler",
            name="Level 3 Handler",
        )
    )
    contracts["level1.level2.level3.handler"] = level3_contract

    return contracts


@pytest.fixture
def single_contract_path(tmp_path: Path) -> Path:
    """Create a single directory with one handler_contract.yaml file.

    Returns:
        Path to the directory containing the contract file.
    """
    contract_dir = tmp_path / "single_handler"
    contract_dir.mkdir(parents=True)
    contract_file = contract_dir / "handler_contract.yaml"
    contract_file.write_text(
        MINIMAL_HANDLER_CONTRACT_YAML.format(
            handler_id="single.test.handler",
            name="Single Test Handler",
        )
    )
    return contract_dir


@pytest.fixture
def empty_directory(tmp_path: Path) -> Path:
    """Create an empty directory with no contracts.

    Returns:
        Path to the empty directory.
    """
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir(parents=True)
    return empty_dir


@pytest.fixture
def malformed_contract_path(tmp_path: Path) -> Path:
    """Create a directory with a malformed handler_contract.yaml file.

    Returns:
        Path to the directory containing the malformed contract file.
    """
    malformed_dir = tmp_path / "malformed"
    malformed_dir.mkdir(parents=True)
    malformed_file = malformed_dir / "handler_contract.yaml"
    malformed_file.write_text(
        """
this is not valid yaml: [
    unclosed bracket
handler_id: "missing"
"""
    )
    return malformed_dir


# =============================================================================
# HandlerContractSource Import Tests
# =============================================================================


class TestHandlerContractSourceImport:
    """Tests for HandlerContractSource import and instantiation.

    These tests verify the class can be imported from the expected location
    and implements the ProtocolHandlerSource protocol.

    RED Phase: These tests will FAIL with ImportError until implementation.
    """

    def test_handler_contract_source_can_be_imported(self) -> None:
        """HandlerContractSource should be importable from omnibase_infra.runtime.

        RED Phase: This test WILL FAIL until HandlerContractSource is implemented.

        Expected import path:
            from omnibase_infra.runtime.handler_contract_source import HandlerContractSource
        """
        # This import will fail in RED phase - that's expected!
        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        assert HandlerContractSource is not None

    def test_handler_contract_source_implements_protocol(
        self, single_contract_path: Path
    ) -> None:
        """HandlerContractSource should implement ProtocolHandlerSource.

        RED Phase: This test WILL FAIL until HandlerContractSource is implemented.

        The implementation must satisfy the ProtocolHandlerSource protocol from
        omnibase_spi with:
        - source_type property returning "CONTRACT"
        - async discover_handlers() method returning list[ProtocolHandlerDescriptor]
        """
        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        source = HandlerContractSource(contract_paths=[single_contract_path])

        # Protocol compliance check via duck typing (ONEX convention)
        assert hasattr(source, "source_type")
        assert hasattr(source, "discover_handlers")
        assert callable(source.discover_handlers)

        # Runtime checkable protocol verification
        assert isinstance(source, ProtocolHandlerSource)

    def test_handler_contract_source_type_is_contract(
        self, single_contract_path: Path
    ) -> None:
        """HandlerContractSource.source_type should return "CONTRACT".

        RED Phase: This test WILL FAIL until HandlerContractSource is implemented.

        The source_type is used for observability and debugging purposes only.
        The runtime MUST NOT branch on this value.
        """
        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        source = HandlerContractSource(contract_paths=[single_contract_path])

        assert source.source_type == "CONTRACT"


# =============================================================================
# Nested Contract Discovery Tests
# =============================================================================


class TestHandlerContractSourceDiscovery:
    """Tests for HandlerContractSource.discover_handlers() functionality.

    These tests verify that HandlerContractSource correctly discovers
    handler_contract.yaml files in nested directory structures and transforms
    them into ProtocolHandlerDescriptor instances.

    RED Phase: These tests will FAIL until implementation exists.
    """

    @pytest.mark.asyncio
    async def test_discovers_nested_contracts(
        self, tmp_path: Path, nested_contract_structure: dict[str, Path]
    ) -> None:
        """discover_handlers() should find contracts in nested directories.

        RED Phase: This test WILL FAIL until HandlerContractSource is implemented.

        The source should recursively scan all configured paths for files matching
        the pattern **/handler_contract.yaml and return descriptors for each.

        Structure being scanned:
            tmp_path/
            |-- level1/
            |   |-- handler_contract.yaml  -> handler_id: level1.handler
            |   |-- level2/
            |   |   |-- handler_contract.yaml  -> handler_id: level1.level2.handler
            |   |   |-- level3/
            |   |   |   |-- handler_contract.yaml  -> handler_id: level1.level2.level3.handler

        Expected: 3 descriptors discovered
        """
        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        # Configure source with the tmp_path as contract search path
        source = HandlerContractSource(contract_paths=[tmp_path])

        # Discover handlers from nested structure
        descriptors = await source.discover_handlers()

        # Verify all 3 contracts were discovered
        assert len(descriptors) == 3, (
            f"Expected 3 descriptors from nested structure, got {len(descriptors)}"
        )

        # Verify each descriptor is a ProtocolHandlerDescriptor
        for descriptor in descriptors:
            assert isinstance(descriptor, ProtocolHandlerDescriptor)

        # Verify the expected handler_ids were discovered
        discovered_ids = {d.handler_id for d in descriptors}
        expected_ids = set(nested_contract_structure.keys())
        assert discovered_ids == expected_ids, (
            f"Handler ID mismatch. Expected: {expected_ids}, Got: {discovered_ids}"
        )

    @pytest.mark.asyncio
    async def test_discovers_single_contract(self, single_contract_path: Path) -> None:
        """discover_handlers() should find a single contract in a directory.

        RED Phase: This test WILL FAIL until HandlerContractSource is implemented.
        """
        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        source = HandlerContractSource(contract_paths=[single_contract_path])

        descriptors = await source.discover_handlers()

        assert len(descriptors) == 1
        assert descriptors[0].handler_id == "single.test.handler"

    @pytest.mark.asyncio
    async def test_returns_empty_list_for_empty_directory(
        self, empty_directory: Path
    ) -> None:
        """discover_handlers() should return empty list when no contracts found.

        RED Phase: This test WILL FAIL until HandlerContractSource is implemented.
        """
        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        source = HandlerContractSource(contract_paths=[empty_directory])

        descriptors = await source.discover_handlers()

        assert descriptors == []

    @pytest.mark.asyncio
    async def test_discovers_from_multiple_paths(
        self,
        tmp_path: Path,
        single_contract_path: Path,
        nested_contract_structure: dict[str, Path],
    ) -> None:
        """discover_handlers() should aggregate contracts from multiple paths.

        RED Phase: This test WILL FAIL until HandlerContractSource is implemented.

        When multiple contract_paths are provided, all should be scanned and
        results aggregated into a single list.
        """
        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        # Configure source with multiple search paths
        source = HandlerContractSource(contract_paths=[single_contract_path, tmp_path])

        descriptors = await source.discover_handlers()

        # Should find: 1 from single_contract_path + 3 from nested structure
        assert len(descriptors) == 4

    @pytest.mark.asyncio
    async def test_descriptors_have_required_properties(
        self, single_contract_path: Path
    ) -> None:
        """Discovered descriptors should have all required properties.

        RED Phase: This test WILL FAIL until HandlerContractSource is implemented.

        Each descriptor must have:
        - handler_id: str
        - name: str
        - version: str
        - handler_kind: str
        - input_model: str
        - output_model: str
        """
        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        source = HandlerContractSource(contract_paths=[single_contract_path])

        descriptors = await source.discover_handlers()

        assert len(descriptors) == 1
        descriptor = descriptors[0]

        # Verify required properties exist and have correct values
        assert descriptor.handler_id == "single.test.handler"
        assert descriptor.name == "Single Test Handler"
        assert descriptor.version == "1.0.0"
        assert hasattr(descriptor, "handler_kind")
        assert hasattr(descriptor, "input_model")
        assert hasattr(descriptor, "output_model")


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestHandlerContractSourceErrors:
    """Tests for error handling in HandlerContractSource.

    These tests verify proper error handling for invalid contracts,
    missing files, and other failure scenarios.

    RED Phase: These tests will FAIL until implementation exists.
    """

    @pytest.mark.asyncio
    async def test_raises_on_malformed_yaml(
        self, malformed_contract_path: Path
    ) -> None:
        """discover_handlers() should raise for malformed YAML contracts.

        RED Phase: This test WILL FAIL until HandlerContractSource is implemented.

        Malformed YAML should result in a clear error indicating which file
        failed to parse, not a generic YAML parsing error.
        """
        from omnibase_core.models.errors.model_onex_error import ModelOnexError

        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        source = HandlerContractSource(contract_paths=[malformed_contract_path])

        with pytest.raises(ModelOnexError) as exc_info:
            await source.discover_handlers()

        # Error should indicate contract parsing failure
        assert (
            "contract" in str(exc_info.value).lower()
            or "yaml" in str(exc_info.value).lower()
        )

    @pytest.mark.asyncio
    async def test_raises_on_nonexistent_path(self, tmp_path: Path) -> None:
        """discover_handlers() should raise for non-existent contract paths.

        RED Phase: This test WILL FAIL until HandlerContractSource is implemented.
        """
        from omnibase_core.models.errors.model_onex_error import ModelOnexError

        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        nonexistent_path = tmp_path / "does_not_exist"

        source = HandlerContractSource(contract_paths=[nonexistent_path])

        with pytest.raises(ModelOnexError) as exc_info:
            await source.discover_handlers()

        assert (
            "exist" in str(exc_info.value).lower()
            or "not found" in str(exc_info.value).lower()
        )

    def test_raises_on_empty_contract_paths(self) -> None:
        """HandlerContractSource should raise if contract_paths is empty.

        RED Phase: This test WILL FAIL until HandlerContractSource is implemented.
        """
        from omnibase_core.models.errors.model_onex_error import ModelOnexError

        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        with pytest.raises(ModelOnexError) as exc_info:
            HandlerContractSource(contract_paths=[])

        assert (
            "empty" in str(exc_info.value).lower()
            or "required" in str(exc_info.value).lower()
        )


# =============================================================================
# Idempotency Tests
# =============================================================================


class TestHandlerContractSourceIdempotency:
    """Tests for idempotency of discover_handlers().

    Per ProtocolHandlerSource contract, discover_handlers() may be called
    multiple times and should return consistent results.

    RED Phase: These tests will FAIL until implementation exists.
    """

    @pytest.mark.asyncio
    async def test_discover_handlers_is_idempotent(
        self, nested_contract_structure: dict[str, Path], tmp_path: Path
    ) -> None:
        """discover_handlers() should return same results on multiple calls.

        RED Phase: This test WILL FAIL until HandlerContractSource is implemented.
        """
        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        source = HandlerContractSource(contract_paths=[tmp_path])

        # Call discover_handlers multiple times
        result1 = await source.discover_handlers()
        result2 = await source.discover_handlers()
        result3 = await source.discover_handlers()

        # All results should be identical
        assert len(result1) == len(result2) == len(result3) == 3

        ids1 = {d.handler_id for d in result1}
        ids2 = {d.handler_id for d in result2}
        ids3 = {d.handler_id for d in result3}

        assert ids1 == ids2 == ids3


# =============================================================================
# Contract File Pattern Tests
# =============================================================================


class TestHandlerContractSourceFilePattern:
    """Tests for the file pattern used by HandlerContractSource.

    The source should only discover files named exactly 'handler_contract.yaml',
    ignoring other YAML files and variations.

    RED Phase: These tests will FAIL until implementation exists.
    """

    @pytest.mark.asyncio
    async def test_ignores_other_yaml_files(self, tmp_path: Path) -> None:
        """discover_handlers() should only find handler_contract.yaml files.

        RED Phase: This test WILL FAIL until HandlerContractSource is implemented.

        Other YAML files (e.g., config.yaml, contract.yaml) should be ignored.
        """
        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        # Create handler_contract.yaml (should be discovered)
        handler_dir = tmp_path / "handlers"
        handler_dir.mkdir(parents=True)
        (handler_dir / "handler_contract.yaml").write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_id="valid.handler",
                name="Valid Handler",
            )
        )

        # Create other YAML files that should be IGNORED
        (handler_dir / "config.yaml").write_text("some: config")
        (handler_dir / "contract.yaml").write_text("different: contract")
        (handler_dir / "handler_contract.yml").write_text(
            "wrong: extension"
        )  # Wrong extension

        # Put case-different file in separate directory to avoid filesystem
        # case-sensitivity issues (macOS/Windows volumes may be case-insensitive)
        case_test_dir = tmp_path / "case_test"
        case_test_dir.mkdir(parents=True)
        (case_test_dir / "HANDLER_CONTRACT.yaml").write_text(
            "wrong: case"
        )  # Wrong case - should not be discovered

        source = HandlerContractSource(contract_paths=[tmp_path])

        descriptors = await source.discover_handlers()

        # Only the correctly named file should be discovered
        assert len(descriptors) == 1
        assert descriptors[0].handler_id == "valid.handler"


# =============================================================================
# Malformed Contract Validation Tests (OMN-1097 - TDD RED Phase)
# =============================================================================


class TestHandlerContractSourceValidation:
    """Tests for graceful malformed file handling in HandlerContractSource.

    These tests verify that HandlerContractSource gracefully handles malformed
    YAML files by producing structured errors rather than crashing. Valid
    contracts should still be discovered even when malformed contracts are
    present in the search path.

    Part of OMN-1097: HandlerContractSource + Filesystem Discovery.

    TDD RED Phase Notes:
        - These tests will fail with ImportError initially (expected)
        - Import error is acceptable for RED phase
        - Tests define the expected behavior before implementation

    Key Behavior:
        - Malformed contracts produce ModelHandlerValidationError, not exceptions
        - Valid contracts are still discovered (error isolation)
        - Structured logging includes discovered_contract_count and validation_failure_count

    Note:
        These tests assume a NEW behavior different from TestHandlerContractSourceErrors.
        The existing error tests expect exceptions to be raised for malformed contracts.
        These validation tests expect GRACEFUL handling with structured errors.

        The implementation should support BOTH modes:
        - Strict mode (default): Raise on malformed contracts
        - Graceful mode: Continue discovery, collect errors
    """

    @pytest.fixture
    def valid_handler_contract_content(self) -> str:
        """Return valid handler_contract.yaml content."""
        return """\
handler_id: "test.handler.valid"
name: "Test Valid Handler"
version: "1.0.0"
description: "A valid test handler for TDD"
descriptor:
  handler_kind: "compute"
input_model: "omnibase_infra.models.test.ModelTestInput"
output_model: "omnibase_infra.models.test.ModelTestOutput"
"""

    @pytest.fixture
    def malformed_yaml_syntax_content(self) -> str:
        """Return malformed YAML with syntax errors (unclosed quote)."""
        return """\
handler_id: "test.handler.malformed
name: missing closing quote
version: "1.0.0
"""

    @pytest.fixture
    def missing_required_fields_content(self) -> str:
        """Return YAML with missing required fields."""
        return """\
name: "Test Handler Without ID"
# Missing: handler_id, version, descriptor, input_model, output_model
"""

    @pytest.fixture
    def invalid_version_content(self) -> str:
        """Return YAML with invalid version format."""
        return """\
handler_id: "test.handler.invalid_version"
name: "Test Handler Invalid Version"
version: "not-a-semver"
description: "Handler with invalid version"
descriptor:
  handler_kind: "compute"
input_model: "omnibase_infra.models.test.ModelTestInput"
output_model: "omnibase_infra.models.test.ModelTestOutput"
"""

    @pytest.fixture
    def handler_directory_with_mixed_contracts(
        self,
        tmp_path: Path,
        valid_handler_contract_content: str,
        malformed_yaml_syntax_content: str,
        missing_required_fields_content: str,
        invalid_version_content: str,
    ) -> Path:
        """Create a temporary directory with a mix of valid and invalid contracts.

        Directory structure:
            tmp_path/
                valid_handler/
                    handler_contract.yaml  (valid)
                malformed_syntax/
                    handler_contract.yaml  (invalid YAML syntax)
                missing_fields/
                    handler_contract.yaml  (missing required fields)
                invalid_version/
                    handler_contract.yaml  (invalid version format)
                nested/
                    deep/
                        valid_nested/
                            handler_contract.yaml  (valid, nested)

        Returns:
            Path to the root temporary directory.
        """
        # Create valid handler
        valid_dir = tmp_path / "valid_handler"
        valid_dir.mkdir()
        (valid_dir / "handler_contract.yaml").write_text(valid_handler_contract_content)

        # Create malformed syntax handler
        malformed_dir = tmp_path / "malformed_syntax"
        malformed_dir.mkdir()
        (malformed_dir / "handler_contract.yaml").write_text(
            malformed_yaml_syntax_content
        )

        # Create missing fields handler
        missing_dir = tmp_path / "missing_fields"
        missing_dir.mkdir()
        (missing_dir / "handler_contract.yaml").write_text(
            missing_required_fields_content
        )

        # Create invalid version handler
        invalid_dir = tmp_path / "invalid_version"
        invalid_dir.mkdir()
        (invalid_dir / "handler_contract.yaml").write_text(invalid_version_content)

        # Create nested valid handler
        nested_dir = tmp_path / "nested" / "deep" / "valid_nested"
        nested_dir.mkdir(parents=True)
        (nested_dir / "handler_contract.yaml").write_text(
            valid_handler_contract_content
        )

        return tmp_path

    @pytest.mark.asyncio
    async def test_ignores_malformed_contracts_with_structured_error(
        self,
        handler_directory_with_mixed_contracts: Path,
    ) -> None:
        """Test that malformed contracts produce structured errors, not crashes.

        Given a directory containing both valid and malformed handler_contract.yaml
        files, discover_handlers() with graceful_mode=True should:
            1. Successfully discover and return valid contracts
            2. Produce structured ModelHandlerValidationError for each malformed contract
            3. Not raise exceptions for parse errors (graceful degradation)
            4. Include file_path in error context for debugging
            5. Include error_type appropriate to the failure mode

        Expected errors:
            - malformed_syntax: CONTRACT_PARSE_ERROR (YAML syntax error)
            - missing_fields: CONTRACT_VALIDATION_ERROR (missing required fields)
            - invalid_version: CONTRACT_VALIDATION_ERROR (invalid version format)

        This test verifies error isolation - malformed contracts should not
        prevent valid contracts from being discovered.
        """
        # Import will fail in RED phase - this is expected
        from omnibase_infra.enums import EnumHandlerErrorType
        from omnibase_infra.models.errors import ModelHandlerValidationError
        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        # Create handler contract source with search paths
        # graceful_mode=True enables structured error collection instead of raising
        source = HandlerContractSource(
            contract_paths=[handler_directory_with_mixed_contracts],
            graceful_mode=True,  # NEW parameter for graceful error handling
        )

        # Discover handlers - should not raise even with malformed contracts
        result = await source.discover_handlers()

        # Verify valid contracts were discovered (2 valid: root valid + nested valid)
        assert len(result.descriptors) == 2, (
            f"Expected 2 valid descriptors, got {len(result.descriptors)}. "
            "Malformed contracts should not prevent valid contract discovery."
        )

        # Verify validation errors were collected
        assert len(result.validation_errors) == 3, (
            f"Expected 3 validation errors, got {len(result.validation_errors)}. "
            "Each malformed contract should produce a structured error."
        )

        # Verify all errors are properly structured ModelHandlerValidationError
        for error in result.validation_errors:
            assert isinstance(error, ModelHandlerValidationError), (
                f"Expected ModelHandlerValidationError, got {type(error).__name__}"
            )
            # All errors should have file_path for debugging
            assert error.file_path is not None, (
                "Validation error must include file_path for debugging"
            )
            # All errors should have rule_id
            assert error.rule_id is not None, (
                "Validation error must include rule_id for categorization"
            )
            # All errors should have remediation_hint
            assert error.remediation_hint is not None, (
                "Validation error must include remediation_hint for fix guidance"
            )

        # Verify error types are appropriate
        error_types = {e.error_type for e in result.validation_errors}
        assert EnumHandlerErrorType.CONTRACT_PARSE_ERROR in error_types, (
            "YAML syntax errors should produce CONTRACT_PARSE_ERROR"
        )
        assert EnumHandlerErrorType.CONTRACT_VALIDATION_ERROR in error_types, (
            "Missing fields and invalid versions should produce CONTRACT_VALIDATION_ERROR"
        )

        # Verify file paths are included in errors
        error_paths = {e.file_path for e in result.validation_errors}
        assert any("malformed_syntax" in str(p) for p in error_paths), (
            "Error for malformed_syntax directory should be included"
        )
        assert any("missing_fields" in str(p) for p in error_paths), (
            "Error for missing_fields directory should be included"
        )
        assert any("invalid_version" in str(p) for p in error_paths), (
            "Error for invalid_version directory should be included"
        )

    @pytest.mark.asyncio
    async def test_logs_discovery_counts(
        self,
        handler_directory_with_mixed_contracts: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that discovery logs include structured counts.

        HandlerContractSource should emit structured log messages containing:
            - discovered_contract_count: Number of valid contracts found
            - validation_failure_count: Number of contracts that failed validation

        These counts enable monitoring and alerting on contract health.
        """
        import logging

        # Import will fail in RED phase - this is expected
        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        # Enable debug logging to capture discovery logs
        with caplog.at_level(logging.INFO, logger="omnibase_infra"):
            source = HandlerContractSource(
                contract_paths=[handler_directory_with_mixed_contracts],
                graceful_mode=True,
            )
            await source.discover_handlers()

        # Check that structured discovery logs were emitted
        log_messages = [record.message for record in caplog.records]
        log_text = " ".join(log_messages)

        # Should log discovered_contract_count
        assert (
            "discovered_contract_count" in log_text or "discovered" in log_text.lower()
        ), "Discovery should log the count of discovered contracts"

        # Should log validation_failure_count
        assert (
            "validation_failure_count" in log_text or "failure" in log_text.lower()
        ), "Discovery should log the count of validation failures"

        # Check for structured log extras (if using structured logging)
        for record in caplog.records:
            if hasattr(record, "discovered_contract_count"):
                assert record.discovered_contract_count == 2
            if hasattr(record, "validation_failure_count"):
                assert record.validation_failure_count == 3
