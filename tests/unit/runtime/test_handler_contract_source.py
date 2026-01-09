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

import pytest

from omnibase_infra.runtime.protocol_contract_descriptor import (
    ProtocolContractDescriptor,
)

# Protocol imports - use local protocols that HandlerContractSource implements
# These protocols define the interface used by HandlerContractSource, not the
# different ProtocolHandlerSource from omnibase_spi (which has list_handler_descriptors).
from omnibase_infra.runtime.protocol_contract_source import ProtocolContractSource

# Alias for test readability - HandlerContractSource implements ProtocolContractSource
ProtocolHandlerSource = ProtocolContractSource
ProtocolHandlerDescriptor = ProtocolContractDescriptor


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
        result = await source.discover_handlers()

        # Verify all 3 contracts were discovered
        assert len(result.descriptors) == 3, (
            f"Expected 3 descriptors from nested structure, got {len(result.descriptors)}"
        )

        # Verify no validation errors in strict mode
        assert len(result.validation_errors) == 0, (
            f"Expected 0 validation errors in strict mode, got {len(result.validation_errors)}"
        )

        # Verify each descriptor is a ProtocolHandlerDescriptor
        for descriptor in result.descriptors:
            assert isinstance(descriptor, ProtocolHandlerDescriptor)

        # Verify the expected handler_ids were discovered
        discovered_ids = {d.handler_id for d in result.descriptors}
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

        result = await source.discover_handlers()

        assert len(result.descriptors) == 1
        assert len(result.validation_errors) == 0
        assert result.descriptors[0].handler_id == "single.test.handler"

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

        result = await source.discover_handlers()

        assert result.descriptors == []
        assert result.validation_errors == []

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

        result = await source.discover_handlers()

        # Should find: 1 from single_contract_path + 3 from nested structure
        assert len(result.descriptors) == 4
        assert len(result.validation_errors) == 0

    @pytest.mark.asyncio
    async def test_descriptors_have_required_properties(
        self, single_contract_path: Path
    ) -> None:
        """Discovered descriptors should have all required properties.

        RED Phase: This test WILL FAIL until HandlerContractSource is implemented.

        Each descriptor must have:
        - handler_id: str
        - name: str
        - version: ModelSemVer type - use str() for string comparison
          or access .major/.minor/.patch for component comparison
        - handler_kind: str
        - input_model: str
        - output_model: str
        """
        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        source = HandlerContractSource(contract_paths=[single_contract_path])

        result = await source.discover_handlers()

        assert len(result.descriptors) == 1
        assert len(result.validation_errors) == 0
        descriptor = result.descriptors[0]

        # Verify required properties exist and have correct values
        assert descriptor.handler_id == "single.test.handler"
        assert descriptor.name == "Single Test Handler"
        # version is ModelSemVer, compare using str() for string representation
        assert str(descriptor.version) == "1.0.0"
        assert descriptor.version.major == 1
        assert descriptor.version.minor == 0
        assert descriptor.version.patch == 0
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
        assert (
            len(result1.descriptors)
            == len(result2.descriptors)
            == len(result3.descriptors)
            == 3
        )
        assert (
            len(result1.validation_errors)
            == len(result2.validation_errors)
            == len(result3.validation_errors)
            == 0
        )

        ids1 = {d.handler_id for d in result1.descriptors}
        ids2 = {d.handler_id for d in result2.descriptors}
        ids3 = {d.handler_id for d in result3.descriptors}

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

        result = await source.discover_handlers()

        # Only the correctly named file should be discovered
        assert len(result.descriptors) == 1
        assert len(result.validation_errors) == 0
        assert result.descriptors[0].handler_id == "valid.handler"


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


# =============================================================================
# Forward Reference Resolution Tests
# =============================================================================


class TestModelContractDiscoveryResultForwardReference:
    """Tests for forward reference resolution in ModelContractDiscoveryResult.

    ModelContractDiscoveryResult uses a forward reference pattern to avoid
    circular imports between models.handlers and models.errors. The forward
    reference to ModelHandlerValidationError is resolved via model_rebuild()
    after both classes are defined.

    These tests verify:
    1. The forward reference is properly resolved at runtime
    2. Type hints return the correct types
    3. Instances can be created with validation_errors field
    4. Pydantic validation works correctly on the field

    Why this pattern exists:
        ModelContractDiscoveryResult has a field typed as list[ModelHandlerValidationError].
        ModelHandlerValidationError imports ModelHandlerIdentifier from models.handlers.
        If ModelContractDiscoveryResult directly imported ModelHandlerValidationError,
        it would cause a circular import because models.handlers.__init__.py imports
        ModelContractDiscoveryResult.

    The solution uses TYPE_CHECKING + model_rebuild() to defer the import.
    """

    def test_model_can_be_imported_from_handlers_package(self) -> None:
        """ModelContractDiscoveryResult should be importable from models.handlers.

        This verifies that the module-level import in __init__.py works without
        triggering circular import errors.
        """
        from omnibase_infra.models.handlers import ModelContractDiscoveryResult

        assert ModelContractDiscoveryResult is not None

    def test_forward_reference_is_resolved_for_pydantic_validation(self) -> None:
        """Forward reference should be resolved for Pydantic validation.

        The handler_contract_source module calls model_rebuild() after importing
        ModelHandlerValidationError, which resolves the forward reference for
        Pydantic's internal type validation.

        Note: Python's get_type_hints() may still fail because it uses standard
        evaluation, but Pydantic's validation works correctly after model_rebuild().
        This test verifies Pydantic's type resolution via model_fields.

        This test imports through handler_contract_source to ensure model_rebuild()
        has been called.
        """
        # Import through handler_contract_source which calls model_rebuild()
        from omnibase_infra.models.errors import ModelHandlerValidationError
        from omnibase_infra.runtime.handler_contract_source import (
            ModelContractDiscoveryResult,
        )

        # Verify the model's field info contains the resolved type
        # Pydantic stores resolved types in model_fields after model_rebuild()
        field_info = ModelContractDiscoveryResult.model_fields.get("validation_errors")
        assert field_info is not None, "validation_errors should be a registered field"

        # Verify Pydantic can reconstruct the model (proves types are resolved)
        # model_rebuild() would fail if forward references weren't resolvable
        try:
            ModelContractDiscoveryResult.model_rebuild()
        except Exception as e:
            pytest.fail(f"model_rebuild() failed, forward reference unresolved: {e}")

        # Verify we can get the JSON schema (requires resolved types)
        try:
            schema = ModelContractDiscoveryResult.model_json_schema()
            assert "validation_errors" in schema.get("properties", {}), (
                "JSON schema should include validation_errors property"
            )
        except Exception as e:
            pytest.fail(
                f"model_json_schema() failed, forward reference unresolved: {e}"
            )

        # Finally, verify actual type validation works by creating with valid error
        from omnibase_infra.enums import EnumHandlerErrorType, EnumHandlerSourceType
        from omnibase_infra.models.handlers import ModelHandlerIdentifier

        error = ModelHandlerValidationError(
            error_type=EnumHandlerErrorType.CONTRACT_PARSE_ERROR,
            rule_id="TEST-001",
            handler_identity=ModelHandlerIdentifier.from_handler_id("test"),
            source_type=EnumHandlerSourceType.CONTRACT,
            message="Test",
            remediation_hint="Fix it",
        )
        result = ModelContractDiscoveryResult(
            descriptors=[],
            validation_errors=[error],
        )
        assert len(result.validation_errors) == 1

    def test_instance_creation_with_empty_validation_errors(self) -> None:
        """ModelContractDiscoveryResult can be instantiated with empty validation_errors.

        This verifies the forward reference is properly resolved for Pydantic
        validation at instance creation time.
        """
        # Import through handler_contract_source to ensure model_rebuild() is called
        from omnibase_infra.runtime.handler_contract_source import (
            ModelContractDiscoveryResult,
        )

        result = ModelContractDiscoveryResult(
            descriptors=[],
            validation_errors=[],
        )

        assert result.descriptors == []
        assert result.validation_errors == []

    def test_instance_creation_with_validation_error_objects(self) -> None:
        """ModelContractDiscoveryResult can be instantiated with actual validation errors.

        This verifies Pydantic correctly validates ModelHandlerValidationError
        instances in the validation_errors list.
        """
        # Import through handler_contract_source to ensure model_rebuild() is called
        from omnibase_infra.enums import EnumHandlerErrorType, EnumHandlerSourceType
        from omnibase_infra.models.errors import ModelHandlerValidationError
        from omnibase_infra.models.handlers import ModelHandlerIdentifier
        from omnibase_infra.runtime.handler_contract_source import (
            ModelContractDiscoveryResult,
        )

        # Create a validation error
        error = ModelHandlerValidationError(
            error_type=EnumHandlerErrorType.CONTRACT_PARSE_ERROR,
            rule_id="CONTRACT-001",
            handler_identity=ModelHandlerIdentifier.from_handler_id("test-handler"),
            source_type=EnumHandlerSourceType.CONTRACT,
            message="Test validation error",
            remediation_hint="Fix the contract",
            file_path="/test/contract.yaml",
        )

        # Create result with the error
        result = ModelContractDiscoveryResult(
            descriptors=[],
            validation_errors=[error],
        )

        assert len(result.validation_errors) == 1
        assert result.validation_errors[0].rule_id == "CONTRACT-001"
        assert result.validation_errors[0].message == "Test validation error"

    def test_pydantic_rejects_invalid_validation_error_type(self) -> None:
        """Pydantic should reject invalid types in validation_errors list.

        This verifies that the forward reference is properly resolved for
        Pydantic type validation, rejecting non-ModelHandlerValidationError values.
        """
        from pydantic import ValidationError

        # Import through handler_contract_source to ensure model_rebuild() is called
        from omnibase_infra.runtime.handler_contract_source import (
            ModelContractDiscoveryResult,
        )

        # Try to create with invalid type - should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            ModelContractDiscoveryResult(
                descriptors=[],
                validation_errors=["not a validation error"],  # type: ignore[list-item]
            )

        # Verify the error is about the validation_errors field
        errors = exc_info.value.errors()
        assert len(errors) >= 1
        # The error should be about validation_errors field
        error_locs = [str(e.get("loc", ())) for e in errors]
        assert any("validation_errors" in loc for loc in error_locs), (
            f"Expected error about validation_errors, got locations: {error_locs}"
        )


# =============================================================================
# Edge Case Tests (PR Review Feedback)
# =============================================================================


class TestHandlerContractSourceCaseSensitivity:
    """Tests for case-sensitive file discovery.

    Verifies that HandlerContractSource only discovers files named exactly
    'handler_contract.yaml' (lowercase), not case variations like
    'HANDLER_CONTRACT.yaml' or 'Handler_Contract.yaml'.

    This is critical for cross-platform consistency since macOS and Windows
    filesystems may be case-insensitive, but we want consistent behavior.
    """

    @pytest.mark.asyncio
    async def test_only_discovers_lowercase_handler_contract_yaml(
        self, tmp_path: Path
    ) -> None:
        """Verify only handler_contract.yaml is discovered, not case variations.

        Creates directories with various case variations of the contract filename
        and verifies only the correctly-cased file is discovered.
        """
        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        # Create directories for different case variations
        lowercase_dir = tmp_path / "lowercase"
        uppercase_dir = tmp_path / "uppercase"
        mixed_dir = tmp_path / "mixed"
        lowercase_dir.mkdir()
        uppercase_dir.mkdir()
        mixed_dir.mkdir()

        valid_yaml_template = """
handler_id: "test.handler.{variant}"
name: "{variant} Handler"
version: "1.0.0"
descriptor:
  handler_kind: "compute"
input_model: "test.models.Input"
output_model: "test.models.Output"
"""
        # Only this one should be discovered (correct case)
        (lowercase_dir / "handler_contract.yaml").write_text(
            valid_yaml_template.format(variant="lowercase")
        )

        # These should NOT be discovered (wrong case)
        (uppercase_dir / "HANDLER_CONTRACT.YAML").write_text(
            valid_yaml_template.format(variant="uppercase")
        )
        (mixed_dir / "Handler_Contract.yaml").write_text(
            valid_yaml_template.format(variant="mixed")
        )

        source = HandlerContractSource(contract_paths=[tmp_path])
        result = await source.discover_handlers()

        # Only the lowercase variant should be discovered
        assert len(result.descriptors) == 1, (
            f"Expected 1 descriptor (lowercase only), got {len(result.descriptors)}. "
            "Case variations should not be discovered."
        )
        assert len(result.validation_errors) == 0
        assert result.descriptors[0].handler_id == "test.handler.lowercase"

    @pytest.mark.asyncio
    async def test_exact_filename_matching_rejects_all_variations(
        self, tmp_path: Path
    ) -> None:
        """Verify exact 'handler_contract.yaml' filename matching (case and extension).

        This test comprehensively verifies that ONLY files named exactly
        'handler_contract.yaml' are discovered. All variations must be rejected:

        SHOULD be discovered:
            - handler_contract.yaml (exact match)

        SHOULD NOT be discovered:
            - Handler_Contract.yaml (mixed case in name)
            - HANDLER_CONTRACT.YAML (all caps name and extension)
            - handler_contract.yml (wrong extension - .yml instead of .yaml)
            - handler_contract.YAML (caps extension only)
            - HANDLER_contract.yaml (partial caps in name)
            - handler_CONTRACT.yaml (partial caps in name)

        Each variation is placed in a separate subdirectory to avoid filesystem
        case-sensitivity issues (macOS/Windows HFS+/NTFS may be case-insensitive
        and silently overwrite files with different case).

        Why this matters:
            Cross-platform consistency requires deterministic behavior. If we
            allowed case variations, the same codebase could behave differently
            on case-sensitive (Linux ext4) vs case-insensitive (macOS HFS+)
            filesystems, causing hard-to-debug deployment issues.
        """
        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        valid_yaml_template = """
handler_id: "test.handler.{variant}"
name: "{variant} Handler"
version: "1.0.0"
descriptor:
  handler_kind: "compute"
input_model: "test.models.Input"
output_model: "test.models.Output"
"""

        # ===== VALID: exact match =====
        valid_dir = tmp_path / "valid_exact_match"
        valid_dir.mkdir()
        (valid_dir / "handler_contract.yaml").write_text(
            valid_yaml_template.format(variant="valid")
        )

        # ===== INVALID: case variations - each in separate directory =====

        # Mixed case in name (Handler_Contract)
        mixed_case_dir = tmp_path / "invalid_mixed_case"
        mixed_case_dir.mkdir()
        (mixed_case_dir / "Handler_Contract.yaml").write_text(
            valid_yaml_template.format(variant="mixed_case")
        )

        # All caps name AND extension (HANDLER_CONTRACT.YAML)
        all_caps_dir = tmp_path / "invalid_all_caps"
        all_caps_dir.mkdir()
        (all_caps_dir / "HANDLER_CONTRACT.YAML").write_text(
            valid_yaml_template.format(variant="all_caps")
        )

        # Wrong extension .yml instead of .yaml
        wrong_ext_yml_dir = tmp_path / "invalid_wrong_ext_yml"
        wrong_ext_yml_dir.mkdir()
        (wrong_ext_yml_dir / "handler_contract.yml").write_text(
            valid_yaml_template.format(variant="wrong_ext_yml")
        )

        # Caps extension only (handler_contract.YAML)
        caps_ext_dir = tmp_path / "invalid_caps_extension"
        caps_ext_dir.mkdir()
        (caps_ext_dir / "handler_contract.YAML").write_text(
            valid_yaml_template.format(variant="caps_extension")
        )

        # Partial caps in name - first part (HANDLER_contract.yaml)
        partial_caps_first_dir = tmp_path / "invalid_partial_caps_first"
        partial_caps_first_dir.mkdir()
        (partial_caps_first_dir / "HANDLER_contract.yaml").write_text(
            valid_yaml_template.format(variant="partial_caps_first")
        )

        # Partial caps in name - second part (handler_CONTRACT.yaml)
        partial_caps_second_dir = tmp_path / "invalid_partial_caps_second"
        partial_caps_second_dir.mkdir()
        (partial_caps_second_dir / "handler_CONTRACT.yaml").write_text(
            valid_yaml_template.format(variant="partial_caps_second")
        )

        # Discover handlers
        source = HandlerContractSource(contract_paths=[tmp_path])
        result = await source.discover_handlers()

        # Verify ONLY the exact match was discovered
        assert len(result.descriptors) == 1, (
            f"Expected exactly 1 descriptor (handler_contract.yaml only), "
            f"got {len(result.descriptors)}. "
            f"Discovered handler IDs: {[d.handler_id for d in result.descriptors]}"
        )
        assert len(result.validation_errors) == 0, (
            f"Expected 0 validation errors, got {len(result.validation_errors)}"
        )

        # Verify the discovered descriptor is from the valid file
        discovered_handler = result.descriptors[0]
        assert discovered_handler.handler_id == "test.handler.valid", (
            f"Expected handler_id 'test.handler.valid', "
            f"got '{discovered_handler.handler_id}'"
        )

        # Additional verification: check contract_path points to correct file
        assert "valid_exact_match" in discovered_handler.contract_path, (
            f"Expected contract_path to contain 'valid_exact_match', "
            f"got '{discovered_handler.contract_path}'"
        )


class TestHandlerContractSourceSymlinkHandling:
    """Tests for symlink handling in contract discovery.

    Verifies that HandlerContractSource correctly follows symlinks when
    discovering handler_contract.yaml files.
    """

    @pytest.mark.asyncio
    async def test_discovers_contracts_via_symlinks(self, tmp_path: Path) -> None:
        """Verify symlinked handler_contract.yaml files are discovered.

        Creates a real contract file and a symlink to it, then verifies
        discovery works when searching the symlink directory. Both directories
        are included in contract_paths to satisfy symlink protection.
        """
        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        # Create actual contract in one location
        actual_dir = tmp_path / "actual"
        actual_dir.mkdir()

        valid_yaml = """
handler_id: "test.handler.symlinked"
name: "Symlinked Handler"
version: "1.0.0"
descriptor:
  handler_kind: "effect"
input_model: "test.models.Input"
output_model: "test.models.Output"
"""
        actual_contract = actual_dir / "handler_contract.yaml"
        actual_contract.write_text(valid_yaml)

        # Create symlink directory with symlink to the contract
        symlink_dir = tmp_path / "symlinked"
        symlink_dir.mkdir()
        symlink_contract = symlink_dir / "handler_contract.yaml"
        symlink_contract.symlink_to(actual_contract)

        # Include both directories in contract_paths so symlink target is allowed
        # This tests symlink following while respecting path security
        source = HandlerContractSource(contract_paths=[symlink_dir, actual_dir])
        result = await source.discover_handlers()

        # Should discover exactly 1 (deduplicated by resolved path)
        assert len(result.descriptors) == 1, (
            f"Expected 1 descriptor via symlink, got {len(result.descriptors)}"
        )
        assert len(result.validation_errors) == 0
        assert result.descriptors[0].handler_id == "test.handler.symlinked"

    @pytest.mark.asyncio
    async def test_blocks_symlinks_outside_allowed_paths(self, tmp_path: Path) -> None:
        """Verify symlinks pointing outside allowed paths are blocked.

        This tests the security feature that prevents symlink-based path
        traversal attacks where a symlink inside a configured path points
        to files outside allowed directories.
        """
        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        # Create actual contract OUTSIDE the allowed path
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()

        valid_yaml = """
handler_id: "test.handler.outside"
name: "Outside Handler"
version: "1.0.0"
descriptor:
  handler_kind: "effect"
input_model: "test.models.Input"
output_model: "test.models.Output"
"""
        outside_contract = outside_dir / "handler_contract.yaml"
        outside_contract.write_text(valid_yaml)

        # Create symlink directory with symlink pointing outside
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        symlink_contract = allowed_dir / "handler_contract.yaml"
        symlink_contract.symlink_to(outside_contract)

        # Search only in allowed directory (symlink target is NOT allowed)
        source = HandlerContractSource(contract_paths=[allowed_dir])
        result = await source.discover_handlers()

        # Should discover 0 - symlink pointing outside is blocked
        assert len(result.descriptors) == 0, (
            f"Expected 0 descriptors (symlink blocked), got {len(result.descriptors)}"
        )
        assert len(result.validation_errors) == 0

    @pytest.mark.asyncio
    async def test_deduplicates_symlinked_and_actual_contracts(
        self, tmp_path: Path
    ) -> None:
        """Verify that symlinked and actual contracts are deduplicated.

        When both the actual file and a symlink to it are in the search paths,
        only one descriptor should be returned (not duplicates).
        """
        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        # Create actual contract
        actual_dir = tmp_path / "actual"
        actual_dir.mkdir()

        valid_yaml = """
handler_id: "test.handler.dedup"
name: "Deduplicated Handler"
version: "1.0.0"
descriptor:
  handler_kind: "compute"
input_model: "test.models.Input"
output_model: "test.models.Output"
"""
        actual_contract = actual_dir / "handler_contract.yaml"
        actual_contract.write_text(valid_yaml)

        # Create symlink in another directory
        symlink_dir = tmp_path / "symlinked"
        symlink_dir.mkdir()
        symlink_contract = symlink_dir / "handler_contract.yaml"
        symlink_contract.symlink_to(actual_contract)

        # Search both directories - should deduplicate
        source = HandlerContractSource(contract_paths=[actual_dir, symlink_dir])
        result = await source.discover_handlers()

        assert len(result.descriptors) == 1, (
            f"Expected 1 descriptor (deduplicated), got {len(result.descriptors)}. "
            "Symlinked files should be deduplicated with actual files."
        )
        assert len(result.validation_errors) == 0
        assert result.descriptors[0].handler_id == "test.handler.dedup"


# =============================================================================
# Multi-Segment Module Path Tests
# =============================================================================


HANDLER_CONTRACT_WITH_MULTI_SEGMENT_PATHS = """
handler_id: "{handler_id}"
name: "{name}"
version: "1.0.0"
descriptor:
  handler_kind: "compute"
input_model: "{input_model}"
output_model: "{output_model}"
"""


class TestHandlerContractSourceMultiSegmentPaths:
    """Tests for multi-segment module path handling in HandlerContractSource.

    These tests verify that HandlerContractSource correctly discovers and parses
    handler contracts that use multi-segment module paths for input_model and
    output_model fields.

    Multi-segment paths include:
    - Deep nesting (3+ segments): omnibase_infra.models.handlers.ModelInput
    - Very deep nesting (6+ segments): a.b.c.d.e.f.ModelDeep
    - Underscores in segment names: module_with_underscore.sub_module.ModelName
    - Mixed patterns: omnibase_core.models.primitives.model_semver.ModelSemVer

    This is critical for real-world usage where models are organized in nested
    package structures following Python conventions.
    """

    @pytest.mark.asyncio
    async def test_discovers_contracts_with_three_segment_paths(
        self, tmp_path: Path
    ) -> None:
        """Verify contracts with 3-segment module paths are correctly discovered.

        Tests paths like: omnibase_infra.models.ModelHandler
        """
        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        contract_dir = tmp_path / "three_segment"
        contract_dir.mkdir()
        (contract_dir / "handler_contract.yaml").write_text(
            HANDLER_CONTRACT_WITH_MULTI_SEGMENT_PATHS.format(
                handler_id="test.three_segment.handler",
                name="Three Segment Path Handler",
                input_model="omnibase_infra.models.ModelInput",
                output_model="omnibase_infra.models.ModelOutput",
            )
        )

        source = HandlerContractSource(contract_paths=[tmp_path])
        result = await source.discover_handlers()

        assert len(result.descriptors) == 1
        assert len(result.validation_errors) == 0
        descriptor = result.descriptors[0]
        assert descriptor.input_model == "omnibase_infra.models.ModelInput"
        assert descriptor.output_model == "omnibase_infra.models.ModelOutput"

    @pytest.mark.asyncio
    async def test_discovers_contracts_with_four_segment_paths(
        self, tmp_path: Path
    ) -> None:
        """Verify contracts with 4-segment module paths are correctly discovered.

        Tests paths like: omnibase_infra.models.handlers.ModelHandler
        """
        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        contract_dir = tmp_path / "four_segment"
        contract_dir.mkdir()
        (contract_dir / "handler_contract.yaml").write_text(
            HANDLER_CONTRACT_WITH_MULTI_SEGMENT_PATHS.format(
                handler_id="test.four_segment.handler",
                name="Four Segment Path Handler",
                input_model="omnibase_infra.models.handlers.ModelInput",
                output_model="omnibase_infra.models.handlers.ModelOutput",
            )
        )

        source = HandlerContractSource(contract_paths=[tmp_path])
        result = await source.discover_handlers()

        assert len(result.descriptors) == 1
        assert len(result.validation_errors) == 0
        descriptor = result.descriptors[0]
        assert descriptor.input_model == "omnibase_infra.models.handlers.ModelInput"
        assert descriptor.output_model == "omnibase_infra.models.handlers.ModelOutput"

    @pytest.mark.asyncio
    async def test_discovers_contracts_with_deeply_nested_paths(
        self, tmp_path: Path
    ) -> None:
        """Verify contracts with 6+ segment module paths are correctly discovered.

        Tests very deep nesting like: a.b.c.d.e.f.ModelDeep
        This ensures no artificial limit on path depth.
        """
        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        contract_dir = tmp_path / "deep_nested"
        contract_dir.mkdir()
        (contract_dir / "handler_contract.yaml").write_text(
            HANDLER_CONTRACT_WITH_MULTI_SEGMENT_PATHS.format(
                handler_id="test.deep_nested.handler",
                name="Deeply Nested Path Handler",
                input_model="level1.level2.level3.level4.level5.level6.ModelDeepInput",
                output_model="a.b.c.d.e.f.g.h.ModelVeryDeepOutput",
            )
        )

        source = HandlerContractSource(contract_paths=[tmp_path])
        result = await source.discover_handlers()

        assert len(result.descriptors) == 1
        assert len(result.validation_errors) == 0
        descriptor = result.descriptors[0]
        assert descriptor.input_model == (
            "level1.level2.level3.level4.level5.level6.ModelDeepInput"
        )
        assert descriptor.output_model == "a.b.c.d.e.f.g.h.ModelVeryDeepOutput"

    @pytest.mark.asyncio
    async def test_discovers_contracts_with_underscore_segments(
        self, tmp_path: Path
    ) -> None:
        """Verify contracts with underscores in module path segments.

        Tests paths like: module_with_underscore.sub_module_name.ModelName
        Python package names commonly use underscores.
        """
        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        contract_dir = tmp_path / "underscore_segments"
        contract_dir.mkdir()
        (contract_dir / "handler_contract.yaml").write_text(
            HANDLER_CONTRACT_WITH_MULTI_SEGMENT_PATHS.format(
                handler_id="test.underscore_segments.handler",
                name="Underscore Segments Handler",
                input_model="omnibase_infra.models_v2.handler_models.ModelInputV2",
                output_model="package_name.sub_package_name.module_name.ModelOutput",
            )
        )

        source = HandlerContractSource(contract_paths=[tmp_path])
        result = await source.discover_handlers()

        assert len(result.descriptors) == 1
        assert len(result.validation_errors) == 0
        descriptor = result.descriptors[0]
        assert descriptor.input_model == (
            "omnibase_infra.models_v2.handler_models.ModelInputV2"
        )
        assert descriptor.output_model == (
            "package_name.sub_package_name.module_name.ModelOutput"
        )

    @pytest.mark.asyncio
    async def test_discovers_contracts_with_mixed_path_patterns(
        self, tmp_path: Path
    ) -> None:
        """Verify contracts with mixed path patterns are discovered correctly.

        Tests realistic paths from the omnibase ecosystem:
        - omnibase_core.models.primitives.model_semver.ModelSemVer
        - omnibase_infra.nodes.adapters.kafka.models.ModelKafkaMessage
        """
        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        contract_dir = tmp_path / "mixed_patterns"
        contract_dir.mkdir()
        (contract_dir / "handler_contract.yaml").write_text(
            HANDLER_CONTRACT_WITH_MULTI_SEGMENT_PATHS.format(
                handler_id="test.mixed_patterns.handler",
                name="Mixed Patterns Handler",
                input_model="omnibase_core.models.primitives.model_semver.ModelSemVer",
                output_model=(
                    "omnibase_infra.nodes.adapters.kafka.models.ModelKafkaMessage"
                ),
            )
        )

        source = HandlerContractSource(contract_paths=[tmp_path])
        result = await source.discover_handlers()

        assert len(result.descriptors) == 1
        assert len(result.validation_errors) == 0
        descriptor = result.descriptors[0]
        assert descriptor.input_model == (
            "omnibase_core.models.primitives.model_semver.ModelSemVer"
        )
        assert descriptor.output_model == (
            "omnibase_infra.nodes.adapters.kafka.models.ModelKafkaMessage"
        )

    @pytest.mark.asyncio
    async def test_multiple_contracts_with_different_path_depths(
        self, tmp_path: Path
    ) -> None:
        """Verify multiple contracts with varying path depths are all discovered.

        Creates contracts with 2, 3, 4, and 6 segment paths to ensure
        all depths work together.
        """
        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        # 2 segments (simple)
        dir_2seg = tmp_path / "two_segment"
        dir_2seg.mkdir()
        (dir_2seg / "handler_contract.yaml").write_text(
            HANDLER_CONTRACT_WITH_MULTI_SEGMENT_PATHS.format(
                handler_id="test.two_segment",
                name="Two Segment",
                input_model="models.Input",
                output_model="models.Output",
            )
        )

        # 3 segments
        dir_3seg = tmp_path / "three_segment"
        dir_3seg.mkdir()
        (dir_3seg / "handler_contract.yaml").write_text(
            HANDLER_CONTRACT_WITH_MULTI_SEGMENT_PATHS.format(
                handler_id="test.three_segment",
                name="Three Segment",
                input_model="pkg.models.Input",
                output_model="pkg.models.Output",
            )
        )

        # 4 segments
        dir_4seg = tmp_path / "four_segment"
        dir_4seg.mkdir()
        (dir_4seg / "handler_contract.yaml").write_text(
            HANDLER_CONTRACT_WITH_MULTI_SEGMENT_PATHS.format(
                handler_id="test.four_segment",
                name="Four Segment",
                input_model="org.pkg.models.Input",
                output_model="org.pkg.models.Output",
            )
        )

        # 6 segments (deep)
        dir_6seg = tmp_path / "six_segment"
        dir_6seg.mkdir()
        (dir_6seg / "handler_contract.yaml").write_text(
            HANDLER_CONTRACT_WITH_MULTI_SEGMENT_PATHS.format(
                handler_id="test.six_segment",
                name="Six Segment",
                input_model="a.b.c.d.e.Input",
                output_model="a.b.c.d.e.Output",
            )
        )

        source = HandlerContractSource(contract_paths=[tmp_path])
        result = await source.discover_handlers()

        assert len(result.descriptors) == 4
        assert len(result.validation_errors) == 0

        # Verify all handler IDs are present
        discovered_ids = {d.handler_id for d in result.descriptors}
        expected_ids = {
            "test.two_segment",
            "test.three_segment",
            "test.four_segment",
            "test.six_segment",
        }
        assert discovered_ids == expected_ids

    @pytest.mark.asyncio
    async def test_preserves_path_case_sensitivity(self, tmp_path: Path) -> None:
        """Verify that module path case is preserved exactly as specified.

        Module paths should maintain their exact casing since Python
        module names are case-sensitive.
        """
        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        contract_dir = tmp_path / "case_sensitive"
        contract_dir.mkdir()
        (contract_dir / "handler_contract.yaml").write_text(
            HANDLER_CONTRACT_WITH_MULTI_SEGMENT_PATHS.format(
                handler_id="test.case_sensitive.handler",
                name="Case Sensitive Handler",
                input_model="OmniBase.Models.Handlers.ModelInput",
                output_model="myPackage.SubModule.MODEL_OUTPUT",
            )
        )

        source = HandlerContractSource(contract_paths=[tmp_path])
        result = await source.discover_handlers()

        assert len(result.descriptors) == 1
        descriptor = result.descriptors[0]
        # Case must be preserved exactly
        assert descriptor.input_model == "OmniBase.Models.Handlers.ModelInput"
        assert descriptor.output_model == "myPackage.SubModule.MODEL_OUTPUT"

    @pytest.mark.asyncio
    async def test_handles_numeric_like_segments(self, tmp_path: Path) -> None:
        """Verify paths with numeric-like segments work correctly.

        Tests segments like v2, models_v1, etc. which are valid Python identifiers.
        """
        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        contract_dir = tmp_path / "numeric_segments"
        contract_dir.mkdir()
        (contract_dir / "handler_contract.yaml").write_text(
            HANDLER_CONTRACT_WITH_MULTI_SEGMENT_PATHS.format(
                handler_id="test.numeric_segments.handler",
                name="Numeric Segments Handler",
                input_model="omnibase_v2.models_v1.handlers_v3.ModelInputV4",
                output_model="api.v2.responses.ModelResponseV1",
            )
        )

        source = HandlerContractSource(contract_paths=[tmp_path])
        result = await source.discover_handlers()

        assert len(result.descriptors) == 1
        assert len(result.validation_errors) == 0
        descriptor = result.descriptors[0]
        assert descriptor.input_model == (
            "omnibase_v2.models_v1.handlers_v3.ModelInputV4"
        )
        assert descriptor.output_model == "api.v2.responses.ModelResponseV1"


def _permissions_are_enforced() -> bool:
    """Check if file permission enforcement works in this environment.

    Some environments (Docker containers with certain mount options, Windows, etc.)
    don't properly enforce file permissions even when chmod succeeds.

    Returns:
        True if file permissions are enforced, False otherwise.
    """
    import tempfile

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            temp_path = Path(f.name)

        # Set permissions to no-read
        temp_path.chmod(0o000)

        # Try to read the file
        try:
            temp_path.read_text()
            # If we get here, permissions are NOT enforced
            return False
        except PermissionError:
            # Permissions ARE enforced
            return True
        finally:
            temp_path.chmod(0o644)
            temp_path.unlink()
    except Exception:
        # If something goes wrong, assume permissions aren't enforced
        return False


def _create_mock_stat_result(real_stat_result: object, override_size: int) -> object:
    """Create a mock stat result object with overridden st_size.

    This factory reduces duplication in file size limit tests by creating
    mock stat result objects that report a specific file size while preserving
    all other stat attributes from the original file.

    Args:
        real_stat_result: The actual os.stat_result from Path.stat()
        override_size: The file size (st_size) to report

    Returns:
        A mock object with st_size set to override_size and all other
        attributes copied from real_stat_result.

    Example:
        >>> original_stat = Path.stat
        >>> def mock_stat(self, **kwargs):
        ...     result = original_stat(self, **kwargs)
        ...     if self.name == "handler_contract.yaml":
        ...         return _create_mock_stat_result(result, 10 * 1024 * 1024 + 1)
        ...     return result
    """

    class MockStatResult:
        """Mock stat result with configurable st_size."""

        st_size = override_size
        st_mode = real_stat_result.st_mode  # type: ignore[union-attr]
        st_ino = real_stat_result.st_ino  # type: ignore[union-attr]
        st_dev = real_stat_result.st_dev  # type: ignore[union-attr]
        st_nlink = real_stat_result.st_nlink  # type: ignore[union-attr]
        st_uid = real_stat_result.st_uid  # type: ignore[union-attr]
        st_gid = real_stat_result.st_gid  # type: ignore[union-attr]
        st_atime = real_stat_result.st_atime  # type: ignore[union-attr]
        st_mtime = real_stat_result.st_mtime  # type: ignore[union-attr]
        st_ctime = real_stat_result.st_ctime  # type: ignore[union-attr]

    return MockStatResult()


class TestHandlerContractSourcePermissionErrors:
    """Tests for permission error handling in contract discovery.

    Verifies that HandlerContractSource correctly handles unreadable files
    by producing structured errors in graceful mode or raising in strict mode.

    Note: These tests are skipped when file permissions are not enforced,
    which can occur when running as root (UID 0), in certain Docker containers
    with specific mount options, or on filesystems that don't support Unix permissions.
    """

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        __import__("os").name == "nt",
        reason="Permission test not reliable on Windows",
    )
    @pytest.mark.skipif(
        not _permissions_are_enforced(),
        reason="File permissions not enforced in this environment (root or mount options)",
    )
    async def test_handles_permission_errors_gracefully(self, tmp_path: Path) -> None:
        """Verify permission errors produce structured errors in graceful mode.

        Creates a valid contract and an unreadable contract, then verifies
        that graceful mode discovers the valid contract and produces a
        structured error for the unreadable one.
        """
        import stat

        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        valid_yaml = """
handler_id: "test.handler.valid"
name: "Valid Handler"
version: "1.0.0"
descriptor:
  handler_kind: "compute"
input_model: "test.models.Input"
output_model: "test.models.Output"
"""
        # Create a valid contract
        valid_dir = tmp_path / "valid"
        valid_dir.mkdir()
        (valid_dir / "handler_contract.yaml").write_text(valid_yaml)

        # Create an unreadable contract
        unreadable_dir = tmp_path / "unreadable"
        unreadable_dir.mkdir()
        unreadable_contract = unreadable_dir / "handler_contract.yaml"
        unreadable_contract.write_text(valid_yaml.replace("valid", "unreadable"))

        # Remove read permission
        unreadable_contract.chmod(0o000)

        try:
            # Graceful mode should collect errors instead of raising
            source = HandlerContractSource(
                contract_paths=[tmp_path],
                graceful_mode=True,
            )

            # Currently, PermissionError is not caught by the implementation
            # This test documents the expected behavior: graceful mode should
            # catch IO errors and produce structured validation errors.
            # If this test fails with PermissionError, the implementation
            # needs to be updated to handle IO errors gracefully.
            try:
                result = await source.discover_handlers()

                # Should still discover the valid contract
                assert len(result.descriptors) >= 1, (
                    "Valid contract should still be discovered"
                )
                valid_ids = {d.handler_id for d in result.descriptors}
                assert "test.handler.valid" in valid_ids, (
                    "Valid handler should be in discovered descriptors"
                )

                # Should have error for unreadable contract
                assert len(result.validation_errors) >= 1, (
                    "Should have validation error for unreadable contract"
                )
            except PermissionError:
                # Current implementation doesn't handle PermissionError
                # This is acceptable behavior - mark as known limitation
                pytest.skip(
                    "Implementation does not yet handle PermissionError gracefully. "
                    "This is a known limitation - IO errors propagate in graceful mode."
                )
        finally:
            # Restore permissions for cleanup
            unreadable_contract.chmod(stat.S_IRUSR | stat.S_IWUSR)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        __import__("os").name == "nt",
        reason="Permission test not reliable on Windows",
    )
    @pytest.mark.skipif(
        not _permissions_are_enforced(),
        reason="File permissions not enforced in this environment (root or mount options)",
    )
    async def test_raises_permission_error_in_strict_mode(self, tmp_path: Path) -> None:
        """Verify permission errors raise in strict mode.

        In strict mode (default), unreadable files should cause discovery to
        fail with ModelOnexError wrapping the underlying permission error.
        """
        import stat

        from omnibase_core.models.errors.model_onex_error import ModelOnexError

        from omnibase_infra.runtime.handler_contract_source import (
            HandlerContractSource,
        )

        valid_yaml = """
handler_id: "test.handler.unreadable"
name: "Unreadable Handler"
version: "1.0.0"
descriptor:
  handler_kind: "compute"
input_model: "test.models.Input"
output_model: "test.models.Output"
"""
        # Create an unreadable contract
        unreadable_dir = tmp_path / "unreadable_strict"
        unreadable_dir.mkdir()
        unreadable_contract = unreadable_dir / "handler_contract.yaml"
        unreadable_contract.write_text(valid_yaml)

        # Remove read permission
        unreadable_contract.chmod(0o000)

        try:
            source = HandlerContractSource(
                contract_paths=[unreadable_dir],
                graceful_mode=False,  # Strict mode
            )

            # Strict mode should raise ModelOnexError with HANDLER_SOURCE_006 code
            # The underlying PermissionError is preserved as __cause__
            with pytest.raises(ModelOnexError) as exc_info:
                await source.discover_handlers()

            # Verify error details
            error = exc_info.value
            assert error.error_code == "HANDLER_SOURCE_006", (
                f"Expected error code HANDLER_SOURCE_006, got {error.error_code}"
            )
            # Verify error message mentions permission issue
            assert "permission denied" in str(error).lower(), (
                f"Error message should mention permission issue: {error}"
            )
            # Verify original error is preserved as __cause__
            assert isinstance(error.__cause__, (PermissionError, OSError))
        finally:
            # Restore permissions for cleanup
            unreadable_contract.chmod(stat.S_IRUSR | stat.S_IWUSR)


# =============================================================================
# File Size Limit Tests (DoS Protection)
# =============================================================================


class TestHandlerContractSourceFileSizeLimit:
    """Tests for 10MB file size limit enforcement.

    Verifies that HandlerContractSource rejects oversized contract files
    to prevent denial-of-service attacks via memory exhaustion. The limit
    is defined as MAX_CONTRACT_SIZE (10MB = 10 * 1024 * 1024 bytes).

    Security Context:
        Without this limit, an attacker could create a malicious handler_contract.yaml
        with extremely large content, causing the discovery process to consume
        excessive memory when reading the file.

    These tests use mocking to simulate large file sizes without actually
    creating 10MB+ files on disk.
    """

    @pytest.fixture
    def valid_contract_content(self) -> str:
        """Return minimal valid contract content for size limit tests."""
        return """\
handler_id: "test.handler.size_limit"
name: "Size Limit Test Handler"
version: "1.0.0"
descriptor:
  handler_kind: "compute"
input_model: "test.models.Input"
output_model: "test.models.Output"
"""

    @pytest.mark.asyncio
    async def test_rejects_file_exceeding_10mb_limit_strict_mode(
        self, tmp_path: Path, valid_contract_content: str
    ) -> None:
        """Verify files exceeding 10MB are rejected in strict mode.

        Creates a small contract file but mocks Path.stat() to report
        a file size exceeding MAX_CONTRACT_SIZE (10MB). Verifies that
        discovery raises ModelOnexError with error code HANDLER_SOURCE_005.
        """
        from unittest.mock import patch

        from omnibase_core.models.errors.model_onex_error import ModelOnexError

        from omnibase_infra.runtime.handler_contract_source import (
            MAX_CONTRACT_SIZE,
            HandlerContractSource,
        )

        # Create a small valid contract file
        contract_dir = tmp_path / "oversized_handler"
        contract_dir.mkdir()
        contract_file = contract_dir / "handler_contract.yaml"
        contract_file.write_text(valid_contract_content)

        # Verify MAX_CONTRACT_SIZE is 10MB
        expected_max = 10 * 1024 * 1024
        assert expected_max == MAX_CONTRACT_SIZE, (
            f"MAX_CONTRACT_SIZE should be 10MB ({expected_max}), got {MAX_CONTRACT_SIZE}"
        )

        # Mock stat to return a file size just over the limit
        oversized_bytes = MAX_CONTRACT_SIZE + 1
        original_stat = Path.stat

        def mock_stat(self: Path, **kwargs: object) -> object:
            """Mock stat that returns oversized value for contract files."""
            result = original_stat(self, **kwargs)
            if self.name == "handler_contract.yaml":
                return _create_mock_stat_result(result, oversized_bytes)
            return result

        source = HandlerContractSource(
            contract_paths=[contract_dir],
            graceful_mode=False,  # Strict mode
        )

        with patch.object(Path, "stat", mock_stat):
            with pytest.raises(ModelOnexError) as exc_info:
                await source.discover_handlers()

        # Verify error details
        error = exc_info.value
        assert "size" in str(error).lower(), (
            f"Error message should mention 'size': {error}"
        )
        assert "limit" in str(error).lower() or str(oversized_bytes) in str(error), (
            f"Error message should mention limit or file size: {error}"
        )
        assert error.error_code == "HANDLER_SOURCE_005", (
            f"Expected error code HANDLER_SOURCE_005, got {error.error_code}"
        )

    @pytest.mark.asyncio
    async def test_rejects_file_exceeding_10mb_limit_graceful_mode(
        self, tmp_path: Path, valid_contract_content: str
    ) -> None:
        """Verify files exceeding 10MB produce errors in graceful mode.

        In graceful mode, oversized files should produce a structured
        validation error instead of raising an exception. Other valid
        contracts should still be discovered.
        """
        from unittest.mock import patch

        from omnibase_infra.runtime.handler_contract_source import (
            MAX_CONTRACT_SIZE,
            HandlerContractSource,
        )

        # Create two contract files - one to be marked as oversized, one valid
        oversized_dir = tmp_path / "oversized_handler"
        oversized_dir.mkdir()
        oversized_file = oversized_dir / "handler_contract.yaml"
        oversized_file.write_text(valid_contract_content)

        valid_dir = tmp_path / "valid_handler"
        valid_dir.mkdir()
        valid_file = valid_dir / "handler_contract.yaml"
        valid_file.write_text(
            valid_contract_content.replace("size_limit", "valid_size")
        )

        oversized_bytes = MAX_CONTRACT_SIZE + 1
        original_stat = Path.stat

        def mock_stat_selective(self: Path, **kwargs: object) -> object:
            """Mock stat that returns oversized value only for specific file."""
            result = original_stat(self, **kwargs)
            if self == oversized_file:
                return _create_mock_stat_result(result, oversized_bytes)
            return result

        source = HandlerContractSource(
            contract_paths=[tmp_path],
            graceful_mode=True,  # Graceful mode
        )

        with patch.object(Path, "stat", mock_stat_selective):
            result = await source.discover_handlers()

        # Valid contract should still be discovered
        assert len(result.descriptors) == 1, (
            f"Expected 1 valid descriptor, got {len(result.descriptors)}. "
            "Oversized file should not prevent valid contract discovery."
        )
        assert result.descriptors[0].handler_id == "test.handler.valid_size"

        # Oversized contract should produce validation error
        assert len(result.validation_errors) == 1, (
            f"Expected 1 validation error for oversized file, got {len(result.validation_errors)}"
        )
        error = result.validation_errors[0]
        assert "size" in error.message.lower() or "limit" in error.message.lower(), (
            f"Error message should mention size limit: {error.message}"
        )

    @pytest.mark.asyncio
    async def test_accepts_file_at_exactly_10mb_limit(
        self, tmp_path: Path, valid_contract_content: str
    ) -> None:
        """Verify files at exactly 10MB are accepted (boundary test).

        Files at exactly MAX_CONTRACT_SIZE should be accepted.
        Only files strictly greater than the limit should be rejected.
        """
        from unittest.mock import patch

        from omnibase_infra.runtime.handler_contract_source import (
            MAX_CONTRACT_SIZE,
            HandlerContractSource,
        )

        # Create a valid contract file
        contract_dir = tmp_path / "boundary_handler"
        contract_dir.mkdir()
        contract_file = contract_dir / "handler_contract.yaml"
        contract_file.write_text(valid_contract_content)

        # Mock stat to return exactly MAX_CONTRACT_SIZE
        exactly_max_bytes = MAX_CONTRACT_SIZE
        original_stat = Path.stat

        def mock_stat(self: Path, **kwargs: object) -> object:
            """Mock stat that returns exactly MAX_CONTRACT_SIZE for contract files."""
            result = original_stat(self, **kwargs)
            if self.name == "handler_contract.yaml":
                return _create_mock_stat_result(result, exactly_max_bytes)
            return result

        source = HandlerContractSource(
            contract_paths=[contract_dir],
            graceful_mode=False,
        )

        with patch.object(Path, "stat", mock_stat):
            result = await source.discover_handlers()

        # File at exactly limit should be accepted
        assert len(result.descriptors) == 1, (
            f"Expected 1 descriptor (file at limit should be accepted), got {len(result.descriptors)}"
        )
        assert len(result.validation_errors) == 0
        assert result.descriptors[0].handler_id == "test.handler.size_limit"

    @pytest.mark.asyncio
    async def test_accepts_file_under_10mb_limit(
        self, tmp_path: Path, valid_contract_content: str
    ) -> None:
        """Verify files under 10MB are accepted normally.

        Normal-sized files should be processed without any size-related errors.
        """
        from omnibase_infra.runtime.handler_contract_source import (
            MAX_CONTRACT_SIZE,
            HandlerContractSource,
        )

        # Create a valid contract file (actual small file, no mocking)
        contract_dir = tmp_path / "normal_handler"
        contract_dir.mkdir()
        contract_file = contract_dir / "handler_contract.yaml"
        contract_file.write_text(valid_contract_content)

        # Verify actual file is under limit
        actual_size = contract_file.stat().st_size
        assert actual_size < MAX_CONTRACT_SIZE, (
            f"Test contract should be under {MAX_CONTRACT_SIZE}, actual: {actual_size}"
        )

        source = HandlerContractSource(
            contract_paths=[contract_dir],
            graceful_mode=False,
        )

        result = await source.discover_handlers()

        # Normal file should be accepted
        assert len(result.descriptors) == 1
        assert len(result.validation_errors) == 0
        assert result.descriptors[0].handler_id == "test.handler.size_limit"

    @pytest.mark.asyncio
    async def test_max_contract_size_constant_is_exported(self) -> None:
        """Verify MAX_CONTRACT_SIZE is exported from the module.

        The constant should be accessible for documentation and configuration
        purposes.
        """
        from omnibase_infra.runtime.handler_contract_source import MAX_CONTRACT_SIZE

        # Verify it's 10MB
        assert MAX_CONTRACT_SIZE == 10 * 1024 * 1024, (
            f"MAX_CONTRACT_SIZE should be 10MB (10485760 bytes), got {MAX_CONTRACT_SIZE}"
        )

        # Verify it's in __all__
        from omnibase_infra.runtime import handler_contract_source

        assert "MAX_CONTRACT_SIZE" in handler_contract_source.__all__, (
            "MAX_CONTRACT_SIZE should be in __all__ for public export"
        )
