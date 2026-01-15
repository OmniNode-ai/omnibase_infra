# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for error handling when projector contracts are malformed.

These tests verify that ProjectorPluginLoader and ModelProjectorContract
properly handle and report errors for various types of malformed contracts.

Test Categories:
    1. Missing required fields (projector_id, name, aggregate_type, etc.)
    2. Invalid field values (wrong types, invalid enums)
    3. Malformed YAML syntax (parse errors)
    4. Missing or invalid schema definition
    5. Invalid projection behavior configuration

Related Tickets:
    - OMN-1170: Create registration_projector.yaml contract with parity tests
    - OMN-1169: ProjectorShell contract-driven projections
    - PR #146: Add error handling tests for malformed contracts

.. versionadded:: 0.7.0
    Created for PR #146 review feedback - error handling coverage.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omnibase_core.models.errors.model_onex_error import ModelOnexError
from pydantic import ValidationError

# =============================================================================
# Test Markers
# =============================================================================

pytestmark = [
    pytest.mark.unit,
]


# =============================================================================
# Constants - Malformed Contract YAML Content
# =============================================================================

# Missing projector_id (required field)
CONTRACT_MISSING_PROJECTOR_ID = """
projector_kind: materialized_view
name: "Test Projector"
version: "1.0.0"
aggregate_type: "TestAggregate"
consumed_events:
  - test.created.v1
projection_schema:
  table: test_projections
  primary_key: id
  columns:
    - name: id
      type: UUID
      source: event.payload.id
behavior:
  mode: upsert
"""

# Missing aggregate_type (required field)
CONTRACT_MISSING_AGGREGATE_TYPE = """
projector_kind: materialized_view
projector_id: "missing-aggregate-projector"
name: "Test Projector"
version: "1.0.0"
consumed_events:
  - test.created.v1
projection_schema:
  table: test_projections
  primary_key: id
  columns:
    - name: id
      type: UUID
      source: event.payload.id
behavior:
  mode: upsert
"""

# Missing projection_schema (required field)
CONTRACT_MISSING_SCHEMA = """
projector_kind: materialized_view
projector_id: "missing-schema-projector"
name: "Test Projector"
version: "1.0.0"
aggregate_type: "TestAggregate"
consumed_events:
  - test.created.v1
behavior:
  mode: upsert
"""

# Missing behavior section (required field)
CONTRACT_MISSING_BEHAVIOR = """
projector_kind: materialized_view
projector_id: "missing-behavior-projector"
name: "Test Projector"
version: "1.0.0"
aggregate_type: "TestAggregate"
consumed_events:
  - test.created.v1
projection_schema:
  table: test_projections
  primary_key: id
  columns:
    - name: id
      type: UUID
      source: event.payload.id
"""

# Invalid projector_kind value (wrong enum)
CONTRACT_INVALID_PROJECTOR_KIND = """
projector_kind: invalid_kind
projector_id: "invalid-kind-projector"
name: "Test Projector"
version: "1.0.0"
aggregate_type: "TestAggregate"
consumed_events:
  - test.created.v1
projection_schema:
  table: test_projections
  primary_key: id
  columns:
    - name: id
      type: UUID
      source: event.payload.id
behavior:
  mode: upsert
"""

# Invalid behavior mode (wrong enum)
CONTRACT_INVALID_BEHAVIOR_MODE = """
projector_kind: materialized_view
projector_id: "invalid-mode-projector"
name: "Test Projector"
version: "1.0.0"
aggregate_type: "TestAggregate"
consumed_events:
  - test.created.v1
projection_schema:
  table: test_projections
  primary_key: id
  columns:
    - name: id
      type: UUID
      source: event.payload.id
behavior:
  mode: invalid_mode
"""

# Empty consumed_events (validation error)
CONTRACT_EMPTY_CONSUMED_EVENTS = """
projector_kind: materialized_view
projector_id: "empty-events-projector"
name: "Test Projector"
version: "1.0.0"
aggregate_type: "TestAggregate"
consumed_events: []
projection_schema:
  table: test_projections
  primary_key: id
  columns:
    - name: id
      type: UUID
      source: event.payload.id
behavior:
  mode: upsert
"""

# Empty columns in schema (validation error)
CONTRACT_EMPTY_COLUMNS = """
projector_kind: materialized_view
projector_id: "empty-columns-projector"
name: "Test Projector"
version: "1.0.0"
aggregate_type: "TestAggregate"
consumed_events:
  - test.created.v1
projection_schema:
  table: test_projections
  primary_key: id
  columns: []
behavior:
  mode: upsert
"""

# Blank table name (validation error)
CONTRACT_BLANK_TABLE_NAME = """
projector_kind: materialized_view
projector_id: "blank-table-projector"
name: "Test Projector"
version: "1.0.0"
aggregate_type: "TestAggregate"
consumed_events:
  - test.created.v1
projection_schema:
  table: "   "
  primary_key: id
  columns:
    - name: id
      type: UUID
      source: event.payload.id
behavior:
  mode: upsert
"""

# Missing column name (validation error)
CONTRACT_COLUMN_MISSING_NAME = """
projector_kind: materialized_view
projector_id: "column-no-name-projector"
name: "Test Projector"
version: "1.0.0"
aggregate_type: "TestAggregate"
consumed_events:
  - test.created.v1
projection_schema:
  table: test_projections
  primary_key: id
  columns:
    - type: UUID
      source: event.payload.id
behavior:
  mode: upsert
"""

# Missing column source (validation error)
CONTRACT_COLUMN_MISSING_SOURCE = """
projector_kind: materialized_view
projector_id: "column-no-source-projector"
name: "Test Projector"
version: "1.0.0"
aggregate_type: "TestAggregate"
consumed_events:
  - test.created.v1
projection_schema:
  table: test_projections
  primary_key: id
  columns:
    - name: id
      type: UUID
behavior:
  mode: upsert
"""

# Malformed YAML syntax (unclosed quote)
CONTRACT_MALFORMED_YAML_UNCLOSED_QUOTE = """
projector_kind: materialized_view
projector_id: "malformed-yaml
name: "Test Projector"
"""

# Malformed YAML syntax (invalid indentation)
CONTRACT_MALFORMED_YAML_INDENTATION = """
projector_kind: materialized_view
projector_id: "malformed-indent"
  name: "Wrong Indent"
 aggregate_type: "TestAggregate"
"""

# Malformed YAML syntax (invalid character)
CONTRACT_MALFORMED_YAML_INVALID_CHAR = """
projector_kind: materialized_view
projector_id: "malformed-char"
consumed_events: [
  - test.created.v1
]
"""

# Type mismatch - string where list expected
CONTRACT_TYPE_MISMATCH_EVENTS = """
projector_kind: materialized_view
projector_id: "type-mismatch-projector"
name: "Test Projector"
version: "1.0.0"
aggregate_type: "TestAggregate"
consumed_events: "test.created.v1"
projection_schema:
  table: test_projections
  primary_key: id
  columns:
    - name: id
      type: UUID
      source: event.payload.id
behavior:
  mode: upsert
"""

# Type mismatch - list where string expected
CONTRACT_TYPE_MISMATCH_NAME = """
projector_kind: materialized_view
projector_id: "type-mismatch-name-projector"
name:
  - "Invalid"
  - "Name"
version: "1.0.0"
aggregate_type: "TestAggregate"
consumed_events:
  - test.created.v1
projection_schema:
  table: test_projections
  primary_key: id
  columns:
    - name: id
      type: UUID
      source: event.payload.id
behavior:
  mode: upsert
"""


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_schema_manager():
    """Create a mock schema manager for loader tests."""
    from unittest.mock import MagicMock

    mock = MagicMock()
    mock.validate_schema.return_value = True
    return mock


def _write_contract(
    tmp_path: Path, content: str, filename: str = "test_projector.yaml"
) -> Path:
    """Write contract content to a temporary file.

    Args:
        tmp_path: pytest tmp_path fixture.
        content: YAML content to write.
        filename: Name of the contract file.

    Returns:
        Path to the created contract file.
    """
    contract_file = tmp_path / filename
    contract_file.write_text(content)
    return contract_file


# =============================================================================
# Test Classes
# =============================================================================


class TestMissingRequiredFields:
    """Tests for contracts missing required fields.

    These tests verify that appropriate validation errors are raised
    when contracts are missing required fields like projector_id,
    aggregate_type, projection_schema, or behavior.
    """

    @pytest.mark.asyncio
    async def test_missing_projector_id_raises_validation_error(
        self, tmp_path: Path, mock_schema_manager
    ) -> None:
        """Contract missing projector_id should raise validation error.

        Given: Contract YAML without projector_id field
        When: Loader attempts to load the contract
        Then: Raises ModelOnexError with validation message
        """
        from omnibase_infra.runtime.projector_plugin_loader import ProjectorPluginLoader

        contract_file = _write_contract(tmp_path, CONTRACT_MISSING_PROJECTOR_ID)
        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        with pytest.raises(ModelOnexError) as exc_info:
            await loader.load_from_contract(contract_file)

        error_msg = str(exc_info.value).lower()
        assert "validation" in error_msg
        assert "projector_id" in error_msg

    @pytest.mark.asyncio
    async def test_missing_aggregate_type_raises_validation_error(
        self, tmp_path: Path, mock_schema_manager
    ) -> None:
        """Contract missing aggregate_type should raise validation error.

        Given: Contract YAML without aggregate_type field
        When: Loader attempts to load the contract
        Then: Raises ModelOnexError with validation message about aggregate_type
        """
        from omnibase_infra.runtime.projector_plugin_loader import ProjectorPluginLoader

        contract_file = _write_contract(tmp_path, CONTRACT_MISSING_AGGREGATE_TYPE)
        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        with pytest.raises(ModelOnexError) as exc_info:
            await loader.load_from_contract(contract_file)

        error_msg = str(exc_info.value).lower()
        assert "validation" in error_msg
        assert "aggregate_type" in error_msg

    @pytest.mark.asyncio
    async def test_missing_projection_schema_raises_validation_error(
        self, tmp_path: Path, mock_schema_manager
    ) -> None:
        """Contract missing projection_schema should raise validation error.

        Given: Contract YAML without projection_schema section
        When: Loader attempts to load the contract
        Then: Raises ModelOnexError with validation message about schema
        """
        from omnibase_infra.runtime.projector_plugin_loader import ProjectorPluginLoader

        contract_file = _write_contract(tmp_path, CONTRACT_MISSING_SCHEMA)
        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        with pytest.raises(ModelOnexError) as exc_info:
            await loader.load_from_contract(contract_file)

        error_msg = str(exc_info.value).lower()
        assert "validation" in error_msg
        assert "projection_schema" in error_msg or "schema" in error_msg

    @pytest.mark.asyncio
    async def test_missing_behavior_raises_validation_error(
        self, tmp_path: Path, mock_schema_manager
    ) -> None:
        """Contract missing behavior section should raise validation error.

        Given: Contract YAML without behavior section
        When: Loader attempts to load the contract
        Then: Raises ModelOnexError with validation message about behavior
        """
        from omnibase_infra.runtime.projector_plugin_loader import ProjectorPluginLoader

        contract_file = _write_contract(tmp_path, CONTRACT_MISSING_BEHAVIOR)
        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        with pytest.raises(ModelOnexError) as exc_info:
            await loader.load_from_contract(contract_file)

        error_msg = str(exc_info.value).lower()
        assert "validation" in error_msg
        assert "behavior" in error_msg


class TestInvalidFieldValues:
    """Tests for contracts with invalid field values.

    These tests verify that appropriate validation errors are raised
    when contracts contain invalid values like wrong enum types,
    empty lists where non-empty is required, or blank strings.
    """

    @pytest.mark.asyncio
    async def test_invalid_projector_kind_raises_validation_error(
        self, tmp_path: Path, mock_schema_manager
    ) -> None:
        """Contract with invalid projector_kind should raise validation error.

        Given: Contract YAML with invalid projector_kind enum value
        When: Loader attempts to load the contract
        Then: Raises ModelOnexError with validation message about projector_kind
        """
        from omnibase_infra.runtime.projector_plugin_loader import ProjectorPluginLoader

        contract_file = _write_contract(tmp_path, CONTRACT_INVALID_PROJECTOR_KIND)
        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        with pytest.raises(ModelOnexError) as exc_info:
            await loader.load_from_contract(contract_file)

        error_msg = str(exc_info.value).lower()
        assert "validation" in error_msg

    @pytest.mark.asyncio
    async def test_invalid_behavior_mode_raises_validation_error(
        self, tmp_path: Path, mock_schema_manager
    ) -> None:
        """Contract with invalid behavior mode should raise validation error.

        Given: Contract YAML with invalid behavior.mode enum value
        When: Loader attempts to load the contract
        Then: Raises ModelOnexError with validation message about mode
        """
        from omnibase_infra.runtime.projector_plugin_loader import ProjectorPluginLoader

        contract_file = _write_contract(tmp_path, CONTRACT_INVALID_BEHAVIOR_MODE)
        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        with pytest.raises(ModelOnexError) as exc_info:
            await loader.load_from_contract(contract_file)

        error_msg = str(exc_info.value).lower()
        assert "validation" in error_msg
        assert "mode" in error_msg or "behavior" in error_msg

    @pytest.mark.asyncio
    async def test_empty_consumed_events_raises_validation_error(
        self, tmp_path: Path, mock_schema_manager
    ) -> None:
        """Contract with empty consumed_events should raise validation error.

        Given: Contract YAML with consumed_events: []
        When: Loader attempts to load the contract
        Then: Raises ModelOnexError indicating consumed_events cannot be empty
        """
        from omnibase_infra.runtime.projector_plugin_loader import ProjectorPluginLoader

        contract_file = _write_contract(tmp_path, CONTRACT_EMPTY_CONSUMED_EVENTS)
        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        with pytest.raises(ModelOnexError) as exc_info:
            await loader.load_from_contract(contract_file)

        error_msg = str(exc_info.value).lower()
        # Should indicate validation error about events or min length
        assert "validation" in error_msg or "event" in error_msg or "min" in error_msg

    @pytest.mark.asyncio
    async def test_empty_columns_raises_validation_error(
        self, tmp_path: Path, mock_schema_manager
    ) -> None:
        """Contract with empty columns list should raise validation error.

        Given: Contract YAML with projection_schema.columns: []
        When: Loader attempts to load the contract
        Then: Raises ModelOnexError indicating columns cannot be empty
        """
        from omnibase_infra.runtime.projector_plugin_loader import ProjectorPluginLoader

        contract_file = _write_contract(tmp_path, CONTRACT_EMPTY_COLUMNS)
        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        with pytest.raises(ModelOnexError) as exc_info:
            await loader.load_from_contract(contract_file)

        error_msg = str(exc_info.value).lower()
        assert "validation" in error_msg or "column" in error_msg or "min" in error_msg

    @pytest.mark.asyncio
    async def test_blank_table_name_loads_but_will_fail_at_runtime(
        self, tmp_path: Path, mock_schema_manager
    ) -> None:
        """Contract with blank table name currently loads but will fail at SQL execution.

        Note: ModelProjectorContract currently does not validate that table names
        are non-blank. This test documents current behavior. The blank table name
        will cause SQL execution failures when the projector attempts to project
        events (invalid SQL identifier).

        Given: Contract YAML with projection_schema.table: "   " (whitespace only)
        When: Loader attempts to load the contract
        Then: Contract loads successfully (validation gap - will fail at SQL runtime)
        """
        from omnibase_infra.runtime.projector_plugin_loader import ProjectorPluginLoader

        contract_file = _write_contract(tmp_path, CONTRACT_BLANK_TABLE_NAME)
        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        # NOTE: This currently succeeds despite blank table name.
        # A stricter implementation would validate table names at contract load time.
        projector = await loader.load_from_contract(contract_file)

        # Document the gap: blank table name is accepted by model validation
        assert projector.contract.projection_schema.table == "   "


class TestMalformedYAML:
    """Tests for malformed YAML syntax.

    These tests verify that appropriate parse errors are raised
    when contracts contain invalid YAML syntax like unclosed quotes,
    invalid indentation, or invalid characters.
    """

    @pytest.mark.asyncio
    async def test_unclosed_quote_raises_parse_error(
        self, tmp_path: Path, mock_schema_manager
    ) -> None:
        """Contract with unclosed quote should raise parse error.

        Given: Contract YAML with unclosed string quote
        When: Loader attempts to load the contract
        Then: Raises ModelOnexError with YAML parse error message
        """
        from omnibase_infra.runtime.projector_plugin_loader import ProjectorPluginLoader

        contract_file = _write_contract(
            tmp_path, CONTRACT_MALFORMED_YAML_UNCLOSED_QUOTE
        )
        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        with pytest.raises(ModelOnexError) as exc_info:
            await loader.load_from_contract(contract_file)

        error_msg = str(exc_info.value).lower()
        assert "parse" in error_msg or "yaml" in error_msg

    @pytest.mark.asyncio
    async def test_invalid_indentation_raises_parse_error(
        self, tmp_path: Path, mock_schema_manager
    ) -> None:
        """Contract with invalid indentation should raise parse error.

        Given: Contract YAML with inconsistent indentation
        When: Loader attempts to load the contract
        Then: Raises ModelOnexError with YAML parse error message
        """
        from omnibase_infra.runtime.projector_plugin_loader import ProjectorPluginLoader

        contract_file = _write_contract(tmp_path, CONTRACT_MALFORMED_YAML_INDENTATION)
        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        with pytest.raises(ModelOnexError) as exc_info:
            await loader.load_from_contract(contract_file)

        error_msg = str(exc_info.value).lower()
        assert "parse" in error_msg or "yaml" in error_msg or "validation" in error_msg

    @pytest.mark.asyncio
    async def test_invalid_yaml_syntax_raises_parse_error(
        self, tmp_path: Path, mock_schema_manager
    ) -> None:
        """Contract with invalid YAML syntax should raise parse error.

        Given: Contract YAML with invalid character sequence
        When: Loader attempts to load the contract
        Then: Raises ModelOnexError with YAML parse error message
        """
        from omnibase_infra.runtime.projector_plugin_loader import ProjectorPluginLoader

        contract_file = _write_contract(tmp_path, CONTRACT_MALFORMED_YAML_INVALID_CHAR)
        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        with pytest.raises(ModelOnexError) as exc_info:
            await loader.load_from_contract(contract_file)

        error_msg = str(exc_info.value).lower()
        assert "parse" in error_msg or "yaml" in error_msg or "validation" in error_msg


class TestTypeMismatches:
    """Tests for type mismatches in contract fields.

    These tests verify that appropriate validation errors are raised
    when contract fields have incorrect types, like a string where
    a list is expected or vice versa.
    """

    @pytest.mark.asyncio
    async def test_string_instead_of_list_raises_validation_error(
        self, tmp_path: Path, mock_schema_manager
    ) -> None:
        """Contract with string where list expected should raise validation error.

        Given: Contract YAML with consumed_events as string instead of list
        When: Loader attempts to load the contract
        Then: Raises ModelOnexError with type validation message
        """
        from omnibase_infra.runtime.projector_plugin_loader import ProjectorPluginLoader

        contract_file = _write_contract(tmp_path, CONTRACT_TYPE_MISMATCH_EVENTS)
        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        with pytest.raises(ModelOnexError) as exc_info:
            await loader.load_from_contract(contract_file)

        error_msg = str(exc_info.value).lower()
        assert "validation" in error_msg

    @pytest.mark.asyncio
    async def test_list_instead_of_string_raises_validation_error(
        self, tmp_path: Path, mock_schema_manager
    ) -> None:
        """Contract with list where string expected should raise validation error.

        Given: Contract YAML with name as list instead of string
        When: Loader attempts to load the contract
        Then: Raises ModelOnexError with type validation message
        """
        from omnibase_infra.runtime.projector_plugin_loader import ProjectorPluginLoader

        contract_file = _write_contract(tmp_path, CONTRACT_TYPE_MISMATCH_NAME)
        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        with pytest.raises(ModelOnexError) as exc_info:
            await loader.load_from_contract(contract_file)

        error_msg = str(exc_info.value).lower()
        assert "validation" in error_msg


class TestMissingSchemaDefinition:
    """Tests for missing or incomplete schema definitions.

    These tests verify that appropriate validation errors are raised
    when the projection_schema section is missing required fields
    like column name or source.
    """

    @pytest.mark.asyncio
    async def test_column_missing_name_raises_validation_error(
        self, tmp_path: Path, mock_schema_manager
    ) -> None:
        """Column without name should raise validation error.

        Given: Contract YAML with column definition missing 'name' field
        When: Loader attempts to load the contract
        Then: Raises ModelOnexError with validation message about column name
        """
        from omnibase_infra.runtime.projector_plugin_loader import ProjectorPluginLoader

        contract_file = _write_contract(tmp_path, CONTRACT_COLUMN_MISSING_NAME)
        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        with pytest.raises(ModelOnexError) as exc_info:
            await loader.load_from_contract(contract_file)

        error_msg = str(exc_info.value).lower()
        assert "validation" in error_msg
        assert "name" in error_msg or "column" in error_msg

    @pytest.mark.asyncio
    async def test_column_missing_source_raises_validation_error(
        self, tmp_path: Path, mock_schema_manager
    ) -> None:
        """Column without source should raise validation error.

        Given: Contract YAML with column definition missing 'source' field
        When: Loader attempts to load the contract
        Then: Raises ModelOnexError with validation message about column source
        """
        from omnibase_infra.runtime.projector_plugin_loader import ProjectorPluginLoader

        contract_file = _write_contract(tmp_path, CONTRACT_COLUMN_MISSING_SOURCE)
        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        with pytest.raises(ModelOnexError) as exc_info:
            await loader.load_from_contract(contract_file)

        error_msg = str(exc_info.value).lower()
        assert "validation" in error_msg
        assert "source" in error_msg or "column" in error_msg


class TestGracefulModeErrorCollection:
    """Tests for graceful mode error collection with malformed contracts.

    These tests verify that in graceful mode, errors are collected
    and reported rather than raising on the first error, allowing
    discovery of multiple issues in a single pass.
    """

    @pytest.mark.asyncio
    async def test_graceful_mode_collects_multiple_errors(
        self, tmp_path: Path, mock_schema_manager
    ) -> None:
        """Graceful mode should collect errors from multiple invalid contracts.

        Given: Directory with multiple malformed contracts
        When: Loader discovers with graceful_mode=True
        Then: Returns result with errors collected, valid contracts loaded
        """
        from omnibase_infra.runtime.projector_plugin_loader import ProjectorPluginLoader

        # Create one valid and multiple invalid contracts
        valid_content = """
projector_kind: materialized_view
projector_id: "valid-projector"
name: "Valid Projector"
version: "1.0.0"
aggregate_type: "ValidAggregate"
consumed_events:
  - test.created.v1
projection_schema:
  table: valid_projections
  primary_key: id
  columns:
    - name: id
      type: UUID
      source: event.payload.id
behavior:
  mode: upsert
"""
        _write_contract(tmp_path, valid_content, "valid_projector.yaml")
        _write_contract(
            tmp_path,
            CONTRACT_MALFORMED_YAML_UNCLOSED_QUOTE,
            "malformed1_projector.yaml",
        )
        _write_contract(
            tmp_path, CONTRACT_MISSING_AGGREGATE_TYPE, "malformed2_projector.yaml"
        )

        loader = ProjectorPluginLoader(
            schema_manager=mock_schema_manager,
            graceful_mode=True,
        )

        result = await loader.discover_with_errors(tmp_path)

        # Should have loaded the valid contract
        assert len(result.projectors) == 1
        assert result.projectors[0].projector_id == "valid-projector"

        # Should have collected errors for the invalid contracts
        assert len(result.validation_errors) >= 2

    @pytest.mark.asyncio
    async def test_graceful_mode_error_messages_are_helpful(
        self, tmp_path: Path, mock_schema_manager
    ) -> None:
        """Graceful mode errors should contain helpful remediation hints.

        Given: Malformed contract in graceful mode
        When: Loader discovers with graceful_mode=True
        Then: Error contains remediation hint
        """
        from omnibase_infra.runtime.projector_plugin_loader import ProjectorPluginLoader

        _write_contract(
            tmp_path, CONTRACT_MALFORMED_YAML_UNCLOSED_QUOTE, "malformed_projector.yaml"
        )

        loader = ProjectorPluginLoader(
            schema_manager=mock_schema_manager,
            graceful_mode=True,
        )

        result = await loader.discover_with_errors(tmp_path)

        assert len(result.validation_errors) == 1
        error = result.validation_errors[0]

        # Error should have helpful information
        assert error.error_type is not None
        assert error.message is not None
        assert error.remediation_hint is not None
        assert len(error.remediation_hint) > 0


class TestDirectModelValidation:
    """Tests for direct ModelProjectorContract validation.

    These tests verify that the Pydantic model itself properly
    validates contracts and provides helpful error messages,
    independent of the loader.
    """

    def test_direct_model_validation_missing_projector_id(self) -> None:
        """Direct model validation should catch missing projector_id.

        Given: Contract data dict without projector_id
        When: ModelProjectorContract.model_validate() is called
        Then: Raises ValidationError with field location
        """
        from omnibase_core.models.projectors import ModelProjectorContract

        data = {
            "projector_kind": "materialized_view",
            "name": "Test",
            "version": "1.0.0",
            "aggregate_type": "Test",
            "consumed_events": ["test.v1"],
            "projection_schema": {
                "table": "test",
                "primary_key": "id",
                "columns": [{"name": "id", "type": "UUID", "source": "event.id"}],
            },
            "behavior": {"mode": "upsert"},
        }

        with pytest.raises(ValidationError) as exc_info:
            ModelProjectorContract.model_validate(data)

        errors = exc_info.value.errors()
        assert len(errors) > 0
        # Should reference the missing field
        error_locs = [str(e.get("loc", ())) for e in errors]
        assert any("projector_id" in loc for loc in error_locs)

    def test_direct_model_validation_invalid_mode(self) -> None:
        """Direct model validation should catch invalid behavior mode.

        Given: Contract data dict with invalid mode value
        When: ModelProjectorContract.model_validate() is called
        Then: Raises ValidationError indicating invalid mode
        """
        from omnibase_core.models.projectors import ModelProjectorContract

        data = {
            "projector_kind": "materialized_view",
            "projector_id": "test",
            "name": "Test",
            "version": "1.0.0",
            "aggregate_type": "Test",
            "consumed_events": ["test.v1"],
            "projection_schema": {
                "table": "test",
                "primary_key": "id",
                "columns": [{"name": "id", "type": "UUID", "source": "event.id"}],
            },
            "behavior": {"mode": "invalid_mode"},
        }

        with pytest.raises(ValidationError) as exc_info:
            ModelProjectorContract.model_validate(data)

        errors = exc_info.value.errors()
        assert len(errors) > 0

    def test_direct_model_validation_empty_events_list(self) -> None:
        """Direct model validation should catch empty consumed_events.

        Given: Contract data dict with empty consumed_events list
        When: ModelProjectorContract.model_validate() is called
        Then: Raises ValidationError indicating min_length constraint
        """
        from omnibase_core.models.projectors import ModelProjectorContract

        data = {
            "projector_kind": "materialized_view",
            "projector_id": "test",
            "name": "Test",
            "version": "1.0.0",
            "aggregate_type": "Test",
            "consumed_events": [],  # Empty - should fail
            "projection_schema": {
                "table": "test",
                "primary_key": "id",
                "columns": [{"name": "id", "type": "UUID", "source": "event.id"}],
            },
            "behavior": {"mode": "upsert"},
        }

        with pytest.raises(ValidationError) as exc_info:
            ModelProjectorContract.model_validate(data)

        errors = exc_info.value.errors()
        assert len(errors) > 0


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "TestDirectModelValidation",
    "TestGracefulModeErrorCollection",
    "TestInvalidFieldValues",
    "TestMalformedYAML",
    "TestMissingRequiredFields",
    "TestMissingSchemaDefinition",
    "TestTypeMismatches",
]
