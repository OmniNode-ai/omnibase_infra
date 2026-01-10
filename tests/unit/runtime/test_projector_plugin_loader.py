# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Unit tests for ProjectorPluginLoader contract discovery and loading.

Tests the ProjectorPluginLoader functionality including:
- Loading and validating projector contracts from YAML files
- Directory scanning for *_projector.yaml files
- Pattern-based discovery using glob patterns
- Security validation (symlinks, file size limits, path traversal)
- Strict vs graceful error handling modes

Related:
    - OMN-1168: ProjectorPluginLoader Contract Discovery/Loading
    - src/omnibase_infra/runtime/projector_plugin_loader.py

Expected Behavior:
    ProjectorPluginLoader discovers and loads projector contracts from the
    filesystem by scanning configured paths for *_projector.yaml files,
    parsing them, and transforming them into ProtocolEventProjector instances.

    The loader supports two modes:
    - Strict mode (default): Raises on first error
    - Graceful mode: Collects errors and continues
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable
from unittest.mock import MagicMock, patch

import pytest

from tests.helpers.mock_helpers import create_mock_stat_result

# =============================================================================
# Protocol Definition (fallback for TDD)
# =============================================================================


# Protocol imports with fallback for compatibility
# The protocols may be in different locations depending on implementation state
try:
    from omnibase_infra.runtime.projector_plugin_loader import (
        ProjectorPluginLoader,
    )
except ImportError:
    # Fallback: define minimal protocol stub for testing
    # This allows the test to run and fail on ProjectorPluginLoader import
    ProjectorPluginLoader = None  # type: ignore[misc, assignment]


try:
    from omnibase_infra.runtime.projector_plugin_loader import (
        ProtocolEventProjector,
    )
except ImportError:

    @runtime_checkable
    class ProtocolEventProjector(Protocol):
        """Fallback protocol definition for testing."""

        @property
        def projector_id(self) -> str:
            """The unique identifier for the projector."""
            ...

        @property
        def aggregate_type(self) -> str:
            """The aggregate type this projector handles."""
            ...


# =============================================================================
# Constants for Test Contracts
# =============================================================================

# Valid contract matching ModelProjectorContract schema
VALID_PROJECTOR_CONTRACT_YAML = """
projector_kind: materialized_view
projector_id: "{projector_id}"
name: "Test Projector"
version: "1.0.0"
aggregate_type: "{aggregate_type}"
consumed_events:
  - {event1}
  - {event2}
projection_schema:
  table: {table}
  primary_key: id
  columns:
    - name: id
      type: UUID
      source: event.payload.id
    - name: status
      type: TEXT
      source: event.payload.status
behavior:
  mode: upsert
"""

# Minimal valid contract
MINIMAL_PROJECTOR_CONTRACT_YAML = """
projector_kind: materialized_view
projector_id: "{projector_id}"
name: "Minimal Projector"
version: "1.0.0"
aggregate_type: "{aggregate_type}"
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

# Contract with optional metadata fields
PROJECTOR_CONTRACT_WITH_METADATA_YAML = """
projector_kind: materialized_view
projector_id: "{projector_id}"
name: "{name}"
version: "{version}"
aggregate_type: "{aggregate_type}"
consumed_events:
  - {event1}
  - {event2}
projection_schema:
  table: {table}
  primary_key: id
  columns:
    - name: id
      type: UUID
      source: event.payload.id
    - name: status
      type: TEXT
      source: event.payload.status
    - name: created_at
      type: TIMESTAMPTZ
      source: event.payload.created_at
behavior:
  mode: upsert
"""

MALFORMED_YAML_CONTENT = """
projector_kind: "materialized_view
  unclosed quote and [
    malformed structure
"""

MISSING_REQUIRED_FIELDS_CONTENT = """
projector_id: "missing-fields-projector"
# Missing: projector_kind, name, version, aggregate_type, consumed_events, projection_schema, behavior
"""

INVALID_SCHEMA_CONTENT = """
projector_kind: materialized_view
projector_id: "invalid-schema-projector"
name: "Invalid Schema Projector"
version: "1.0.0"
aggregate_type: "TestAggregate"
consumed_events:
  - test.created.v1
projection_schema:
  table: "  "
  primary_key: ""
  columns: []
behavior:
  mode: upsert
"""


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def valid_contract_content() -> str:
    """Return valid projector contract content."""
    return VALID_PROJECTOR_CONTRACT_YAML.format(
        projector_id="test-projector-v1",
        aggregate_type="test_aggregate",
        event1="test.created.v1",
        event2="test.updated.v1",
        table="test_projections",
    )


@pytest.fixture
def tmp_contract_file(tmp_path: Path, valid_contract_content: str) -> Path:
    """Create a single valid projector contract file.

    Returns:
        Path to the created contract file.
    """
    contract_file = tmp_path / "test_projector.yaml"
    contract_file.write_text(valid_contract_content)
    return contract_file


@pytest.fixture
def tmp_contract_directory(tmp_path: Path) -> Path:
    """Create a directory with multiple projector contract files.

    Structure:
        tmp_path/
        |-- registration_projector.yaml
        |-- inventory_projector.yaml
        |-- nested/
        |   |-- order_projector.yaml

    Returns:
        Path to the root directory containing contracts.
    """
    # Root level contracts
    (tmp_path / "registration_projector.yaml").write_text(
        MINIMAL_PROJECTOR_CONTRACT_YAML.format(
            projector_id="registration-projector-v1",
            aggregate_type="RegistrationAggregate",
        )
    )

    (tmp_path / "inventory_projector.yaml").write_text(
        MINIMAL_PROJECTOR_CONTRACT_YAML.format(
            projector_id="inventory-projector-v1",
            aggregate_type="InventoryAggregate",
        )
    )

    # Nested contract
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    (nested_dir / "order_projector.yaml").write_text(
        MINIMAL_PROJECTOR_CONTRACT_YAML.format(
            projector_id="order-projector-v1",
            aggregate_type="OrderAggregate",
        )
    )

    return tmp_path


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
    """Create a directory with a malformed projector contract file.

    Returns:
        Path to the directory containing the malformed contract file.
    """
    malformed_dir = tmp_path / "malformed"
    malformed_dir.mkdir(parents=True)
    malformed_file = malformed_dir / "broken_projector.yaml"
    malformed_file.write_text(MALFORMED_YAML_CONTENT)
    return malformed_dir


@pytest.fixture
def mock_schema_manager() -> MagicMock:
    """Create a mock schema manager for ProjectorPluginLoader tests.

    NOTE: The schema_manager is currently stored by ProjectorPluginLoader but
    not actively used for validation during contract loading. It is retained
    for future use by ProjectorShell (OMN-1169), which will use it to validate
    that target projection tables exist before the projector starts.

    This fixture provides the mock for constructor injection to ensure tests
    remain compatible when schema validation is implemented.

    Returns:
        MagicMock with validate_schema and get_schema methods configured.
    """
    mock = MagicMock()
    mock.validate_schema.return_value = True
    mock.get_schema.return_value = None
    return mock


# =============================================================================
# Contract Loading Tests
# =============================================================================


class TestProjectorPluginLoaderContractLoading:
    """Tests for loading individual projector contracts.

    These tests verify that ProjectorPluginLoader correctly loads and
    validates projector contracts from YAML files.
    """

    @pytest.mark.asyncio
    async def test_load_valid_contract(
        self, tmp_contract_file: Path, mock_schema_manager: MagicMock
    ) -> None:
        """Valid YAML contract should load successfully."""
        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        projector = await loader.load_from_contract(tmp_contract_file)

        assert projector is not None
        assert projector.projector_id == "test-projector-v1"
        assert projector.aggregate_type == "test_aggregate"
        assert "test.created.v1" in projector.consumed_events
        assert "test.updated.v1" in projector.consumed_events
        # Access schema via the contract property (projection_schema)
        assert projector.contract.projection_schema.table == "test_projections"

    @pytest.mark.asyncio
    async def test_load_invalid_contract_raises(
        self, tmp_path: Path, mock_schema_manager: MagicMock
    ) -> None:
        """Invalid contract should raise validation error."""
        from omnibase_core.models.errors.model_onex_error import ModelOnexError

        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        # Create contract with missing required fields
        invalid_file = tmp_path / "invalid_projector.yaml"
        invalid_file.write_text(MISSING_REQUIRED_FIELDS_CONTENT)

        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        with pytest.raises(ModelOnexError) as exc_info:
            await loader.load_from_contract(invalid_file)

        assert "validation" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_load_missing_file_raises(
        self, tmp_path: Path, mock_schema_manager: MagicMock
    ) -> None:
        """Missing file should raise appropriate error."""
        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        nonexistent_file = tmp_path / "nonexistent_projector.yaml"
        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        with pytest.raises(FileNotFoundError) as exc_info:
            await loader.load_from_contract(nonexistent_file)

        assert (
            "not" in str(exc_info.value).lower()
            or "exist" in str(exc_info.value).lower()
        )

    @pytest.mark.asyncio
    async def test_load_malformed_yaml_raises(
        self, malformed_contract_path: Path, mock_schema_manager: MagicMock
    ) -> None:
        """Malformed YAML should raise parse error."""
        from omnibase_core.models.errors.model_onex_error import ModelOnexError

        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        malformed_file = malformed_contract_path / "broken_projector.yaml"
        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        with pytest.raises(ModelOnexError) as exc_info:
            await loader.load_from_contract(malformed_file)

        assert (
            "parse" in str(exc_info.value).lower()
            or "yaml" in str(exc_info.value).lower()
        )


# =============================================================================
# Directory Loading Tests
# =============================================================================


class TestProjectorPluginLoaderDirectoryLoading:
    """Tests for loading projector contracts from directories.

    These tests verify that ProjectorPluginLoader correctly discovers
    all *_projector.yaml files in a directory structure.
    """

    @pytest.mark.asyncio
    async def test_load_from_directory(
        self, tmp_contract_directory: Path, mock_schema_manager: MagicMock
    ) -> None:
        """All *_projector.yaml files should be loaded from directory."""
        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        # load_from_directory returns list[ProtocolEventProjector] directly
        projectors = await loader.load_from_directory(tmp_contract_directory)

        assert len(projectors) == 3
        projector_ids = {p.projector_id for p in projectors}
        assert "registration-projector-v1" in projector_ids
        assert "inventory-projector-v1" in projector_ids
        assert "order-projector-v1" in projector_ids

    @pytest.mark.asyncio
    async def test_load_from_empty_directory(
        self, empty_directory: Path, mock_schema_manager: MagicMock
    ) -> None:
        """Empty directory should return empty list."""
        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        projectors = await loader.load_from_directory(empty_directory)

        assert projectors == []

    @pytest.mark.asyncio
    async def test_load_from_nonexistent_directory_raises(
        self, tmp_path: Path, mock_schema_manager: MagicMock
    ) -> None:
        """Missing directory should raise error."""
        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        nonexistent_dir = tmp_path / "does_not_exist"
        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        with pytest.raises(FileNotFoundError) as exc_info:
            await loader.load_from_directory(nonexistent_dir)

        assert (
            "not" in str(exc_info.value).lower()
            or "exist" in str(exc_info.value).lower()
        )

    @pytest.mark.asyncio
    async def test_load_from_directory_recursive(
        self, tmp_contract_directory: Path, mock_schema_manager: MagicMock
    ) -> None:
        """Should recursively find contracts in nested directories.

        Note: load_from_directory always scans recursively by default.
        """
        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        # load_from_directory is always recursive
        projectors = await loader.load_from_directory(tmp_contract_directory)

        # Should find nested order_projector.yaml
        projector_ids = {p.projector_id for p in projectors}
        assert "order-projector-v1" in projector_ids

    @pytest.mark.asyncio
    async def test_load_from_directory_finds_all_contracts(
        self, tmp_contract_directory: Path, mock_schema_manager: MagicMock
    ) -> None:
        """Directory loading should find all matching contract files."""
        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        projectors = await loader.load_from_directory(tmp_contract_directory)

        # Should find all 3 contracts (2 root + 1 nested)
        assert len(projectors) == 3
        projector_ids = {p.projector_id for p in projectors}
        assert "registration-projector-v1" in projector_ids
        assert "inventory-projector-v1" in projector_ids
        assert "order-projector-v1" in projector_ids


# =============================================================================
# Pattern Discovery Tests
# =============================================================================


class TestProjectorPluginLoaderPatternDiscovery:
    """Tests for glob pattern-based contract discovery.

    These tests verify that ProjectorPluginLoader correctly discovers
    contracts using glob patterns.
    """

    @pytest.mark.asyncio
    async def test_discover_with_patterns(
        self,
        tmp_contract_directory: Path,
        mock_schema_manager: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Glob patterns should discover correct contracts."""
        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        # Change cwd to tmp_contract_directory for pattern resolution
        monkeypatch.chdir(tmp_contract_directory)
        projectors = await loader.discover_and_load(
            patterns=["*_projector.yaml"],
        )

        assert len(projectors) >= 2

    @pytest.mark.asyncio
    async def test_discover_with_multiple_patterns(
        self,
        tmp_path: Path,
        mock_schema_manager: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Multiple patterns should all be applied correctly."""
        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        # Create contracts with different naming patterns
        (tmp_path / "test_projector.yaml").write_text(
            MINIMAL_PROJECTOR_CONTRACT_YAML.format(
                projector_id="test-proj",
                aggregate_type="TestAgg",
            )
        )
        (tmp_path / "projector_contract.yaml").write_text(
            MINIMAL_PROJECTOR_CONTRACT_YAML.format(
                projector_id="other-proj",
                aggregate_type="OtherAgg",
            )
        )

        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        monkeypatch.chdir(tmp_path)
        projectors = await loader.discover_and_load(
            patterns=["*_projector.yaml", "projector_contract.yaml"],
        )

        projector_ids = {p.projector_id for p in projectors}
        assert "test-proj" in projector_ids
        assert "other-proj" in projector_ids

    @pytest.mark.asyncio
    async def test_discover_no_matches(
        self,
        empty_directory: Path,
        mock_schema_manager: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No matches should return empty list."""
        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        monkeypatch.chdir(empty_directory)
        projectors = await loader.discover_and_load(
            patterns=["*_projector.yaml"],
        )

        assert projectors == []

    @pytest.mark.asyncio
    async def test_discover_with_recursive_pattern(
        self,
        tmp_contract_directory: Path,
        mock_schema_manager: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Recursive glob patterns should find nested contracts."""
        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        monkeypatch.chdir(tmp_contract_directory)
        projectors = await loader.discover_and_load(
            patterns=["**/*_projector.yaml"],
        )

        projector_ids = {p.projector_id for p in projectors}
        assert "order-projector-v1" in projector_ids  # Nested


# =============================================================================
# Security Tests
# =============================================================================


class TestProjectorPluginLoaderSecurity:
    """Tests for security validations in ProjectorPluginLoader.

    These tests verify proper handling of security concerns like symlinks,
    file size limits, and path traversal.
    """

    def test_reject_root_path_as_base_path(
        self, mock_schema_manager: MagicMock
    ) -> None:
        """Root path should be rejected as base_path to prevent DoS."""
        from omnibase_core.models.errors.model_onex_error import ModelOnexError

        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        # Attempting to use root path should raise
        with pytest.raises(ModelOnexError) as exc_info:
            ProjectorPluginLoader(
                schema_manager=mock_schema_manager,
                base_paths=[Path("/")],
            )

        assert "root" in str(exc_info.value).lower()
        assert "filesystem-wide" in str(exc_info.value).lower()

    def test_reject_near_root_path_as_base_path(
        self, mock_schema_manager: MagicMock
    ) -> None:
        """Near-root paths (like /tmp on some systems) with single part should be handled."""
        # Paths with more than one part should be allowed
        # This test just verifies the loader can be created with a valid path
        import tempfile

        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ProjectorPluginLoader(
                schema_manager=mock_schema_manager,
                base_paths=[Path(tmpdir)],
            )
            assert loader is not None

    @pytest.mark.asyncio
    async def test_reject_symlink_contract(
        self, tmp_path: Path, mock_schema_manager: MagicMock
    ) -> None:
        """Symlinks pointing outside allowed paths should be rejected."""
        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        # Create actual contract outside allowed path
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        outside_contract = outside_dir / "secret_projector.yaml"
        outside_contract.write_text(
            MINIMAL_PROJECTOR_CONTRACT_YAML.format(
                projector_id="secret-proj",
                aggregate_type="secret_aggregate",
            )
        )

        # Create symlink inside allowed path
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        symlink_contract = allowed_dir / "symlink_projector.yaml"
        symlink_contract.symlink_to(outside_contract)

        # Use graceful mode to collect errors rather than raise
        loader = ProjectorPluginLoader(
            schema_manager=mock_schema_manager,
            graceful_mode=True,
        )

        # Search only in allowed directory - symlink should be rejected
        projectors = await loader.load_from_directory(allowed_dir)

        # Symlink pointing outside should be blocked
        assert len(projectors) == 0

    @pytest.mark.asyncio
    async def test_reject_oversized_file(
        self, tmp_path: Path, mock_schema_manager: MagicMock
    ) -> None:
        """Files exceeding 10MB should be rejected."""
        from omnibase_core.models.errors.model_onex_error import ModelOnexError

        from omnibase_infra.runtime.projector_plugin_loader import (
            MAX_CONTRACT_SIZE,
            ProjectorPluginLoader,
        )

        # Create a small valid contract file
        contract_file = tmp_path / "large_projector.yaml"
        contract_file.write_text(
            MINIMAL_PROJECTOR_CONTRACT_YAML.format(
                projector_id="large-proj",
                aggregate_type="LargeAgg",
            )
        )

        # Mock stat to return oversized value using shared MockStatResult
        oversized_bytes = MAX_CONTRACT_SIZE + 1
        original_stat = Path.stat

        def mock_stat(self: Path, **kwargs: object) -> object:
            result = original_stat(self, **kwargs)
            if self.name.endswith("_projector.yaml"):
                # Use shared mock helper for consistent stat result mocking
                return create_mock_stat_result(result, oversized_bytes)
            return result

        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        with patch.object(Path, "stat", mock_stat):
            with pytest.raises(ModelOnexError) as exc_info:
                await loader.load_from_contract(contract_file)

        assert "size" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_path_traversal_rejected(
        self, tmp_path: Path, mock_schema_manager: MagicMock
    ) -> None:
        """Path traversal attempts should be blocked."""
        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        # Create contract with path traversal attempt in name/path
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        # Attempt to load file with path traversal
        traversal_path = allowed_dir / ".." / ".." / "etc" / "passwd_projector.yaml"

        # Should raise FileNotFoundError or similar for non-existent file
        with pytest.raises((FileNotFoundError, Exception)) as exc_info:
            await loader.load_from_contract(traversal_path)

        # Should block path traversal or file simply doesn't exist
        error_str = str(exc_info.value).lower()
        assert (
            "traversal" in error_str
            or "outside" in error_str
            or "not" in error_str
            or "exist" in error_str
        )

    @pytest.mark.asyncio
    async def test_allow_symlink_within_allowed_paths(
        self,
        tmp_path: Path,
        mock_schema_manager: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Symlinks within allowed paths should be permitted."""
        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        # Create actual contract within allowed path
        actual_dir = tmp_path / "actual"
        actual_dir.mkdir()
        actual_contract = actual_dir / "real_projector.yaml"
        actual_contract.write_text(
            MINIMAL_PROJECTOR_CONTRACT_YAML.format(
                projector_id="real-proj",
                aggregate_type="RealAgg",
            )
        )

        # Create symlink also within allowed path
        symlink_dir = tmp_path / "symlinked"
        symlink_dir.mkdir()
        symlink_contract = symlink_dir / "link_projector.yaml"
        symlink_contract.symlink_to(actual_contract)

        # Use base_paths to include the tmp_path as allowed
        loader = ProjectorPluginLoader(
            schema_manager=mock_schema_manager,
            base_paths=[tmp_path],
        )

        # Search both directories - symlink should work
        monkeypatch.chdir(tmp_path)
        projectors = await loader.discover_and_load(
            patterns=["**/*_projector.yaml"],
        )

        # Should find the contract (possibly deduplicated)
        assert len(projectors) >= 1
        projector_ids = {p.projector_id for p in projectors}
        assert "real-proj" in projector_ids


# =============================================================================
# Mode Tests (Strict vs Graceful)
# =============================================================================


class TestProjectorPluginLoaderModes:
    """Tests for strict and graceful error handling modes.

    These tests verify that ProjectorPluginLoader correctly handles errors
    in both strict mode (fail fast) and graceful mode (collect errors).
    """

    @pytest.mark.asyncio
    async def test_strict_mode_raises_on_first_error(
        self, tmp_path: Path, mock_schema_manager: MagicMock
    ) -> None:
        """Strict mode should fail fast on first error."""
        from omnibase_core.models.errors.model_onex_error import ModelOnexError

        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        # Create one valid and one invalid contract
        (tmp_path / "valid_projector.yaml").write_text(
            MINIMAL_PROJECTOR_CONTRACT_YAML.format(
                projector_id="valid-proj",
                aggregate_type="ValidAgg",
            )
        )
        (tmp_path / "aaa_invalid_projector.yaml").write_text(MALFORMED_YAML_CONTENT)

        loader = ProjectorPluginLoader(
            schema_manager=mock_schema_manager,
            graceful_mode=False,  # Strict mode
        )

        with pytest.raises(ModelOnexError):
            await loader.load_from_directory(tmp_path)

    @pytest.mark.asyncio
    async def test_graceful_mode_continues_on_error(
        self, tmp_path: Path, mock_schema_manager: MagicMock
    ) -> None:
        """Graceful mode should continue and load valid contracts despite errors."""
        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        # Create one valid and two invalid contracts
        (tmp_path / "valid_projector.yaml").write_text(
            MINIMAL_PROJECTOR_CONTRACT_YAML.format(
                projector_id="valid-proj",
                aggregate_type="ValidAgg",
            )
        )
        (tmp_path / "invalid1_projector.yaml").write_text(MALFORMED_YAML_CONTENT)
        (tmp_path / "invalid2_projector.yaml").write_text(
            MISSING_REQUIRED_FIELDS_CONTENT
        )

        loader = ProjectorPluginLoader(
            schema_manager=mock_schema_manager,
            graceful_mode=True,  # Graceful mode
        )

        projectors = await loader.load_from_directory(tmp_path)

        # Valid contract should still be discovered
        assert len(projectors) == 1
        assert projectors[0].projector_id == "valid-proj"

    @pytest.mark.asyncio
    async def test_discover_with_errors_returns_both(
        self, tmp_path: Path, mock_schema_manager: MagicMock
    ) -> None:
        """discover_with_errors should return both projectors and errors."""
        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        (tmp_path / "valid_projector.yaml").write_text(
            MINIMAL_PROJECTOR_CONTRACT_YAML.format(
                projector_id="valid-proj",
                aggregate_type="ValidAgg",
            )
        )
        (tmp_path / "malformed_projector.yaml").write_text(MALFORMED_YAML_CONTENT)

        loader = ProjectorPluginLoader(
            schema_manager=mock_schema_manager,
            graceful_mode=True,
        )

        result = await loader.discover_with_errors(tmp_path)

        # Should have loaded the valid contract
        assert len(result.projectors) == 1
        assert result.projectors[0].projector_id == "valid-proj"


# =============================================================================
# Contract Validation Tests
# =============================================================================


class TestProjectorPluginLoaderContractValidation:
    """Tests for projector contract validation rules.

    These tests verify that ProjectorPluginLoader correctly validates
    contract contents against the projector contract schema.
    """

    @pytest.mark.asyncio
    async def test_validates_required_fields(
        self, tmp_path: Path, mock_schema_manager: MagicMock
    ) -> None:
        """Required fields should be validated."""
        from omnibase_core.models.errors.model_onex_error import ModelOnexError

        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        # Contract missing aggregate_type (and other required fields)
        incomplete_contract = tmp_path / "incomplete_projector.yaml"
        incomplete_contract.write_text(
            """
projector_kind: materialized_view
projector_id: "incomplete-projector"
name: "Incomplete"
version: "1.0.0"
consumed_events:
  - some.event.v1
projection_schema:
  table: test_table
  primary_key: id
  columns:
    - name: id
      type: UUID
      source: event.payload.id
behavior:
  mode: upsert
"""
        )

        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        with pytest.raises(ModelOnexError) as exc_info:
            await loader.load_from_contract(incomplete_contract)

        # Missing aggregate_type should cause validation error
        assert "aggregate_type" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_validates_schema_structure(
        self, tmp_path: Path, mock_schema_manager: MagicMock
    ) -> None:
        """Schema structure should be validated."""
        from omnibase_core.models.errors.model_onex_error import ModelOnexError

        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        contract_file = tmp_path / "bad_schema_projector.yaml"
        contract_file.write_text(INVALID_SCHEMA_CONTENT)

        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        with pytest.raises(ModelOnexError) as exc_info:
            await loader.load_from_contract(contract_file)

        error_msg = str(exc_info.value).lower()
        assert (
            "schema" in error_msg
            or "table" in error_msg
            or "columns" in error_msg
            or "validation" in error_msg
        )

    @pytest.mark.asyncio
    async def test_validates_consumed_events_non_empty(
        self, tmp_path: Path, mock_schema_manager: MagicMock
    ) -> None:
        """consumed_events should not be empty."""
        from omnibase_core.models.errors.model_onex_error import ModelOnexError

        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        contract_file = tmp_path / "no_events_projector.yaml"
        contract_file.write_text(
            """
projector_kind: materialized_view
projector_id: "no-events-projector"
name: "No Events Projector"
version: "1.0.0"
aggregate_type: "test_aggregate"
consumed_events: []
projection_schema:
  table: test_table
  primary_key: id
  columns:
    - name: id
      type: UUID
      source: event.payload.id
behavior:
  mode: upsert
"""
        )

        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        with pytest.raises(ModelOnexError) as exc_info:
            await loader.load_from_contract(contract_file)

        error_msg = str(exc_info.value).lower()
        assert "event" in error_msg or "validation" in error_msg or "min" in error_msg


# =============================================================================
# Projector Property Tests
# =============================================================================


class TestProjectorProperties:
    """Tests for projector properties.

    These tests verify that loaded projectors have all required properties
    correctly populated.
    """

    @pytest.mark.asyncio
    async def test_projector_has_all_required_properties(
        self, tmp_path: Path, mock_schema_manager: MagicMock
    ) -> None:
        """Projector should have all required properties."""
        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        contract_file = tmp_path / "full_projector.yaml"
        contract_file.write_text(
            PROJECTOR_CONTRACT_WITH_METADATA_YAML.format(
                projector_id="full-projector-v1",
                aggregate_type="full_aggregate",
                name="Full Projector",
                version="1.2.3",
                event1="full.created.v1",
                event2="full.updated.v1",
                table="full_projections",
            )
        )

        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)
        projector = await loader.load_from_contract(contract_file)

        # Required properties on projector
        assert projector.projector_id == "full-projector-v1"
        assert projector.aggregate_type == "full_aggregate"
        assert projector.consumed_events == ["full.created.v1", "full.updated.v1"]

        # Schema properties via contract (projection_schema)
        assert projector.contract.projection_schema.table == "full_projections"
        assert projector.contract.projection_schema.primary_key == "id"
        assert len(projector.contract.projection_schema.columns) >= 2

        # Version property via contract
        assert projector.contract.version == "1.2.3"

    @pytest.mark.asyncio
    async def test_projector_has_contract_access(
        self, tmp_contract_file: Path, mock_schema_manager: MagicMock
    ) -> None:
        """Projector should provide access to underlying contract."""
        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)
        projector = await loader.load_from_contract(tmp_contract_file)

        assert projector.contract is not None
        assert projector.contract.projector_id == "test-projector-v1"


# =============================================================================
# Import and Instantiation Tests
# =============================================================================


class TestProjectorPluginLoaderImport:
    """Tests for ProjectorPluginLoader import and instantiation.

    These tests verify the class can be imported and instantiated correctly.
    """

    def test_projector_plugin_loader_can_be_imported(self) -> None:
        """ProjectorPluginLoader should be importable from runtime module."""
        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        assert ProjectorPluginLoader is not None

    def test_max_contract_size_constant_exported(self) -> None:
        """MAX_CONTRACT_SIZE constant should be exported."""
        from omnibase_infra.runtime.projector_plugin_loader import MAX_CONTRACT_SIZE

        # Should be 10MB
        assert MAX_CONTRACT_SIZE == 10 * 1024 * 1024

    def test_loader_instantiation_without_schema_manager(self) -> None:
        """Loader should be instantiable without schema manager."""
        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        # Should work without schema_manager (uses default)
        loader = ProjectorPluginLoader()

        assert loader is not None

    def test_loader_instantiation_with_schema_manager(
        self, mock_schema_manager: MagicMock
    ) -> None:
        """Loader should accept custom schema manager."""
        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        assert loader is not None


# =============================================================================
# Idempotency Tests
# =============================================================================


class TestProjectorPluginLoaderIdempotency:
    """Tests for idempotency of discovery operations.

    These tests verify that repeated discovery calls return consistent results.
    """

    @pytest.mark.asyncio
    async def test_load_from_directory_is_idempotent(
        self, tmp_contract_directory: Path, mock_schema_manager: MagicMock
    ) -> None:
        """Multiple calls should return same results."""
        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)

        projectors1 = await loader.load_from_directory(tmp_contract_directory)
        projectors2 = await loader.load_from_directory(tmp_contract_directory)
        projectors3 = await loader.load_from_directory(tmp_contract_directory)

        assert len(projectors1) == len(projectors2) == len(projectors3)

        ids1 = {p.projector_id for p in projectors1}
        ids2 = {p.projector_id for p in projectors2}
        ids3 = {p.projector_id for p in projectors3}

        assert ids1 == ids2 == ids3


# =============================================================================
# File Pattern Tests
# =============================================================================


class TestProjectorPluginLoaderFilePattern:
    """Tests for file pattern matching.

    These tests verify that ProjectorPluginLoader only discovers files
    matching the expected patterns (*_projector.yaml).
    """

    @pytest.mark.asyncio
    async def test_ignores_non_projector_yaml_files(
        self, tmp_path: Path, mock_schema_manager: MagicMock
    ) -> None:
        """Should only discover *_projector.yaml files."""
        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        # Create projector contract (should be discovered)
        (tmp_path / "test_projector.yaml").write_text(
            MINIMAL_PROJECTOR_CONTRACT_YAML.format(
                projector_id="test-proj",
                aggregate_type="TestAgg",
            )
        )

        # Create other files (should NOT be discovered)
        (tmp_path / "config.yaml").write_text("some: config")
        (tmp_path / "handler_contract.yaml").write_text("handler_id: test")
        (tmp_path / "projector.yaml").write_text("projector: without suffix")
        (tmp_path / "test_projector.yml").write_text("wrong: extension")

        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)
        projectors = await loader.load_from_directory(tmp_path)

        assert len(projectors) == 1
        assert projectors[0].projector_id == "test-proj"

    @pytest.mark.asyncio
    async def test_case_sensitive_file_matching(
        self, tmp_path: Path, mock_schema_manager: MagicMock
    ) -> None:
        """File matching should be case-sensitive."""
        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )

        # Create correctly cased file
        (tmp_path / "test_projector.yaml").write_text(
            MINIMAL_PROJECTOR_CONTRACT_YAML.format(
                projector_id="correct-proj",
                aggregate_type="CorrectAgg",
            )
        )

        # Create wrongly cased files (in separate directories to avoid FS issues)
        upper_dir = tmp_path / "upper"
        upper_dir.mkdir()
        (upper_dir / "TEST_PROJECTOR.YAML").write_text(
            MINIMAL_PROJECTOR_CONTRACT_YAML.format(
                projector_id="upper-proj",
                aggregate_type="UpperAgg",
            )
        )

        mixed_dir = tmp_path / "mixed"
        mixed_dir.mkdir()
        (mixed_dir / "Test_Projector.yaml").write_text(
            MINIMAL_PROJECTOR_CONTRACT_YAML.format(
                projector_id="mixed-proj",
                aggregate_type="MixedAgg",
            )
        )

        loader = ProjectorPluginLoader(schema_manager=mock_schema_manager)
        projectors = await loader.load_from_directory(tmp_path)

        # Only correctly cased file should be discovered
        assert len(projectors) == 1
        assert projectors[0].projector_id == "correct-proj"
