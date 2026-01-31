# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for contract publisher sources."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.services.contract_publisher.sources import (
    ModelDiscoveredContract,
    SourceContractComposite,
    SourceContractFilesystem,
    SourceContractPackage,
)


@pytest.fixture
def valid_contract_yaml() -> str:
    """Return valid contract YAML for testing."""
    return """handler_id: test.handler
name: Test Handler
contract_version:
  major: 1
  minor: 0
  patch: 0
descriptor:
  node_archetype: compute
input_model: str
output_model: str
"""


@pytest.fixture
def valid_contract_yaml_v2() -> str:
    """Return another valid contract YAML with different handler_id."""
    return """handler_id: test.handler.v2
name: Test Handler V2
contract_version:
  major: 2
  minor: 0
  patch: 0
descriptor:
  node_archetype: effect
input_model: str
output_model: str
"""


@pytest.fixture
def valid_contract_yaml_alt_content() -> str:
    """Return contract YAML with same handler_id but different content."""
    return """handler_id: test.handler
name: Test Handler Updated
contract_version:
  major: 1
  minor: 1
  patch: 0
descriptor:
  node_archetype: compute
input_model: str
output_model: str
"""


class TestSourceContractFilesystem:
    """Tests for SourceContractFilesystem."""

    @pytest.mark.asyncio
    async def test_filesystem_discover_finds_contracts(
        self,
        valid_contract_yaml: str,
    ) -> None:
        """Test that filesystem source discovers contract.yaml files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create nested contract structure
            handler_dir = root / "handlers" / "test"
            handler_dir.mkdir(parents=True)
            contract_file = handler_dir / "contract.yaml"
            contract_file.write_text(valid_contract_yaml)

            source = SourceContractFilesystem(root)
            contracts = await source.discover_contracts()

            assert len(contracts) == 1
            assert contracts[0].origin == "filesystem"
            assert "contract.yaml" in str(contracts[0].ref)
            assert contracts[0].text == valid_contract_yaml

    @pytest.mark.asyncio
    async def test_filesystem_discover_multiple_contracts(
        self,
        valid_contract_yaml: str,
        valid_contract_yaml_v2: str,
    ) -> None:
        """Test discovery of multiple contracts in nested directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create first handler
            handler1_dir = root / "handlers" / "foo"
            handler1_dir.mkdir(parents=True)
            (handler1_dir / "contract.yaml").write_text(valid_contract_yaml)

            # Create second handler
            handler2_dir = root / "handlers" / "bar"
            handler2_dir.mkdir(parents=True)
            (handler2_dir / "contract.yaml").write_text(valid_contract_yaml_v2)

            source = SourceContractFilesystem(root)
            contracts = await source.discover_contracts()

            assert len(contracts) == 2
            handler_ids = set()
            for contract in contracts:
                assert contract.origin == "filesystem"
                handler_ids.add(contract.text.split("\n")[0].split(": ")[1])

            assert "test.handler" in handler_ids
            assert "test.handler.v2" in handler_ids

    @pytest.mark.asyncio
    async def test_filesystem_discover_empty_dir(self) -> None:
        """Test that empty directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = SourceContractFilesystem(root)
            contracts = await source.discover_contracts()

            assert len(contracts) == 0

    @pytest.mark.asyncio
    async def test_filesystem_discover_nonexistent_dir(self) -> None:
        """Test that nonexistent directory returns empty list."""
        source = SourceContractFilesystem(Path("/nonexistent/path/to/contracts"))
        contracts = await source.discover_contracts()

        assert len(contracts) == 0

    @pytest.mark.asyncio
    async def test_filesystem_discover_not_directory(self) -> None:
        """Test that file path returns empty list."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmpfile:
            source = SourceContractFilesystem(Path(tmpfile.name))
            contracts = await source.discover_contracts()

            assert len(contracts) == 0

    def test_filesystem_source_properties(self) -> None:
        """Test source_type and source_description properties."""
        root = Path("/app/contracts")
        source = SourceContractFilesystem(root)

        assert source.source_type == "filesystem"
        assert "filesystem:" in source.source_description
        assert "/app/contracts" in source.source_description

    def test_filesystem_root_property(self) -> None:
        """Test root property returns the configured root path."""
        root = Path("/app/contracts")
        source = SourceContractFilesystem(root)

        assert source.root == root

    @pytest.mark.asyncio
    async def test_filesystem_ignores_non_contract_files(
        self,
        valid_contract_yaml: str,
    ) -> None:
        """Test that source only discovers contract.yaml, not other YAML files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            handler_dir = root / "handlers" / "test"
            handler_dir.mkdir(parents=True)

            # Create contract.yaml
            (handler_dir / "contract.yaml").write_text(valid_contract_yaml)

            # Create other YAML files that should be ignored
            (handler_dir / "config.yaml").write_text("key: value")
            (handler_dir / "handler_contract.yaml").write_text("different: format")

            source = SourceContractFilesystem(root)
            contracts = await source.discover_contracts()

            # Should only find contract.yaml
            assert len(contracts) == 1
            assert "contract.yaml" in str(contracts[0].ref)


class TestSourceContractPackage:
    """Tests for SourceContractPackage."""

    @pytest.mark.asyncio
    async def test_package_discover_module_not_found(self) -> None:
        """Test that missing module returns empty list."""
        source = SourceContractPackage("nonexistent.module.that.does.not.exist")
        contracts = await source.discover_contracts()

        assert len(contracts) == 0

    def test_package_source_properties(self) -> None:
        """Test source_type and source_description properties."""
        source = SourceContractPackage("myapp.contracts")

        assert source.source_type == "package"
        assert "package:" in source.source_description
        assert "myapp.contracts" in source.source_description

    def test_package_module_property(self) -> None:
        """Test package_module property returns the configured module name."""
        source = SourceContractPackage("myapp.contracts")

        assert source.package_module == "myapp.contracts"

    @pytest.mark.asyncio
    async def test_package_discover_with_mock_resources(self) -> None:
        """Test package discovery with mocked resources."""
        # Create mock Traversable for resources.files()
        mock_contract_file = MagicMock()
        mock_contract_file.name = "contract.yaml"
        mock_contract_file.is_file.return_value = True
        mock_contract_file.is_dir.return_value = False
        mock_contract_file.read_text.return_value = "handler_id: test.mock\n"

        mock_handler_dir = MagicMock()
        mock_handler_dir.name = "handlers"
        mock_handler_dir.is_file.return_value = False
        mock_handler_dir.is_dir.return_value = True
        mock_handler_dir.iterdir.return_value = [mock_contract_file]

        mock_root = MagicMock()
        mock_root.is_dir.return_value = True
        mock_root.iterdir.return_value = [mock_handler_dir]

        with patch(
            "omnibase_infra.services.contract_publisher.sources.source_package.resources.files"
        ) as mock_files:
            mock_files.return_value = mock_root

            source = SourceContractPackage("myapp.contracts")
            contracts = await source.discover_contracts()

            assert len(contracts) == 1
            assert contracts[0].origin == "package"
            assert "myapp.contracts" in str(contracts[0].ref)


class TestSourceContractComposite:
    """Tests for SourceContractComposite."""

    @pytest.mark.asyncio
    async def test_composite_merge_no_conflicts(
        self,
        valid_contract_yaml: str,
        valid_contract_yaml_v2: str,
    ) -> None:
        """Test that different handler_ids merge cleanly."""
        # Create mock filesystem source
        mock_filesystem = MagicMock()
        mock_filesystem.root = Path("/app/contracts")
        mock_filesystem.discover_contracts = AsyncMock(
            return_value=[
                ModelDiscoveredContract(
                    origin="filesystem",
                    ref=Path("/app/contracts/foo/contract.yaml"),
                    text=valid_contract_yaml,
                )
            ]
        )

        # Create mock package source
        mock_package = MagicMock()
        mock_package.package_module = "myapp.contracts"
        mock_package.discover_contracts = AsyncMock(
            return_value=[
                ModelDiscoveredContract(
                    origin="package",
                    ref="myapp.contracts:bar/contract.yaml",
                    text=valid_contract_yaml_v2,
                )
            ]
        )

        composite = SourceContractComposite(mock_filesystem, mock_package)
        contracts = await composite.discover_contracts()

        # Should have both contracts
        assert len(contracts) == 2

        # No merge errors
        errors = composite.get_merge_errors()
        assert len(errors) == 0

        # No deduplication
        assert composite.get_dedup_count() == 0

    @pytest.mark.asyncio
    async def test_composite_merge_dedup_same_hash(
        self,
        valid_contract_yaml: str,
    ) -> None:
        """Test that same handler_id with same hash dedups silently."""
        # Create mock filesystem source
        mock_filesystem = MagicMock()
        mock_filesystem.root = Path("/app/contracts")
        mock_filesystem.discover_contracts = AsyncMock(
            return_value=[
                ModelDiscoveredContract(
                    origin="filesystem",
                    ref=Path("/app/contracts/foo/contract.yaml"),
                    text=valid_contract_yaml,
                )
            ]
        )

        # Create mock package source with SAME content
        mock_package = MagicMock()
        mock_package.package_module = "myapp.contracts"
        mock_package.discover_contracts = AsyncMock(
            return_value=[
                ModelDiscoveredContract(
                    origin="package",
                    ref="myapp.contracts:foo/contract.yaml",
                    text=valid_contract_yaml,  # Same content
                )
            ]
        )

        composite = SourceContractComposite(mock_filesystem, mock_package)
        contracts = await composite.discover_contracts()

        # Should only have one contract (deduplicated)
        assert len(contracts) == 1

        # No merge errors
        errors = composite.get_merge_errors()
        assert len(errors) == 0

        # Should have dedup count of 1
        assert composite.get_dedup_count() == 1

    @pytest.mark.asyncio
    async def test_composite_merge_conflict_different_hash(
        self,
        valid_contract_yaml: str,
        valid_contract_yaml_alt_content: str,
    ) -> None:
        """Test that same handler_id with different hash creates conflict error."""
        # Create mock filesystem source
        mock_filesystem = MagicMock()
        mock_filesystem.root = Path("/app/contracts")
        mock_filesystem.discover_contracts = AsyncMock(
            return_value=[
                ModelDiscoveredContract(
                    origin="filesystem",
                    ref=Path("/app/contracts/foo/contract.yaml"),
                    text=valid_contract_yaml,
                )
            ]
        )

        # Create mock package source with DIFFERENT content for same handler_id
        mock_package = MagicMock()
        mock_package.package_module = "myapp.contracts"
        mock_package.discover_contracts = AsyncMock(
            return_value=[
                ModelDiscoveredContract(
                    origin="package",
                    ref="myapp.contracts:foo/contract.yaml",
                    text=valid_contract_yaml_alt_content,  # Different content, same handler_id
                )
            ]
        )

        composite = SourceContractComposite(mock_filesystem, mock_package)
        contracts = await composite.discover_contracts()

        # Should only have first contract (conflict excluded)
        assert len(contracts) == 1
        assert contracts[0].origin == "filesystem"

        # Should have merge error
        errors = composite.get_merge_errors()
        assert len(errors) == 1
        assert errors[0].error_type == "duplicate_conflict"
        assert "test.handler" in errors[0].message

    @pytest.mark.asyncio
    async def test_composite_get_dedup_count(
        self,
        valid_contract_yaml: str,
    ) -> None:
        """Test that get_dedup_count tracks deduplication correctly."""
        # Create source with same content in both
        mock_filesystem = MagicMock()
        mock_filesystem.root = Path("/app/contracts")
        mock_filesystem.discover_contracts = AsyncMock(
            return_value=[
                ModelDiscoveredContract(
                    origin="filesystem",
                    ref=Path("/app/contracts/foo/contract.yaml"),
                    text=valid_contract_yaml,
                )
            ]
        )

        mock_package = MagicMock()
        mock_package.package_module = "myapp.contracts"
        mock_package.discover_contracts = AsyncMock(
            return_value=[
                ModelDiscoveredContract(
                    origin="package",
                    ref="myapp.contracts:foo/contract.yaml",
                    text=valid_contract_yaml,
                )
            ]
        )

        composite = SourceContractComposite(mock_filesystem, mock_package)

        # Before discovery
        assert composite.get_dedup_count() == 0

        # After discovery
        await composite.discover_contracts()
        assert composite.get_dedup_count() == 1

    @pytest.mark.asyncio
    async def test_composite_get_merge_errors(
        self,
        valid_contract_yaml: str,
        valid_contract_yaml_alt_content: str,
    ) -> None:
        """Test that get_merge_errors retrieves conflict errors."""
        mock_filesystem = MagicMock()
        mock_filesystem.root = Path("/app/contracts")
        mock_filesystem.discover_contracts = AsyncMock(
            return_value=[
                ModelDiscoveredContract(
                    origin="filesystem",
                    ref=Path("/app/contracts/foo/contract.yaml"),
                    text=valid_contract_yaml,
                )
            ]
        )

        mock_package = MagicMock()
        mock_package.package_module = "myapp.contracts"
        mock_package.discover_contracts = AsyncMock(
            return_value=[
                ModelDiscoveredContract(
                    origin="package",
                    ref="myapp.contracts:foo/contract.yaml",
                    text=valid_contract_yaml_alt_content,
                )
            ]
        )

        composite = SourceContractComposite(mock_filesystem, mock_package)

        # Before discovery
        assert len(composite.get_merge_errors()) == 0

        # After discovery with conflict
        await composite.discover_contracts()
        errors = composite.get_merge_errors()
        assert len(errors) == 1
        assert errors[0].error_type == "duplicate_conflict"

    def test_composite_requires_at_least_one_source(self) -> None:
        """Test that composite requires at least one source."""
        with pytest.raises(ValueError) as exc_info:
            SourceContractComposite(None, None)

        assert "at least one source" in str(exc_info.value)

    def test_composite_filesystem_only(self) -> None:
        """Test composite with only filesystem source."""
        mock_filesystem = MagicMock()
        mock_filesystem.root = Path("/app/contracts")

        composite = SourceContractComposite(mock_filesystem, None)

        assert composite.filesystem_source is mock_filesystem
        assert composite.package_source is None

    def test_composite_package_only(self) -> None:
        """Test composite with only package source."""
        mock_package = MagicMock()
        mock_package.package_module = "myapp.contracts"

        composite = SourceContractComposite(None, mock_package)

        assert composite.filesystem_source is None
        assert composite.package_source is mock_package

    @pytest.mark.asyncio
    async def test_composite_handler_id_extraction(
        self,
        valid_contract_yaml: str,
    ) -> None:
        """Test that composite extracts handler_id from YAML."""
        mock_filesystem = MagicMock()
        mock_filesystem.root = Path("/app/contracts")
        mock_filesystem.discover_contracts = AsyncMock(
            return_value=[
                ModelDiscoveredContract(
                    origin="filesystem",
                    ref=Path("/app/contracts/foo/contract.yaml"),
                    text=valid_contract_yaml,
                )
            ]
        )

        composite = SourceContractComposite(mock_filesystem, None)
        contracts = await composite.discover_contracts()

        assert len(contracts) == 1
        # After composite processing, handler_id should be extracted
        assert contracts[0].handler_id == "test.handler"

    @pytest.mark.asyncio
    async def test_composite_invalid_yaml_passes_through(self) -> None:
        """Test that contracts with invalid YAML pass through (fail validation later)."""
        invalid_yaml = "not: [valid yaml"

        mock_filesystem = MagicMock()
        mock_filesystem.root = Path("/app/contracts")
        mock_filesystem.discover_contracts = AsyncMock(
            return_value=[
                ModelDiscoveredContract(
                    origin="filesystem",
                    ref=Path("/app/contracts/bad/contract.yaml"),
                    text=invalid_yaml,
                )
            ]
        )

        composite = SourceContractComposite(mock_filesystem, None)
        contracts = await composite.discover_contracts()

        # Contract should pass through (validation happens later)
        assert len(contracts) == 1
        assert contracts[0].handler_id is None  # Could not extract

    def test_composite_source_properties(self) -> None:
        """Test source_type and source_description properties."""
        mock_filesystem = MagicMock()
        mock_filesystem.root = Path("/app/contracts")

        mock_package = MagicMock()
        mock_package.package_module = "myapp.contracts"

        composite = SourceContractComposite(mock_filesystem, mock_package)

        assert composite.source_type == "composite"
        assert "composite:" in composite.source_description
        assert "filesystem=" in composite.source_description
        assert "package=" in composite.source_description

    @pytest.mark.asyncio
    async def test_composite_clears_state_on_each_discovery(
        self,
        valid_contract_yaml: str,
        valid_contract_yaml_alt_content: str,
    ) -> None:
        """Test that composite clears state on each discover_contracts call."""
        mock_filesystem = MagicMock()
        mock_filesystem.root = Path("/app/contracts")
        mock_filesystem.discover_contracts = AsyncMock(
            return_value=[
                ModelDiscoveredContract(
                    origin="filesystem",
                    ref=Path("/app/contracts/foo/contract.yaml"),
                    text=valid_contract_yaml,
                )
            ]
        )

        mock_package = MagicMock()
        mock_package.package_module = "myapp.contracts"
        mock_package.discover_contracts = AsyncMock(
            return_value=[
                ModelDiscoveredContract(
                    origin="package",
                    ref="myapp.contracts:foo/contract.yaml",
                    text=valid_contract_yaml_alt_content,  # Conflict
                )
            ]
        )

        composite = SourceContractComposite(mock_filesystem, mock_package)

        # First discovery - conflict
        await composite.discover_contracts()
        assert len(composite.get_merge_errors()) == 1

        # Update package source to return same content (no conflict)
        mock_package.discover_contracts = AsyncMock(
            return_value=[
                ModelDiscoveredContract(
                    origin="package",
                    ref="myapp.contracts:foo/contract.yaml",
                    text=valid_contract_yaml,  # Same content - will dedup
                )
            ]
        )

        # Second discovery - should clear previous errors
        await composite.discover_contracts()
        assert len(composite.get_merge_errors()) == 0
        assert composite.get_dedup_count() == 1


class TestModelDiscoveredContract:
    """Tests for ModelDiscoveredContract methods."""

    def test_with_content_hash(self, valid_contract_yaml: str) -> None:
        """Test with_content_hash computes SHA-256 hash."""
        contract = ModelDiscoveredContract(
            origin="filesystem",
            ref=Path("/test/contract.yaml"),
            text=valid_contract_yaml,
        )

        hashed = contract.with_content_hash()

        assert hashed.content_hash is not None
        assert len(hashed.content_hash) == 64  # SHA-256 hex digest length
        # Original should be unchanged (frozen model)
        assert contract.content_hash is None

    def test_with_parsed_data(self, valid_contract_yaml: str) -> None:
        """Test with_parsed_data sets handler_id."""
        contract = ModelDiscoveredContract(
            origin="filesystem",
            ref=Path("/test/contract.yaml"),
            text=valid_contract_yaml,
        )

        parsed = contract.with_parsed_data(handler_id="test.handler")

        assert parsed.handler_id == "test.handler"
        # Original should be unchanged (frozen model)
        assert contract.handler_id is None

    def test_sort_key(self, valid_contract_yaml: str) -> None:
        """Test sort_key returns correct tuple."""
        contract = ModelDiscoveredContract(
            origin="filesystem",
            ref=Path("/test/contract.yaml"),
            text=valid_contract_yaml,
            handler_id="test.handler",
        )

        key = contract.sort_key()

        assert key == ("test.handler", "filesystem", "/test/contract.yaml")

    def test_sort_key_no_handler_id(self, valid_contract_yaml: str) -> None:
        """Test sort_key with no handler_id uses empty string."""
        contract = ModelDiscoveredContract(
            origin="filesystem",
            ref=Path("/test/contract.yaml"),
            text=valid_contract_yaml,
        )

        key = contract.sort_key()

        assert key[0] == ""  # Empty string for missing handler_id

    def test_compute_content_hash_static(self) -> None:
        """Test static compute_content_hash method."""
        text = "handler_id: test\n"
        hash1 = ModelDiscoveredContract.compute_content_hash(text)
        hash2 = ModelDiscoveredContract.compute_content_hash(text)

        assert hash1 == hash2
        assert len(hash1) == 64

        # Different content should have different hash
        hash3 = ModelDiscoveredContract.compute_content_hash("different content")
        assert hash3 != hash1

    def test_model_is_frozen(self, valid_contract_yaml: str) -> None:
        """Test that model is frozen (immutable)."""
        contract = ModelDiscoveredContract(
            origin="filesystem",
            ref=Path("/test/contract.yaml"),
            text=valid_contract_yaml,
        )

        with pytest.raises(Exception):  # ValidationError for frozen model
            contract.handler_id = "modified"  # type: ignore[misc]
