# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ServiceContractPublisher."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.services.contract_publisher import (
    ContractPublishingInfraError,
    ModelContractPublisherConfig,
    NoContractsFoundError,
    ServiceContractPublisher,
)
from omnibase_infra.services.contract_publisher.models import (
    ModelContractError,
    ModelInfraError,
)
from omnibase_infra.services.contract_publisher.sources import (
    ModelDiscoveredContract,
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
    """Return another valid contract YAML for testing."""
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
def invalid_yaml() -> str:
    """Return invalid YAML for testing."""
    return """handler_id: [unclosed bracket
name: This YAML is broken
"""


@pytest.fixture
def invalid_schema_yaml() -> str:
    """Return YAML that fails schema validation."""
    return """handler_id: test.invalid
name: Missing required fields
"""


@pytest.fixture
def mock_publisher() -> AsyncMock:
    """Return mock event bus publisher."""
    publisher = AsyncMock()
    publisher.publish = AsyncMock(return_value=None)
    return publisher


@pytest.fixture
def mock_source(valid_contract_yaml: str) -> MagicMock:
    """Return mock contract source with valid contracts."""
    source = MagicMock()
    source.source_type = "test"
    source.source_description = "test source"
    source.discover_contracts = AsyncMock(
        return_value=[
            ModelDiscoveredContract(
                origin="filesystem",
                ref=Path("/test/handlers/foo/contract.yaml"),
                text=valid_contract_yaml,
            )
        ]
    )
    return source


@pytest.fixture
def filesystem_config() -> ModelContractPublisherConfig:
    """Return filesystem config for testing."""
    return ModelContractPublisherConfig(
        mode="filesystem",
        filesystem_root=Path("/test"),
    )


class TestServiceContractPublisherPublishAll:
    """Tests for ServiceContractPublisher.publish_all()."""

    @pytest.mark.asyncio
    async def test_publish_all_success(
        self,
        mock_publisher: AsyncMock,
        mock_source: MagicMock,
        filesystem_config: ModelContractPublisherConfig,
    ) -> None:
        """Test successful publishing of valid contracts."""
        service = ServiceContractPublisher(
            mock_publisher, mock_source, filesystem_config
        )
        result = await service.publish_all()

        # Verify result
        assert len(result.published) == 1
        assert "test.handler" in result.published
        assert not result.has_contract_errors
        assert not result.has_infra_errors

        # Verify publisher was called
        mock_publisher.publish.assert_called_once()
        call_kwargs = mock_publisher.publish.call_args.kwargs
        assert "topic" in call_kwargs
        assert call_kwargs["topic"] == "dev.onex.evt.contract-registered.v1"

    @pytest.mark.asyncio
    async def test_publish_all_multiple_contracts(
        self,
        mock_publisher: AsyncMock,
        filesystem_config: ModelContractPublisherConfig,
        valid_contract_yaml: str,
        valid_contract_yaml_v2: str,
    ) -> None:
        """Test publishing multiple valid contracts."""
        source = MagicMock()
        source.source_type = "test"
        source.source_description = "test source"
        source.discover_contracts = AsyncMock(
            return_value=[
                ModelDiscoveredContract(
                    origin="filesystem",
                    ref=Path("/test/handlers/foo/contract.yaml"),
                    text=valid_contract_yaml,
                ),
                ModelDiscoveredContract(
                    origin="filesystem",
                    ref=Path("/test/handlers/bar/contract.yaml"),
                    text=valid_contract_yaml_v2,
                ),
            ]
        )

        service = ServiceContractPublisher(mock_publisher, source, filesystem_config)
        result = await service.publish_all()

        assert len(result.published) == 2
        assert "test.handler" in result.published
        assert "test.handler.v2" in result.published
        assert mock_publisher.publish.call_count == 2

    @pytest.mark.asyncio
    async def test_publish_all_with_validation_errors(
        self,
        mock_publisher: AsyncMock,
        filesystem_config: ModelContractPublisherConfig,
        valid_contract_yaml: str,
        invalid_schema_yaml: str,
    ) -> None:
        """Test publishing with some contracts failing validation."""
        source = MagicMock()
        source.source_type = "test"
        source.source_description = "test source"
        source.discover_contracts = AsyncMock(
            return_value=[
                ModelDiscoveredContract(
                    origin="filesystem",
                    ref=Path("/test/handlers/valid/contract.yaml"),
                    text=valid_contract_yaml,
                ),
                ModelDiscoveredContract(
                    origin="filesystem",
                    ref=Path("/test/handlers/invalid/contract.yaml"),
                    text=invalid_schema_yaml,
                ),
            ]
        )

        service = ServiceContractPublisher(mock_publisher, source, filesystem_config)
        result = await service.publish_all()

        # Valid contract should be published
        assert len(result.published) == 1
        assert "test.handler" in result.published

        # Invalid contract should have error
        assert len(result.contract_errors) == 1
        assert result.contract_errors[0].error_type == "schema_validation"
        assert "invalid/contract.yaml" in result.contract_errors[0].contract_path

    @pytest.mark.asyncio
    async def test_publish_all_empty_raises_when_not_allowed(
        self,
        mock_publisher: AsyncMock,
    ) -> None:
        """Test that empty contracts raises NoContractsFoundError when not allowed."""
        config = ModelContractPublisherConfig(
            mode="filesystem",
            filesystem_root=Path("/test"),
            allow_zero_contracts=False,
        )

        source = MagicMock()
        source.source_type = "test"
        source.source_description = "test source"
        source.discover_contracts = AsyncMock(return_value=[])

        service = ServiceContractPublisher(mock_publisher, source, config)

        with pytest.raises(NoContractsFoundError) as exc_info:
            await service.publish_all()

        assert "test source" in exc_info.value.source_description

    @pytest.mark.asyncio
    async def test_publish_all_empty_allowed(
        self,
        mock_publisher: AsyncMock,
    ) -> None:
        """Test that empty contracts returns empty result when allowed."""
        config = ModelContractPublisherConfig(
            mode="filesystem",
            filesystem_root=Path("/test"),
            allow_zero_contracts=True,
        )

        source = MagicMock()
        source.source_type = "test"
        source.source_description = "test source"
        source.discover_contracts = AsyncMock(return_value=[])

        service = ServiceContractPublisher(mock_publisher, source, config)
        result = await service.publish_all()

        assert len(result.published) == 0
        assert result.stats.discovered_count == 0
        # Result should be falsy (no published)
        assert not result

    @pytest.mark.asyncio
    async def test_publish_all_fail_fast_raises(
        self,
        filesystem_config: ModelContractPublisherConfig,
        valid_contract_yaml: str,
    ) -> None:
        """Test that infra error with fail_fast=True raises ContractPublishingInfraError."""
        # Create config with fail_fast=True (default)
        config = ModelContractPublisherConfig(
            mode="filesystem",
            filesystem_root=Path("/test"),
            fail_fast=True,
        )

        # Publisher that fails
        mock_publisher = AsyncMock()
        mock_publisher.publish = AsyncMock(
            side_effect=Exception("Kafka connection failed")
        )

        source = MagicMock()
        source.source_type = "test"
        source.source_description = "test source"
        source.discover_contracts = AsyncMock(
            return_value=[
                ModelDiscoveredContract(
                    origin="filesystem",
                    ref=Path("/test/handlers/foo/contract.yaml"),
                    text=valid_contract_yaml,
                )
            ]
        )

        service = ServiceContractPublisher(mock_publisher, source, config)

        with pytest.raises(ContractPublishingInfraError) as exc_info:
            await service.publish_all()

        assert len(exc_info.value.infra_errors) == 1
        assert "Kafka connection failed" in exc_info.value.infra_errors[0].message

    @pytest.mark.asyncio
    async def test_publish_all_fail_fast_false_continues(
        self,
        valid_contract_yaml: str,
        valid_contract_yaml_v2: str,
    ) -> None:
        """Test that infra error is collected when fail_fast=False."""
        config = ModelContractPublisherConfig(
            mode="filesystem",
            filesystem_root=Path("/test"),
            fail_fast=False,
        )

        # Publisher that fails only on first call
        mock_publisher = AsyncMock()
        call_count = 0

        async def fail_first(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("First publish failed")

        mock_publisher.publish = AsyncMock(side_effect=fail_first)

        # Contracts are sorted by (handler_id, origin, ref) before publishing.
        # handler_id is extracted in _discover() before sorting, so:
        # "test.handler" (aaa) comes before "test.handler.v2" (zzz)
        source = MagicMock()
        source.source_type = "test"
        source.source_description = "test source"
        source.discover_contracts = AsyncMock(
            return_value=[
                ModelDiscoveredContract(
                    origin="filesystem",
                    ref=Path("/test/handlers/aaa/contract.yaml"),  # First in sort order
                    text=valid_contract_yaml,
                ),
                ModelDiscoveredContract(
                    origin="filesystem",
                    ref=Path(
                        "/test/handlers/zzz/contract.yaml"
                    ),  # Second in sort order
                    text=valid_contract_yaml_v2,
                ),
            ]
        )

        service = ServiceContractPublisher(mock_publisher, source, config)
        result = await service.publish_all()

        # First contract (aaa) fails, second (zzz) succeeds
        assert len(result.published) == 1
        assert "test.handler.v2" in result.published  # zzz has test.handler.v2

        # Should have collected infra error for first contract (aaa)
        assert len(result.infra_errors) == 1
        assert "First publish failed" in result.infra_errors[0].message
        assert "test.handler" in result.infra_errors[0].message  # aaa has test.handler


class TestServiceContractPublisherValidateContract:
    """Tests for ServiceContractPublisher._validate_contract()."""

    @pytest.mark.asyncio
    async def test_validate_contract_valid(
        self,
        mock_publisher: AsyncMock,
        mock_source: MagicMock,
        filesystem_config: ModelContractPublisherConfig,
        valid_contract_yaml: str,
    ) -> None:
        """Test validation of valid contract YAML."""
        service = ServiceContractPublisher(
            mock_publisher, mock_source, filesystem_config
        )

        contract = ModelDiscoveredContract(
            origin="filesystem",
            ref=Path("/test/contract.yaml"),
            text=valid_contract_yaml,
        )

        parsed, error = service._validate_contract(contract)

        assert parsed is not None
        assert error is None
        assert parsed.handler_id == "test.handler"

    @pytest.mark.asyncio
    async def test_validate_contract_invalid_yaml(
        self,
        mock_publisher: AsyncMock,
        mock_source: MagicMock,
        filesystem_config: ModelContractPublisherConfig,
        invalid_yaml: str,
    ) -> None:
        """Test validation returns yaml_parse error for invalid YAML."""
        service = ServiceContractPublisher(
            mock_publisher, mock_source, filesystem_config
        )

        contract = ModelDiscoveredContract(
            origin="filesystem",
            ref=Path("/test/bad.yaml"),
            text=invalid_yaml,
        )

        parsed, error = service._validate_contract(contract)

        assert parsed is None
        assert error is not None
        assert error.error_type == "yaml_parse"
        assert "Invalid YAML" in error.message

    @pytest.mark.asyncio
    async def test_validate_contract_schema_error(
        self,
        mock_publisher: AsyncMock,
        mock_source: MagicMock,
        filesystem_config: ModelContractPublisherConfig,
        invalid_schema_yaml: str,
    ) -> None:
        """Test validation returns schema_validation error for schema mismatch."""
        service = ServiceContractPublisher(
            mock_publisher, mock_source, filesystem_config
        )

        contract = ModelDiscoveredContract(
            origin="filesystem",
            ref=Path("/test/invalid_schema.yaml"),
            text=invalid_schema_yaml,
        )

        parsed, error = service._validate_contract(contract)

        assert parsed is None
        assert error is not None
        assert error.error_type == "schema_validation"
        assert error.handler_id == "test.invalid"

    @pytest.mark.asyncio
    async def test_validate_contract_not_dict(
        self,
        mock_publisher: AsyncMock,
        mock_source: MagicMock,
        filesystem_config: ModelContractPublisherConfig,
    ) -> None:
        """Test validation returns error when YAML is not a dictionary."""
        service = ServiceContractPublisher(
            mock_publisher, mock_source, filesystem_config
        )

        contract = ModelDiscoveredContract(
            origin="filesystem",
            ref=Path("/test/list.yaml"),
            text="- item1\n- item2\n",
        )

        parsed, error = service._validate_contract(contract)

        assert parsed is None
        assert error is not None
        assert error.error_type == "yaml_parse"
        assert "must be a dictionary" in error.message


class TestServiceContractPublisherResolveTopic:
    """Tests for ServiceContractPublisher.resolve_topic()."""

    @pytest.mark.asyncio
    async def test_resolve_topic_default_environment(
        self,
        mock_publisher: AsyncMock,
        mock_source: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test topic resolution with default environment."""
        monkeypatch.delenv("ONEX_ENV", raising=False)

        config = ModelContractPublisherConfig(
            mode="filesystem",
            filesystem_root=Path("/test"),
        )
        service = ServiceContractPublisher(mock_publisher, mock_source, config)

        topic = service.resolve_topic("onex.evt.contract-registered.v1")
        assert topic == "dev.onex.evt.contract-registered.v1"

    @pytest.mark.asyncio
    async def test_resolve_topic_custom_environment(
        self,
        mock_publisher: AsyncMock,
        mock_source: MagicMock,
    ) -> None:
        """Test topic resolution with custom environment."""
        config = ModelContractPublisherConfig(
            mode="filesystem",
            filesystem_root=Path("/test"),
            environment="prod",
        )
        service = ServiceContractPublisher(mock_publisher, mock_source, config)

        topic = service.resolve_topic("onex.evt.contract-registered.v1")
        assert topic == "prod.onex.evt.contract-registered.v1"

    @pytest.mark.asyncio
    async def test_resolve_topic_env_var(
        self,
        mock_publisher: AsyncMock,
        mock_source: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test topic resolution with ONEX_ENV environment variable."""
        monkeypatch.setenv("ONEX_ENV", "staging")

        config = ModelContractPublisherConfig(
            mode="filesystem",
            filesystem_root=Path("/test"),
        )
        service = ServiceContractPublisher(mock_publisher, mock_source, config)

        topic = service.resolve_topic("onex.evt.contract-registered.v1")
        assert topic == "staging.onex.evt.contract-registered.v1"


class TestServiceContractPublisherCreateSource:
    """Tests for ServiceContractPublisher._create_source()."""

    def test_create_source_filesystem(self) -> None:
        """Test source creation for filesystem mode."""
        config = ModelContractPublisherConfig(
            mode="filesystem",
            filesystem_root=Path("/test/contracts"),
        )

        source = ServiceContractPublisher._create_source(config)

        assert source.source_type == "filesystem"
        assert "/test/contracts" in source.source_description

    def test_create_source_package(self) -> None:
        """Test source creation for package mode."""
        config = ModelContractPublisherConfig(
            mode="package",
            package_module="myapp.contracts",
        )

        source = ServiceContractPublisher._create_source(config)

        assert source.source_type == "package"
        assert "myapp.contracts" in source.source_description

    def test_create_source_composite(self) -> None:
        """Test source creation for composite mode."""
        config = ModelContractPublisherConfig(
            mode="composite",
            filesystem_root=Path("/test/contracts"),
            package_module="myapp.contracts",
        )

        source = ServiceContractPublisher._create_source(config)

        assert source.source_type == "composite"


class TestServiceContractPublisherStats:
    """Tests for publishing statistics tracking."""

    @pytest.mark.asyncio
    async def test_stats_per_origin_count(
        self,
        mock_publisher: AsyncMock,
        valid_contract_yaml: str,
        valid_contract_yaml_v2: str,
    ) -> None:
        """Test that stats track per-origin counts correctly."""
        config = ModelContractPublisherConfig(
            mode="filesystem",
            filesystem_root=Path("/test"),
        )

        source = MagicMock()
        source.source_type = "test"
        source.source_description = "test source"
        source.discover_contracts = AsyncMock(
            return_value=[
                ModelDiscoveredContract(
                    origin="filesystem",
                    ref=Path("/test/handlers/foo/contract.yaml"),
                    text=valid_contract_yaml,
                ),
                ModelDiscoveredContract(
                    origin="package",
                    ref="myapp.contracts:bar/contract.yaml",
                    text=valid_contract_yaml_v2,
                ),
            ]
        )

        service = ServiceContractPublisher(mock_publisher, source, config)
        result = await service.publish_all()

        assert result.stats.filesystem_count == 1
        assert result.stats.package_count == 1
        assert result.stats.discovered_count == 2

    @pytest.mark.asyncio
    async def test_stats_timing(
        self,
        mock_publisher: AsyncMock,
        mock_source: MagicMock,
        filesystem_config: ModelContractPublisherConfig,
    ) -> None:
        """Test that stats include timing information."""
        service = ServiceContractPublisher(
            mock_publisher, mock_source, filesystem_config
        )
        result = await service.publish_all()

        # All timing fields should be non-negative
        assert result.stats.duration_ms >= 0.0
        assert result.stats.discover_ms >= 0.0
        assert result.stats.validate_ms >= 0.0
        assert result.stats.publish_ms >= 0.0

    @pytest.mark.asyncio
    async def test_stats_environment(
        self,
        mock_publisher: AsyncMock,
        mock_source: MagicMock,
    ) -> None:
        """Test that stats include resolved environment."""
        config = ModelContractPublisherConfig(
            mode="filesystem",
            filesystem_root=Path("/test"),
            environment="production",
        )

        service = ServiceContractPublisher(mock_publisher, mock_source, config)
        result = await service.publish_all()

        assert result.stats.environment == "production"


class TestServiceContractPublisherHandlerIdExtraction:
    """Tests for handler_id extraction before sorting."""

    @pytest.mark.asyncio
    async def test_handler_id_extracted_before_sorting_for_filesystem_source(
        self,
        mock_publisher: AsyncMock,
    ) -> None:
        """Test that handler_id is extracted before sorting for filesystem source.

        This verifies the fix for consistent sorting across all source types.
        Before the fix, filesystem/package sources would sort by ref path since
        handler_id was None. Now handler_id is extracted first, ensuring proper
        deterministic sorting by (handler_id, origin, ref).
        """
        config = ModelContractPublisherConfig(
            mode="filesystem",
            filesystem_root=Path("/test"),
        )

        # Create contracts where handler_id sorting differs from path sorting:
        # - Path order: /test/aaa < /test/zzz
        # - Handler ID order: beta < alpha (reversed!)
        contract_alpha = """handler_id: alpha.handler
name: Alpha Handler
contract_version:
  major: 1
  minor: 0
  patch: 0
descriptor:
  node_archetype: compute
input_model: str
output_model: str
"""
        contract_beta = """handler_id: beta.handler
name: Beta Handler
contract_version:
  major: 1
  minor: 0
  patch: 0
descriptor:
  node_archetype: compute
input_model: str
output_model: str
"""

        source = MagicMock()
        source.source_type = "filesystem"
        source.source_description = "test source"
        source.discover_contracts = AsyncMock(
            return_value=[
                # Path "zzz" has handler_id "alpha" (should come first after sorting)
                ModelDiscoveredContract(
                    origin="filesystem",
                    ref=Path("/test/handlers/zzz/contract.yaml"),
                    text=contract_alpha,
                ),
                # Path "aaa" has handler_id "beta" (should come second after sorting)
                ModelDiscoveredContract(
                    origin="filesystem",
                    ref=Path("/test/handlers/aaa/contract.yaml"),
                    text=contract_beta,
                ),
            ]
        )

        service = ServiceContractPublisher(mock_publisher, source, config)
        result = await service.publish_all()

        # Verify both contracts published
        assert len(result.published) == 2

        # Verify sort order is by handler_id, NOT by path
        # alpha.handler should be published first (comes before beta alphabetically)
        assert result.published[0] == "alpha.handler"
        assert result.published[1] == "beta.handler"

    @pytest.mark.asyncio
    async def test_handler_id_extraction_skips_already_extracted(
        self,
        mock_publisher: AsyncMock,
    ) -> None:
        """Test that handler_id extraction is skipped if already present.

        This ensures we don't double-parse YAML for composite sources that
        already extract handler_id internally.
        """
        config = ModelContractPublisherConfig(
            mode="filesystem",
            filesystem_root=Path("/test"),
        )

        contract_yaml = """handler_id: pre.extracted.handler
name: Pre-extracted Handler
contract_version:
  major: 1
  minor: 0
  patch: 0
descriptor:
  node_archetype: compute
input_model: str
output_model: str
"""

        source = MagicMock()
        source.source_type = "filesystem"
        source.source_description = "test source"
        source.discover_contracts = AsyncMock(
            return_value=[
                # Contract with handler_id already extracted (simulating composite source)
                ModelDiscoveredContract(
                    origin="filesystem",
                    ref=Path("/test/handlers/foo/contract.yaml"),
                    text=contract_yaml,
                    handler_id="pre.extracted.handler",  # Already set
                ),
            ]
        )

        service = ServiceContractPublisher(mock_publisher, source, config)
        result = await service.publish_all()

        # Should use the pre-extracted handler_id
        assert len(result.published) == 1
        assert result.published[0] == "pre.extracted.handler"

    @pytest.mark.asyncio
    async def test_handler_id_extraction_handles_yaml_error(
        self,
        mock_publisher: AsyncMock,
    ) -> None:
        """Test that YAML errors during extraction don't break the flow.

        Invalid YAML should be passed through and fail during validation
        with a proper error, not during handler_id extraction.
        """
        config = ModelContractPublisherConfig(
            mode="filesystem",
            filesystem_root=Path("/test"),
            allow_zero_contracts=True,  # Allow zero valid contracts for this test
        )

        invalid_yaml = """handler_id: [unclosed bracket
name: This YAML is broken
"""

        source = MagicMock()
        source.source_type = "filesystem"
        source.source_description = "test source"
        source.discover_contracts = AsyncMock(
            return_value=[
                ModelDiscoveredContract(
                    origin="filesystem",
                    ref=Path("/test/handlers/bad/contract.yaml"),
                    text=invalid_yaml,
                ),
            ]
        )

        service = ServiceContractPublisher(mock_publisher, source, config)
        result = await service.publish_all()

        # No contracts published, but one error recorded
        assert len(result.published) == 0
        assert len(result.contract_errors) == 1
        assert result.contract_errors[0].error_type == "yaml_parse"
