# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for MCPServerLifecycle dev mode contract scanning.

This test suite validates the MCPServerLifecycle class focusing on dev mode
functionality that scans local contract.yaml files to discover MCP-enabled
orchestrators.

Test Organization:
    - TestMCPServerLifecycleInit: Initialization and properties
    - TestMCPServerLifecycleDevModeStart: Dev mode start() behavior
    - TestDiscoverFromContracts: Contract discovery edge cases
    - TestMCPServerLifecycleShutdown: Shutdown and cleanup
    - TestMCPServerLifecycleDescribe: Observability metadata

Note:
    These tests use temporary directories with mock contract.yaml files
    to validate contract scanning without infrastructure dependencies.

Coverage Goals:
    - >90% code coverage for MCPServerLifecycle
    - All dev mode paths tested
    - Edge cases for contract parsing tested
    - Lifecycle state transitions validated

Related Tickets:
    - OMN-1281: MCP Adapter - Expose ONEX Nodes as MCP Tools
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from omnibase_infra.services.mcp.mcp_server_lifecycle import (
    MCPServerLifecycle,
    ModelMCPServerConfig,
)

# =============================================================================
# Test Constants
# =============================================================================

MCP_ENABLED_ORCHESTRATOR_CONTRACT = """\
name: test-orchestrator
description: Test orchestrator for MCP exposure
node_version: "1.2.3"
node_type: ORCHESTRATOR_GENERIC

mcp:
  expose: true
  tool_name: test_tool
  description: AI-friendly tool description
  timeout_seconds: 45
"""

MCP_ENABLED_ORCHESTRATOR_MINIMAL = """\
name: minimal-orchestrator
node_type: ORCHESTRATOR_GENERIC

mcp:
  expose: true
"""

MCP_DISABLED_CONTRACT = """\
name: no-mcp-orchestrator
description: Orchestrator without MCP config
node_version: "1.0.0"
node_type: ORCHESTRATOR_GENERIC
"""

MCP_EXPOSE_FALSE_CONTRACT = """\
name: disabled-mcp-orchestrator
node_type: ORCHESTRATOR_GENERIC

mcp:
  expose: false
"""

NON_ORCHESTRATOR_WITH_MCP = """\
name: effect-with-mcp
description: Effect node with MCP config (should be skipped)
node_version: "1.0.0"
node_type: EFFECT_GENERIC

mcp:
  expose: true
  tool_name: effect_tool
"""

COMPUTE_WITH_MCP = """\
name: compute-with-mcp
node_type: COMPUTE_GENERIC

mcp:
  expose: true
"""

REDUCER_WITH_MCP = """\
name: reducer-with-mcp
node_type: REDUCER_GENERIC

mcp:
  expose: true
"""

INVALID_YAML_CONTRACT = """\
name: invalid-contract
node_type: ORCHESTRATOR_GENERIC
  mcp:
  expose: true
invalid_indentation_here
"""

EMPTY_CONTRACT = ""

CONTRACT_WITH_PARTIAL_MCP = """\
name: partial-mcp
node_type: ORCHESTRATOR_GENERIC

mcp:
  description: Only description, no expose flag
"""


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def tmp_contracts_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for contract files.

    Returns:
        Path to temporary contracts directory.
    """
    contracts_dir = tmp_path / "contracts"
    contracts_dir.mkdir()
    return contracts_dir


@pytest.fixture
def dev_mode_config(tmp_contracts_dir: Path) -> ModelMCPServerConfig:
    """Create a dev mode config pointing to temporary contracts directory.

    Args:
        tmp_contracts_dir: Temporary contracts directory.

    Returns:
        ModelMCPServerConfig with dev_mode=True.
    """
    return ModelMCPServerConfig(
        dev_mode=True,
        contracts_dir=str(tmp_contracts_dir),
        kafka_enabled=False,  # Disable Kafka for unit tests
    )


@pytest.fixture
def production_config() -> ModelMCPServerConfig:
    """Create a production mode config.

    Returns:
        ModelMCPServerConfig with dev_mode=False.
    """
    return ModelMCPServerConfig(
        dev_mode=False,
        kafka_enabled=False,
    )


def create_contract_file(
    contracts_dir: Path,
    node_name: str,
    content: str,
) -> Path:
    """Create a contract.yaml file in a node subdirectory.

    Args:
        contracts_dir: Root contracts directory.
        node_name: Name of the node subdirectory.
        content: YAML content for contract.yaml.

    Returns:
        Path to the created contract.yaml file.
    """
    node_dir = contracts_dir / node_name
    node_dir.mkdir(parents=True, exist_ok=True)
    contract_file = node_dir / "contract.yaml"
    contract_file.write_text(content)
    return contract_file


# =============================================================================
# Test Classes
# =============================================================================


@pytest.mark.unit
class TestMCPServerLifecycleInit:
    """Tests for MCPServerLifecycle initialization."""

    def test_init_creates_instance_with_config(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
    ) -> None:
        """Should create instance with provided configuration.

        Given: A ModelMCPServerConfig
        When: MCPServerLifecycle is instantiated
        Then: Instance is created with correct initial state
        """
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)

        assert lifecycle._container is mock_container
        assert lifecycle._config == dev_mode_config
        assert lifecycle._started is False
        assert lifecycle._registry is None
        assert lifecycle._executor is None

    def test_is_running_returns_false_before_start(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
    ) -> None:
        """Should return False for is_running before start().

        Given: A newly created MCPServerLifecycle
        When: is_running is accessed
        Then: Returns False
        """
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)

        assert lifecycle.is_running is False

    def test_registry_returns_none_before_start(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
    ) -> None:
        """Should return None for registry before start().

        Given: A newly created MCPServerLifecycle
        When: registry is accessed
        Then: Returns None
        """
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)

        assert lifecycle.registry is None

    def test_executor_returns_none_before_start(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
    ) -> None:
        """Should return None for executor before start().

        Given: A newly created MCPServerLifecycle
        When: executor is accessed
        Then: Returns None
        """
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)

        assert lifecycle.executor is None


@pytest.mark.unit
@pytest.mark.asyncio
class TestMCPServerLifecycleDevModeStart:
    """Tests for MCPServerLifecycle.start() in dev mode."""

    async def test_start_in_dev_mode_creates_registry(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
        tmp_contracts_dir: Path,
    ) -> None:
        """Should create registry when started in dev mode.

        Given: A dev mode config
        When: start() is called
        Then: Registry is created and accessible
        """
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)

        await lifecycle.start()

        assert lifecycle.registry is not None
        assert lifecycle.is_running is True

        await lifecycle.shutdown()

    async def test_start_in_dev_mode_creates_executor(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
        tmp_contracts_dir: Path,
    ) -> None:
        """Should create executor when started in dev mode.

        Given: A dev mode config
        When: start() is called
        Then: Executor is created and accessible
        """
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)

        await lifecycle.start()

        assert lifecycle.executor is not None

        await lifecycle.shutdown()

    async def test_start_discovers_mcp_enabled_orchestrator(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
        tmp_contracts_dir: Path,
    ) -> None:
        """Should discover MCP-enabled orchestrator from contract.

        Given: A contract.yaml with mcp.expose=true for an orchestrator
        When: start() is called in dev mode
        Then: Tool is registered in the registry
        """
        create_contract_file(
            tmp_contracts_dir,
            "test-node",
            MCP_ENABLED_ORCHESTRATOR_CONTRACT,
        )
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)

        await lifecycle.start()

        assert lifecycle.registry is not None
        assert lifecycle.registry.tool_count == 1
        tool = await lifecycle.registry.get_tool("test_tool")
        assert tool is not None
        assert tool.name == "test_tool"
        assert tool.description == "AI-friendly tool description"
        assert tool.version == "1.2.3"
        assert tool.timeout_seconds == 45

        await lifecycle.shutdown()

    async def test_start_uses_fallback_values_for_minimal_contract(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
        tmp_contracts_dir: Path,
    ) -> None:
        """Should use fallback values when MCP config is minimal.

        Given: A contract.yaml with minimal MCP config (just expose=true)
        When: start() is called in dev mode
        Then: Tool is registered with fallback values
        """
        create_contract_file(
            tmp_contracts_dir,
            "minimal-node",
            MCP_ENABLED_ORCHESTRATOR_MINIMAL,
        )
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)

        await lifecycle.start()

        assert lifecycle.registry is not None
        assert lifecycle.registry.tool_count == 1
        # Tool name should fall back to contract name
        tool = await lifecycle.registry.get_tool("minimal-orchestrator")
        assert tool is not None
        assert tool.name == "minimal-orchestrator"
        # Description should fall back to "ONEX: {name}"
        assert "ONEX: minimal-orchestrator" in tool.description
        # Default timeout
        assert tool.timeout_seconds == 30
        # Default version
        assert tool.version == "1.0.0"

        await lifecycle.shutdown()

    async def test_start_skips_contract_without_mcp_config(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
        tmp_contracts_dir: Path,
    ) -> None:
        """Should skip contracts without MCP configuration.

        Given: A contract.yaml without mcp config
        When: start() is called in dev mode
        Then: No tool is registered
        """
        create_contract_file(
            tmp_contracts_dir,
            "no-mcp-node",
            MCP_DISABLED_CONTRACT,
        )
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)

        await lifecycle.start()

        assert lifecycle.registry is not None
        assert lifecycle.registry.tool_count == 0

        await lifecycle.shutdown()

    async def test_start_skips_contract_with_mcp_expose_false(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
        tmp_contracts_dir: Path,
    ) -> None:
        """Should skip contracts with mcp.expose=false.

        Given: A contract.yaml with mcp.expose=false
        When: start() is called in dev mode
        Then: No tool is registered
        """
        create_contract_file(
            tmp_contracts_dir,
            "disabled-mcp-node",
            MCP_EXPOSE_FALSE_CONTRACT,
        )
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)

        await lifecycle.start()

        assert lifecycle.registry is not None
        assert lifecycle.registry.tool_count == 0

        await lifecycle.shutdown()

    async def test_start_skips_non_orchestrator_with_mcp_expose(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
        tmp_contracts_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Should skip non-orchestrator nodes even with mcp.expose=true.

        Given: An effect node contract with mcp.expose=true
        When: start() is called in dev mode
        Then: Tool is NOT registered (only orchestrators can be exposed)
        """
        create_contract_file(
            tmp_contracts_dir,
            "effect-node",
            NON_ORCHESTRATOR_WITH_MCP,
        )
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)

        with caplog.at_level(logging.DEBUG):
            await lifecycle.start()

        assert lifecycle.registry is not None
        assert lifecycle.registry.tool_count == 0
        # Verify debug log mentions skipping non-orchestrator
        assert any(
            "Skipping non-orchestrator" in record.message for record in caplog.records
        )

        await lifecycle.shutdown()

    async def test_start_skips_compute_node_with_mcp(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
        tmp_contracts_dir: Path,
    ) -> None:
        """Should skip compute nodes even with mcp.expose=true.

        Given: A compute node contract with mcp.expose=true
        When: start() is called in dev mode
        Then: Tool is NOT registered
        """
        create_contract_file(
            tmp_contracts_dir,
            "compute-node",
            COMPUTE_WITH_MCP,
        )
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)

        await lifecycle.start()

        assert lifecycle.registry is not None
        assert lifecycle.registry.tool_count == 0

        await lifecycle.shutdown()

    async def test_start_skips_reducer_node_with_mcp(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
        tmp_contracts_dir: Path,
    ) -> None:
        """Should skip reducer nodes even with mcp.expose=true.

        Given: A reducer node contract with mcp.expose=true
        When: start() is called in dev mode
        Then: Tool is NOT registered
        """
        create_contract_file(
            tmp_contracts_dir,
            "reducer-node",
            REDUCER_WITH_MCP,
        )
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)

        await lifecycle.start()

        assert lifecycle.registry is not None
        assert lifecycle.registry.tool_count == 0

        await lifecycle.shutdown()

    async def test_start_discovers_multiple_orchestrators(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
        tmp_contracts_dir: Path,
    ) -> None:
        """Should discover multiple MCP-enabled orchestrators.

        Given: Multiple contract.yaml files with mcp.expose=true
        When: start() is called in dev mode
        Then: All tools are registered
        """
        create_contract_file(
            tmp_contracts_dir,
            "node-a",
            MCP_ENABLED_ORCHESTRATOR_CONTRACT,
        )
        create_contract_file(
            tmp_contracts_dir,
            "node-b",
            MCP_ENABLED_ORCHESTRATOR_MINIMAL,
        )
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)

        await lifecycle.start()

        assert lifecycle.registry is not None
        assert lifecycle.registry.tool_count == 2

        await lifecycle.shutdown()

    async def test_start_filters_mixed_node_types(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
        tmp_contracts_dir: Path,
    ) -> None:
        """Should only register orchestrators from mixed contracts.

        Given: Mix of orchestrator and non-orchestrator contracts with mcp.expose
        When: start() is called in dev mode
        Then: Only orchestrators are registered
        """
        # One valid orchestrator
        create_contract_file(
            tmp_contracts_dir,
            "orchestrator-node",
            MCP_ENABLED_ORCHESTRATOR_MINIMAL,
        )
        # Effect node - should be skipped
        create_contract_file(
            tmp_contracts_dir,
            "effect-node",
            NON_ORCHESTRATOR_WITH_MCP,
        )
        # Compute node - should be skipped
        create_contract_file(
            tmp_contracts_dir,
            "compute-node",
            COMPUTE_WITH_MCP,
        )
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)

        await lifecycle.start()

        assert lifecycle.registry is not None
        assert lifecycle.registry.tool_count == 1

        await lifecycle.shutdown()

    async def test_start_is_idempotent(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
        tmp_contracts_dir: Path,
    ) -> None:
        """Should be idempotent - multiple start() calls have no effect.

        Given: An already started lifecycle
        When: start() is called again
        Then: Returns immediately without error
        """
        create_contract_file(
            tmp_contracts_dir,
            "test-node",
            MCP_ENABLED_ORCHESTRATOR_CONTRACT,
        )
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)

        await lifecycle.start()
        first_registry = lifecycle.registry
        first_tool_count = lifecycle.registry.tool_count if lifecycle.registry else 0

        # Second start should be no-op
        await lifecycle.start()

        # Registry should be the same object
        assert lifecycle.registry is first_registry
        assert lifecycle.registry.tool_count == first_tool_count

        await lifecycle.shutdown()


@pytest.mark.unit
@pytest.mark.asyncio
class TestDiscoverFromContracts:
    """Tests for _discover_from_contracts edge cases."""

    async def test_discover_with_empty_contracts_dir(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
        tmp_contracts_dir: Path,
    ) -> None:
        """Should return 0 tools for empty contracts directory.

        Given: An empty contracts directory
        When: start() is called in dev mode
        Then: Registry has 0 tools
        """
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)

        await lifecycle.start()

        assert lifecycle.registry is not None
        assert lifecycle.registry.tool_count == 0

        await lifecycle.shutdown()

    async def test_discover_with_nonexistent_contracts_dir(
        self,
        mock_container: MagicMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Should log warning and return 0 tools for non-existent directory.

        Given: A config with non-existent contracts_dir
        When: start() is called in dev mode
        Then: Logs warning and registry has 0 tools
        """
        config = ModelMCPServerConfig(
            dev_mode=True,
            contracts_dir=str(tmp_path / "nonexistent"),
            kafka_enabled=False,
        )
        lifecycle = MCPServerLifecycle(mock_container, config)

        with caplog.at_level(logging.WARNING):
            await lifecycle.start()

        assert lifecycle.registry is not None
        assert lifecycle.registry.tool_count == 0
        assert any("does not exist" in record.message for record in caplog.records)

        await lifecycle.shutdown()

    async def test_discover_with_no_contracts_dir_config(
        self,
        mock_container: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Should log warning when dev_mode=True but no contracts_dir.

        Given: A dev mode config with contracts_dir=None
        When: start() is called
        Then: Logs warning and registry has 0 tools
        """
        config = ModelMCPServerConfig(
            dev_mode=True,
            contracts_dir=None,
            kafka_enabled=False,
        )
        lifecycle = MCPServerLifecycle(mock_container, config)

        with caplog.at_level(logging.WARNING):
            await lifecycle.start()

        assert lifecycle.registry is not None
        assert lifecycle.registry.tool_count == 0
        assert any(
            "no contracts_dir specified" in record.message for record in caplog.records
        )

        await lifecycle.shutdown()

    async def test_discover_skips_invalid_yaml(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
        tmp_contracts_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Should skip invalid YAML contracts with warning.

        Given: A contract.yaml with invalid YAML syntax
        When: start() is called in dev mode
        Then: Logs warning and continues without crashing
        """
        create_contract_file(
            tmp_contracts_dir,
            "invalid-node",
            INVALID_YAML_CONTRACT,
        )
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)

        with caplog.at_level(logging.WARNING):
            await lifecycle.start()

        assert lifecycle.registry is not None
        assert lifecycle.registry.tool_count == 0
        assert any(
            "Failed to parse contract YAML" in record.message
            for record in caplog.records
        )

        await lifecycle.shutdown()

    async def test_discover_skips_empty_contract_file(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
        tmp_contracts_dir: Path,
    ) -> None:
        """Should skip empty contract files.

        Given: An empty contract.yaml file
        When: start() is called in dev mode
        Then: No tool is registered (file is skipped)
        """
        create_contract_file(
            tmp_contracts_dir,
            "empty-node",
            EMPTY_CONTRACT,
        )
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)

        await lifecycle.start()

        assert lifecycle.registry is not None
        assert lifecycle.registry.tool_count == 0

        await lifecycle.shutdown()

    async def test_discover_skips_partial_mcp_config(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
        tmp_contracts_dir: Path,
    ) -> None:
        """Should skip contracts with MCP config but no expose flag.

        Given: A contract.yaml with mcp section but no expose: true
        When: start() is called in dev mode
        Then: No tool is registered
        """
        create_contract_file(
            tmp_contracts_dir,
            "partial-mcp-node",
            CONTRACT_WITH_PARTIAL_MCP,
        )
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)

        await lifecycle.start()

        assert lifecycle.registry is not None
        assert lifecycle.registry.tool_count == 0

        await lifecycle.shutdown()

    async def test_discover_includes_metadata_in_tool_definition(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
        tmp_contracts_dir: Path,
    ) -> None:
        """Should include contract metadata in tool definition.

        Given: An MCP-enabled orchestrator contract
        When: start() is called in dev mode
        Then: Tool definition includes contract path and source metadata
        """
        contract_file = create_contract_file(
            tmp_contracts_dir,
            "metadata-test",
            MCP_ENABLED_ORCHESTRATOR_CONTRACT,
        )
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)

        await lifecycle.start()

        assert lifecycle.registry is not None
        tool = await lifecycle.registry.get_tool("test_tool")
        assert tool is not None
        assert tool.metadata.get("source") == "local_contract"
        assert tool.metadata.get("contract_path") == str(contract_file)
        assert "ORCHESTRATOR" in str(tool.metadata.get("node_type"))

        await lifecycle.shutdown()

    async def test_discover_handles_nested_directory_structure(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
        tmp_contracts_dir: Path,
    ) -> None:
        """Should discover contracts in nested directories.

        Given: Contracts in deeply nested subdirectories
        When: start() is called in dev mode
        Then: All contracts are discovered via rglob
        """
        # Create nested structure
        nested_dir = tmp_contracts_dir / "nodes" / "orchestrators" / "deep"
        nested_dir.mkdir(parents=True)
        contract_file = nested_dir / "contract.yaml"
        contract_file.write_text(MCP_ENABLED_ORCHESTRATOR_CONTRACT)

        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)

        await lifecycle.start()

        assert lifecycle.registry is not None
        assert lifecycle.registry.tool_count == 1

        await lifecycle.shutdown()


@pytest.mark.unit
@pytest.mark.asyncio
class TestMCPServerLifecycleShutdown:
    """Tests for MCPServerLifecycle.shutdown()."""

    async def test_shutdown_clears_registry(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
        tmp_contracts_dir: Path,
    ) -> None:
        """Should clear registry on shutdown.

        Given: A started lifecycle with registered tools
        When: shutdown() is called
        Then: Registry is cleared and set to None
        """
        create_contract_file(
            tmp_contracts_dir,
            "test-node",
            MCP_ENABLED_ORCHESTRATOR_CONTRACT,
        )
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)
        await lifecycle.start()
        assert lifecycle.registry is not None

        await lifecycle.shutdown()

        assert lifecycle.registry is None
        assert lifecycle.is_running is False

    async def test_shutdown_clears_executor(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
        tmp_contracts_dir: Path,
    ) -> None:
        """Should clear executor on shutdown.

        Given: A started lifecycle
        When: shutdown() is called
        Then: Executor is set to None
        """
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)
        await lifecycle.start()
        assert lifecycle.executor is not None

        await lifecycle.shutdown()

        assert lifecycle.executor is None

    async def test_shutdown_is_idempotent(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
        tmp_contracts_dir: Path,
    ) -> None:
        """Should be idempotent - multiple shutdown() calls have no effect.

        Given: An already shut down lifecycle
        When: shutdown() is called again
        Then: Returns without error
        """
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)
        await lifecycle.start()

        await lifecycle.shutdown()
        # Second shutdown should be no-op
        await lifecycle.shutdown()

        assert lifecycle.is_running is False
        assert lifecycle.registry is None

    async def test_shutdown_before_start_is_safe(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
    ) -> None:
        """Should handle shutdown() before start() gracefully.

        Given: A lifecycle that was never started
        When: shutdown() is called
        Then: Returns without error
        """
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)

        # Should not raise
        await lifecycle.shutdown()

        assert lifecycle.is_running is False


@pytest.mark.unit
class TestMCPServerLifecycleDescribe:
    """Tests for MCPServerLifecycle.describe() observability."""

    def test_describe_before_start(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
    ) -> None:
        """Should return metadata with started=False before start.

        Given: A lifecycle that hasn't been started
        When: describe() is called
        Then: Returns metadata with started=False and 0 tools, no Consul fields
        """
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)

        metadata = lifecycle.describe()

        assert metadata["service_name"] == "MCPServerLifecycle"
        assert metadata["started"] is False
        assert metadata["registry_tool_count"] == 0
        assert metadata["sync_running"] is False
        assert "config" in metadata
        # Consul fields must not appear
        config_meta = metadata["config"]
        assert isinstance(config_meta, dict)
        assert "consul_host" not in config_meta
        assert "consul_port" not in config_meta
        assert "registry_query_limit" in config_meta

    @pytest.mark.asyncio
    async def test_describe_after_start(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
        tmp_contracts_dir: Path,
    ) -> None:
        """Should return metadata with started=True after start.

        Given: A started lifecycle with registered tools
        When: describe() is called
        Then: Returns metadata with started=True and correct tool count
        """
        create_contract_file(
            tmp_contracts_dir,
            "test-node",
            MCP_ENABLED_ORCHESTRATOR_CONTRACT,
        )
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)
        await lifecycle.start()

        metadata = lifecycle.describe()

        assert metadata["started"] is True
        assert metadata["registry_tool_count"] == 1
        assert isinstance(metadata["config"], dict)
        assert metadata["config"]["http_port"] == dev_mode_config.http_port

        await lifecycle.shutdown()

    @pytest.mark.asyncio
    async def test_describe_after_shutdown(
        self,
        mock_container: MagicMock,
        dev_mode_config: ModelMCPServerConfig,
        tmp_contracts_dir: Path,
    ) -> None:
        """Should return metadata with started=False after shutdown.

        Given: A lifecycle that has been started and shutdown
        When: describe() is called
        Then: Returns metadata with started=False
        """
        lifecycle = MCPServerLifecycle(mock_container, dev_mode_config)
        await lifecycle.start()
        await lifecycle.shutdown()

        metadata = lifecycle.describe()

        assert metadata["started"] is False
        assert metadata["registry_tool_count"] == 0


@pytest.mark.unit
class TestModelMCPServerConfig:
    """Tests for ModelMCPServerConfig defaults."""

    def test_default_values(self) -> None:
        """Should have sensible default values with no Consul fields.

        Given: No constructor arguments
        When: ModelMCPServerConfig is created
        Then: Default values are set correctly (no consul_host/port/scheme/token)
        """
        config = ModelMCPServerConfig()

        assert config.registry_query_limit == 100
        assert config.kafka_enabled is True
        assert config.http_host == "0.0.0.0"  # noqa: S104
        assert config.http_port == 8090
        assert config.default_timeout == 30.0
        assert config.dev_mode is False
        assert config.contracts_dir is None
        # Verify Consul fields are gone
        assert not hasattr(config, "consul_host")
        assert not hasattr(config, "consul_port")
        assert not hasattr(config, "consul_scheme")
        assert not hasattr(config, "consul_token")

    def test_dev_mode_config(self) -> None:
        """Should accept dev_mode and contracts_dir.

        Given: dev_mode=True and contracts_dir specified
        When: ModelMCPServerConfig is created
        Then: Values are set correctly
        """
        config = ModelMCPServerConfig(
            dev_mode=True,
            contracts_dir="/path/to/contracts",
        )

        assert config.dev_mode is True
        assert config.contracts_dir == "/path/to/contracts"

    def test_registry_query_limit_config(self) -> None:
        """Should accept registry_query_limit.

        Given: registry_query_limit=200
        When: ModelMCPServerConfig is created
        Then: Value is set correctly
        """
        config = ModelMCPServerConfig(registry_query_limit=200)

        assert config.registry_query_limit == 200
