# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ServiceMCPToolRegistry.register_generated_tool (OMN-13612).

This is the canonical replacement for the bespoke SEA ToolRegistry.register
(onex-self-extending-agent/src/agent/tool_registry.py). The SEA registry stored
generated-tool registration records in its own in-process dict; this routes that
registration *semantics* onto the canonical MCP registry so generated tools surface
through the one canonical ServiceMCPToolRegistry / ServiceMCPToolSync exposure path.

The contract this test is written from (the source of truth):

- register_generated_tool registers a generated COMPUTE node into the canonical
  registry, derived from the same metadata the SEA ToolRegistration captured
  (node name, description, contract/handler artifact hashes, correlation id).
- After registration the tool is retrievable from the *canonical* registry
  (get_tool / has_tool / list_tools / tool_count) -- not a bespoke store.
- The stored tool carries the tags the canonical MCP exposure rule requires for a
  generated compute node: mcp-enabled + node-type:compute + generated + mcp-tool:<name>
  (ServiceMCPToolSync._is_mcp_exposable must accept them).
- The returned ModelMCPGeneratedToolRegistration binds the artifact hashes and the
  registry event id used for the upsert.
- Re-registering the same generated artifact is idempotent (same content -> not a
  new version); a newer correlation/registration yields a fresh registry version.
"""

from __future__ import annotations

import hashlib
from uuid import uuid4

import pytest

from omnibase_infra.models.mcp.model_mcp_generated_tool_registration import (
    ModelMCPGeneratedToolRegistration,
)
from omnibase_infra.services.mcp.service_mcp_tool_registry import ServiceMCPToolRegistry
from omnibase_infra.services.mcp.service_mcp_tool_sync import ServiceMCPToolSync

_CONTRACT_YAML = """\
name: node_generated_sentiment
contract_version: {major: 1, minor: 0, patch: 0}
node_type: COMPUTE_GENERIC
description: Generated sentiment classifier
input_model: {name: ModelSentimentInput, module: generated.models}
output_model: {name: ModelSentimentOutput, module: generated.models}
"""

_HANDLER_SOURCE = "def handle(input_data):\n    return {'label': 'positive'}\n"


@pytest.fixture
def registry() -> ServiceMCPToolRegistry:
    return ServiceMCPToolRegistry()


def _expected_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


class TestRegisterGeneratedTool:
    @pytest.mark.asyncio
    async def test_register_returns_typed_record_with_artifact_hashes(
        self, registry: ServiceMCPToolRegistry
    ) -> None:
        correlation_id = uuid4()
        record = await registry.register_generated_tool(
            node_name="node_generated_sentiment",
            description="Generated sentiment classifier",
            contract_yaml=_CONTRACT_YAML,
            handler_source=_HANDLER_SOURCE,
            correlation_id=correlation_id,
        )

        assert isinstance(record, ModelMCPGeneratedToolRegistration)
        assert record.name == "node_generated_sentiment"
        assert record.description == "Generated sentiment classifier"
        assert record.generated_contract_hash == _expected_hash(_CONTRACT_YAML)
        assert record.generated_handler_hash == _expected_hash(_HANDLER_SOURCE)
        assert record.generation_correlation_id == correlation_id
        assert record.registry_event_version  # non-empty version key

    @pytest.mark.asyncio
    async def test_generated_tool_appears_in_canonical_registry(
        self, registry: ServiceMCPToolRegistry
    ) -> None:
        await registry.register_generated_tool(
            node_name="node_generated_sentiment",
            description="Generated sentiment classifier",
            contract_yaml=_CONTRACT_YAML,
            handler_source=_HANDLER_SOURCE,
            correlation_id=uuid4(),
        )

        assert registry.tool_count == 1
        assert await registry.has_tool("node_generated_sentiment") is True

        tool = await registry.get_tool("node_generated_sentiment")
        assert tool is not None
        assert tool.name == "node_generated_sentiment"
        assert tool.description == "Generated sentiment classifier"

        tools = await registry.list_tools()
        assert [t.name for t in tools] == ["node_generated_sentiment"]

    @pytest.mark.asyncio
    async def test_registered_tool_passes_canonical_mcp_exposure_rule(
        self, registry: ServiceMCPToolRegistry
    ) -> None:
        """The stored tool's tags must satisfy ServiceMCPToolSync._is_mcp_exposable.

        This is what proves the registration rides the *canonical* exposure path:
        a generated compute tool is only surfaced if it carries
        mcp-enabled + node-type:compute + generated.
        """
        await registry.register_generated_tool(
            node_name="node_generated_sentiment",
            description="Generated sentiment classifier",
            contract_yaml=_CONTRACT_YAML,
            handler_source=_HANDLER_SOURCE,
            correlation_id=uuid4(),
        )

        tool = await registry.get_tool("node_generated_sentiment")
        assert tool is not None

        tags = tool.metadata["tags"]
        assert isinstance(tags, list)

        class _StubBus:
            environment = "test"

        sync = ServiceMCPToolSync(registry=registry, bus=_StubBus())  # type: ignore[arg-type]
        assert sync._is_mcp_exposable([str(t) for t in tags]) is True
        assert ServiceMCPToolSync.TAG_MCP_ENABLED in tags
        assert ServiceMCPToolSync.TAG_NODE_TYPE_COMPUTE in tags
        assert ServiceMCPToolSync.TAG_GENERATED in tags
        assert (
            f"{ServiceMCPToolSync.TAG_PREFIX_MCP_TOOL}node_generated_sentiment" in tags
        )

    @pytest.mark.asyncio
    async def test_reregister_same_artifact_is_idempotent(
        self, registry: ServiceMCPToolRegistry
    ) -> None:
        first = await registry.register_generated_tool(
            node_name="node_generated_sentiment",
            description="Generated sentiment classifier",
            contract_yaml=_CONTRACT_YAML,
            handler_source=_HANDLER_SOURCE,
            correlation_id=uuid4(),
        )
        second = await registry.register_generated_tool(
            node_name="node_generated_sentiment",
            description="Generated sentiment classifier",
            contract_yaml=_CONTRACT_YAML,
            handler_source=_HANDLER_SOURCE,
            correlation_id=uuid4(),
        )

        # Same artifact (same content hashes) -> same registry version key,
        # one tool, no duplicate entries.
        assert registry.tool_count == 1
        assert first.registry_event_version == second.registry_event_version

    @pytest.mark.asyncio
    async def test_new_artifact_version_updates_registry(
        self, registry: ServiceMCPToolRegistry
    ) -> None:
        await registry.register_generated_tool(
            node_name="node_generated_sentiment",
            description="v1",
            contract_yaml=_CONTRACT_YAML,
            handler_source=_HANDLER_SOURCE,
            correlation_id=uuid4(),
        )

        new_handler = _HANDLER_SOURCE + "# revised\n"
        record = await registry.register_generated_tool(
            node_name="node_generated_sentiment",
            description="v2",
            contract_yaml=_CONTRACT_YAML,
            handler_source=new_handler,
            correlation_id=uuid4(),
        )

        assert registry.tool_count == 1
        tool = await registry.get_tool("node_generated_sentiment")
        assert tool is not None
        assert tool.description == "v2"
        assert record.generated_handler_hash == _expected_hash(new_handler)

    @pytest.mark.asyncio
    async def test_empty_node_name_is_rejected(
        self, registry: ServiceMCPToolRegistry
    ) -> None:
        with pytest.raises(ValueError):
            await registry.register_generated_tool(
                node_name="",
                description="Generated sentiment classifier",
                contract_yaml=_CONTRACT_YAML,
                handler_source=_HANDLER_SOURCE,
                correlation_id=uuid4(),
            )
