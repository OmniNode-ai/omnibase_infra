# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration tests for contract-driven node capability declaration.

Tests the full flow: contract YAML -> ContractNodeCapabilityExtractor ->
ModelIntrospectionConfig -> MixinNodeIntrospection -> introspection event
with correct declared_capabilities.

OMN-5054: Contract-driven node capability declaration.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from uuid import UUID

import pytest

from omnibase_core.enums import EnumNodeKind
from omnibase_infra.capabilities.contract_node_capability_extractor import (
    ContractNodeCapabilityExtractor,
)
from omnibase_infra.mixins import MixinNodeIntrospection
from omnibase_infra.models.discovery.model_introspection_config import (
    ModelIntrospectionConfig,
)
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
]

TEST_NODE_UUID = UUID("00000000-0000-0000-0000-000000005054")


class NodeWithDeclaredCapabilities(MixinNodeIntrospection):
    """Test node that uses contract-driven declared capabilities."""

    def __init__(
        self,
        node_id: UUID,
        declared_capabilities: ModelNodeCapabilities | None = None,
    ) -> None:
        config = ModelIntrospectionConfig(
            node_id=node_id,
            node_type=EnumNodeKind.EFFECT,
            node_name="test_node_with_caps",
            version="1.0.0",
            declared_capabilities=declared_capabilities,
        )
        self.initialize_introspection(config)

    async def execute_operation(self, data: dict[str, str]) -> dict[str, str]:
        return data


class TestContractDrivenCapabilitiesE2E:
    """End-to-end tests for contract-driven capability flow."""

    async def test_declared_capabilities_flow_to_introspection_event(
        self, tmp_path: Path
    ) -> None:
        """Capabilities from contract YAML should appear in introspection event."""
        # 1. Write contract with node_capabilities
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            dedent("""\
            name: "test_node"
            node_type: "EFFECT_GENERIC"
            node_capabilities:
              postgres: true
              read: true
              write: true
              transactions: true
            """)
        )

        # 2. Extract capabilities from contract
        extractor = ContractNodeCapabilityExtractor()
        caps = extractor.extract_from_yaml(contract)

        # 3. Create node with extracted capabilities
        node = NodeWithDeclaredCapabilities(
            node_id=TEST_NODE_UUID,
            declared_capabilities=caps,
        )

        # 4. Get introspection data
        event = await node.get_introspection_data()

        # 5. Verify declared_capabilities in the event
        assert event.declared_capabilities is not None
        dc = event.declared_capabilities
        assert dc["postgres"] is True
        assert dc["read"] is True
        assert dc["write"] is True
        assert dc["transactions"] is True
        # Unspecified fields should be False
        assert dc["database"] is False
        assert dc["routing"] is False

    async def test_default_capabilities_when_no_contract_block(
        self, tmp_path: Path
    ) -> None:
        """Without node_capabilities block, all capabilities should be False."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            dedent("""\
            name: "test_node"
            node_type: "EFFECT_GENERIC"
            """)
        )

        extractor = ContractNodeCapabilityExtractor()
        caps = extractor.extract_from_yaml(contract)

        node = NodeWithDeclaredCapabilities(
            node_id=TEST_NODE_UUID,
            declared_capabilities=caps,
        )

        event = await node.get_introspection_data()

        # All defaults
        assert event.declared_capabilities["postgres"] is False
        assert event.declared_capabilities["read"] is False
        assert event.declared_capabilities["write"] is False

    async def test_legacy_node_without_declared_capabilities(self) -> None:
        """Legacy nodes (no declared_capabilities) should get all-false defaults."""
        node = NodeWithDeclaredCapabilities(
            node_id=TEST_NODE_UUID,
            declared_capabilities=None,
        )

        event = await node.get_introspection_data()

        # Should be the default ModelNodeCapabilities (all False)
        assert event.declared_capabilities is not None
        assert event.declared_capabilities["postgres"] is False
        assert event.declared_capabilities["read"] is False

    async def test_explicit_override_takes_precedence(self) -> None:
        """Explicitly passed capabilities should override defaults."""
        explicit_caps = ModelNodeCapabilities(
            postgres=True,
            read=True,
            write=True,
            database=True,
            transactions=True,
        )

        node = NodeWithDeclaredCapabilities(
            node_id=TEST_NODE_UUID,
            declared_capabilities=explicit_caps,
        )

        event = await node.get_introspection_data()

        assert event.declared_capabilities["postgres"] is True
        assert event.declared_capabilities["database"] is True
        assert event.declared_capabilities["transactions"] is True

    async def test_capabilities_preserved_on_cache_hit(self, tmp_path: Path) -> None:
        """Declared capabilities should be preserved on cache hit."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            dedent("""\
            name: "test_node"
            node_capabilities:
              postgres: true
              write: true
            """)
        )

        extractor = ContractNodeCapabilityExtractor()
        caps = extractor.extract_from_yaml(contract)

        node = NodeWithDeclaredCapabilities(
            node_id=TEST_NODE_UUID,
            declared_capabilities=caps,
        )

        # First call (cache miss)
        event1 = await node.get_introspection_data()
        assert event1.declared_capabilities["postgres"] is True

        # Second call (cache hit)
        event2 = await node.get_introspection_data()
        assert event2.declared_capabilities["postgres"] is True
        assert event2.declared_capabilities["write"] is True
