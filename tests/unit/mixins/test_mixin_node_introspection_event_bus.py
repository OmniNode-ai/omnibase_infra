# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for MixinNodeIntrospection event_bus extraction (OMN-1613).

This test suite validates:
- Event bus configuration extraction from contracts
- Topic suffix resolution to full realm-agnostic topics
- Fail-fast behavior on unresolved placeholders
- Graceful handling of missing contract/event_bus configurations

Test Organization:
    - TestExtractEventBusConfig: Direct method tests for _extract_event_bus_config()
    - TestEventBusInGetIntrospectionData: Integration with get_introspection_data()
"""

import os
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.mixins.mixin_node_introspection import MixinNodeIntrospection
from omnibase_infra.models.discovery import ModelIntrospectionConfig
from omnibase_infra.models.registration.model_node_event_bus_config import (
    ModelEventBusTopicEntry,
    ModelNodeEventBusConfig,
)

# Test UUIDs - use deterministic values for reproducible tests
TEST_NODE_UUID = UUID("00000000-0000-0000-0000-000000000001")


class MockEventBusSubcontract:
    """Mock event_bus subcontract for testing."""

    def __init__(
        self,
        publish_topics: list[str] | None = None,
        subscribe_topics: list[str] | None = None,
    ) -> None:
        self.publish_topics = publish_topics or []
        self.subscribe_topics = subscribe_topics or []


class MockContract:
    """Mock contract for testing event_bus extraction.

    Includes minimal fields required by ContractCapabilityExtractor:
    - node_type: Required for contract type extraction
    - contract_version: Required for version extraction
    """

    def __init__(self, event_bus: MockEventBusSubcontract | None = None) -> None:
        # Import here to avoid circular imports at module level
        from omnibase_core.models.primitives.model_semver import ModelSemVer

        self.event_bus = event_bus
        # Required by ContractCapabilityExtractor._extract_contract_type()
        self.node_type = "EFFECT_GENERIC"
        # Required by ContractCapabilityExtractor._extract_version()
        self.contract_version = ModelSemVer(major=1, minor=0, patch=0)


class TestableNode(MixinNodeIntrospection):
    """Minimal testable node for mixin testing."""

    def __init__(self) -> None:
        # Don't call super().__init__() - we'll manually initialize the mixin
        pass


class TestExtractEventBusConfig:
    """Tests for _extract_event_bus_config() method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.node = TestableNode()
        # Initialize required mixin attributes
        self.node._introspection_contract = None
        self.node._introspection_initialized = True

    def test_returns_none_when_no_contract(self) -> None:
        """Returns None when contract is not configured."""
        self.node._introspection_contract = None

        result = self.node._extract_event_bus_config("dev")

        assert result is None

    def test_returns_none_when_no_event_bus_subcontract(self) -> None:
        """Returns None when contract has no event_bus subcontract."""
        self.node._introspection_contract = MockContract(event_bus=None)

        result = self.node._extract_event_bus_config("dev")

        assert result is None

    def test_returns_none_when_empty_topics(self) -> None:
        """Returns None when event_bus has empty topic lists."""
        event_bus = MockEventBusSubcontract(
            publish_topics=[],
            subscribe_topics=[],
        )
        self.node._introspection_contract = MockContract(event_bus=event_bus)

        result = self.node._extract_event_bus_config("dev")

        assert result is None

    def test_extracts_publish_topics(self) -> None:
        """Resolves publish topic suffixes to full realm-agnostic topics."""
        event_bus = MockEventBusSubcontract(
            publish_topics=["onex.evt.platform.node-registered.v1"],
            subscribe_topics=[],
        )
        self.node._introspection_contract = MockContract(event_bus=event_bus)

        result = self.node._extract_event_bus_config("dev")

        assert result is not None
        assert len(result.publish_topics) == 1
        assert result.publish_topics[0].topic == "onex.evt.platform.node-registered.v1"
        assert len(result.subscribe_topics) == 0

    def test_extracts_subscribe_topics(self) -> None:
        """Resolves subscribe topic suffixes to full realm-agnostic topics."""
        event_bus = MockEventBusSubcontract(
            publish_topics=[],
            subscribe_topics=["onex.evt.platform.intent-classified.v1"],
        )
        self.node._introspection_contract = MockContract(event_bus=event_bus)

        result = self.node._extract_event_bus_config("dev")

        assert result is not None
        assert len(result.subscribe_topics) == 1
        assert (
            result.subscribe_topics[0].topic == "onex.evt.platform.intent-classified.v1"
        )
        assert len(result.publish_topics) == 0

    def test_extracts_multiple_topics(self) -> None:
        """Handles multiple topics in both lists."""
        event_bus = MockEventBusSubcontract(
            publish_topics=[
                "onex.evt.platform.node-registered.v1",
                "onex.cmd.platform.node-shutdown.v1",
            ],
            subscribe_topics=[
                "onex.evt.platform.intent-classified.v1",
                "onex.cmd.platform.register-node.v1",
            ],
        )
        self.node._introspection_contract = MockContract(event_bus=event_bus)

        result = self.node._extract_event_bus_config("prod")

        assert result is not None
        assert len(result.publish_topics) == 2
        assert result.publish_topics[0].topic == "onex.evt.platform.node-registered.v1"
        assert result.publish_topics[1].topic == "onex.cmd.platform.node-shutdown.v1"
        assert len(result.subscribe_topics) == 2
        assert (
            result.subscribe_topics[0].topic == "onex.evt.platform.intent-classified.v1"
        )
        assert result.subscribe_topics[1].topic == "onex.cmd.platform.register-node.v1"

    def test_topics_are_realm_agnostic(self) -> None:
        """Topics are the same regardless of environment parameter (realm-agnostic)."""
        event_bus = MockEventBusSubcontract(
            publish_topics=["onex.evt.platform.test-event.v1"],
        )
        self.node._introspection_contract = MockContract(event_bus=event_bus)

        dev_result = self.node._extract_event_bus_config("dev")
        staging_result = self.node._extract_event_bus_config("staging")
        prod_result = self.node._extract_event_bus_config("prod")

        # All environments produce the same realm-agnostic topic
        assert dev_result.publish_topics[0].topic == "onex.evt.platform.test-event.v1"
        assert (
            staging_result.publish_topics[0].topic == "onex.evt.platform.test-event.v1"
        )
        assert prod_result.publish_topics[0].topic == "onex.evt.platform.test-event.v1"

    def test_fail_fast_on_unresolved_env_placeholder(self) -> None:
        """Raises ValueError on unresolved {env} placeholder."""
        event_bus = MockEventBusSubcontract(
            publish_topics=["{env}.onex.evt.test.v1"],  # Unresolved placeholder
        )
        self.node._introspection_contract = MockContract(event_bus=event_bus)

        with pytest.raises(ValueError) as exc_info:
            self.node._extract_event_bus_config("dev")

        assert "Unresolved placeholder in topic" in str(exc_info.value)
        assert "{env}" in str(exc_info.value)

    def test_fail_fast_on_unresolved_namespace_placeholder(self) -> None:
        """Raises ValueError on unresolved {namespace} placeholder."""
        event_bus = MockEventBusSubcontract(
            subscribe_topics=["onex.{namespace}.evt.test.v1"],  # Unresolved placeholder
        )
        self.node._introspection_contract = MockContract(event_bus=event_bus)

        with pytest.raises(ValueError) as exc_info:
            self.node._extract_event_bus_config("dev")

        assert "Unresolved placeholder in topic" in str(exc_info.value)
        assert "{namespace}" in str(exc_info.value)

    def test_fail_fast_on_any_curly_brace_placeholder(self) -> None:
        """Raises ValueError on any unresolved placeholder with curly braces."""
        event_bus = MockEventBusSubcontract(
            publish_topics=["onex.evt.{custom_var}.v1"],
        )
        self.node._introspection_contract = MockContract(event_bus=event_bus)

        with pytest.raises(ValueError) as exc_info:
            self.node._extract_event_bus_config("dev")

        assert "Unresolved placeholder in topic" in str(exc_info.value)

    def test_topic_entry_has_correct_defaults(self) -> None:
        """Topic entries have correct default metadata values."""
        event_bus = MockEventBusSubcontract(
            publish_topics=["onex.evt.platform.test-event.v1"],
        )
        self.node._introspection_contract = MockContract(event_bus=event_bus)

        result = self.node._extract_event_bus_config("dev")

        entry = result.publish_topics[0]
        assert entry.topic == "onex.evt.platform.test-event.v1"
        assert entry.event_type is None  # Default
        assert entry.message_category == "EVENT"  # Default
        assert entry.description is None  # Default

    def test_result_model_is_frozen(self) -> None:
        """Result ModelNodeEventBusConfig is immutable."""
        event_bus = MockEventBusSubcontract(
            publish_topics=["onex.evt.platform.test-event.v1"],
        )
        self.node._introspection_contract = MockContract(event_bus=event_bus)

        result = self.node._extract_event_bus_config("dev")

        # Frozen model should raise on modification
        with pytest.raises(Exception):  # Pydantic raises ValidationError
            result.publish_topics = []

    def test_handles_none_topic_lists_gracefully(self) -> None:
        """Handles None topic lists from subcontract (not just empty lists)."""

        class EventBusWithNoneTopics:
            publish_topics = None
            subscribe_topics = None

        self.node._introspection_contract = MockContract(
            event_bus=EventBusWithNoneTopics()
        )

        result = self.node._extract_event_bus_config("dev")

        assert result is None


class TestEventBusInGetIntrospectionData:
    """Tests for event_bus integration in get_introspection_data()."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.node = TestableNode()

    def _initialize_node(self, contract: MockContract | None = None) -> None:
        """Initialize node with mixin configuration.

        Note: contract is injected directly to _introspection_contract after
        initialization because ModelIntrospectionConfig validates contract
        against ModelContractBase which requires many fields.
        """
        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID,
            node_type=EnumNodeKind.EFFECT,
            node_name="test_event_bus_node",
            version="1.0.0",
        )
        self.node.initialize_introspection(config)
        # Inject contract directly to bypass ModelContractBase validation
        self.node._introspection_contract = contract

    @pytest.mark.asyncio
    async def test_event_bus_populated_from_contract(self) -> None:
        """get_introspection_data() populates event_bus from contract."""
        event_bus = MockEventBusSubcontract(
            publish_topics=["onex.evt.platform.node-registered.v1"],
            subscribe_topics=["onex.evt.platform.intent-classified.v1"],
        )
        contract = MockContract(event_bus=event_bus)
        self._initialize_node(contract)

        with patch.dict(os.environ, {"ONEX_ENV": "dev"}):
            event = await self.node.get_introspection_data()

        assert event.event_bus is not None
        assert len(event.event_bus.publish_topics) == 1
        assert (
            event.event_bus.publish_topics[0].topic
            == "onex.evt.platform.node-registered.v1"
        )
        assert len(event.event_bus.subscribe_topics) == 1
        assert (
            event.event_bus.subscribe_topics[0].topic
            == "onex.evt.platform.intent-classified.v1"
        )

    @pytest.mark.asyncio
    async def test_event_bus_none_when_no_contract(self) -> None:
        """get_introspection_data() sets event_bus to None when no contract."""
        self._initialize_node(contract=None)

        event = await self.node.get_introspection_data()

        assert event.event_bus is None

    @pytest.mark.asyncio
    async def test_event_bus_none_when_no_topics(self) -> None:
        """get_introspection_data() sets event_bus to None when no topics."""
        event_bus = MockEventBusSubcontract(
            publish_topics=[],
            subscribe_topics=[],
        )
        contract = MockContract(event_bus=event_bus)
        self._initialize_node(contract)

        event = await self.node.get_introspection_data()

        assert event.event_bus is None

    @pytest.mark.asyncio
    async def test_topics_are_realm_agnostic_regardless_of_onex_env(self) -> None:
        """get_introspection_data() produces realm-agnostic topics regardless of ONEX_ENV."""
        event_bus = MockEventBusSubcontract(
            publish_topics=["onex.evt.platform.test-event.v1"],
        )
        contract = MockContract(event_bus=event_bus)
        self._initialize_node(contract)

        with patch.dict(os.environ, {"ONEX_ENV": "production"}):
            event = await self.node.get_introspection_data()

        assert (
            event.event_bus.publish_topics[0].topic == "onex.evt.platform.test-event.v1"
        )

    @pytest.mark.asyncio
    async def test_produces_realm_agnostic_topics_when_onex_env_not_set(self) -> None:
        """get_introspection_data() produces realm-agnostic topics when ONEX_ENV not set."""
        event_bus = MockEventBusSubcontract(
            publish_topics=["onex.evt.platform.test-event.v1"],
        )
        contract = MockContract(event_bus=event_bus)
        self._initialize_node(contract)

        # Ensure ONEX_ENV is not set
        env = os.environ.copy()
        env.pop("ONEX_ENV", None)
        with patch.dict(os.environ, env, clear=True):
            event = await self.node.get_introspection_data()

        assert (
            event.event_bus.publish_topics[0].topic == "onex.evt.platform.test-event.v1"
        )

    @pytest.mark.asyncio
    async def test_raises_on_unresolved_placeholder(self) -> None:
        """get_introspection_data() raises ProtocolConfigurationError on unresolved placeholder.

        The internal ValueError is wrapped in ProtocolConfigurationError because
        configuration issues should use the proper error type from the error hierarchy.
        """
        event_bus = MockEventBusSubcontract(
            publish_topics=["{env}.onex.evt.test.v1"],
        )
        contract = MockContract(event_bus=event_bus)
        self._initialize_node(contract)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await self.node.get_introspection_data()

        assert "Event bus extraction failed" in str(exc_info.value)
        assert "Unresolved placeholder" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_event_bus_topic_strings_property(self) -> None:
        """event_bus.publish_topic_strings returns list of topic strings."""
        event_bus = MockEventBusSubcontract(
            publish_topics=[
                "onex.evt.platform.node-registered.v1",
                "onex.cmd.platform.node-shutdown.v1",
            ],
        )
        contract = MockContract(event_bus=event_bus)
        self._initialize_node(contract)

        with patch.dict(os.environ, {"ONEX_ENV": "dev"}):
            event = await self.node.get_introspection_data()

        # Use the convenience property for routing lookups
        topic_strings = event.event_bus.publish_topic_strings
        assert topic_strings == [
            "onex.evt.platform.node-registered.v1",
            "onex.cmd.platform.node-shutdown.v1",
        ]


class TestEventBusEdgeCases:
    """Edge case tests for event_bus extraction."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.node = TestableNode()
        self.node._introspection_contract = None
        self.node._introspection_initialized = True

    def test_contract_without_event_bus_attribute(self) -> None:
        """Handles contracts that don't have event_bus attribute at all."""

        class ContractWithoutEventBus:
            pass

        self.node._introspection_contract = ContractWithoutEventBus()

        result = self.node._extract_event_bus_config("dev")

        assert result is None

    def test_empty_string_env_produces_realm_agnostic_topic(self) -> None:
        """Handles empty string env - produces realm-agnostic topic."""
        event_bus = MockEventBusSubcontract(
            publish_topics=["onex.evt.platform.test-event.v1"],
        )
        self.node._introspection_contract = MockContract(event_bus=event_bus)

        result = self.node._extract_event_bus_config("")

        # Realm-agnostic topic without any prefix
        assert result.publish_topics[0].topic == "onex.evt.platform.test-event.v1"

    def test_topic_suffix_with_special_characters(self) -> None:
        """Handles topic suffixes with hyphens in event name (kebab-case)."""
        event_bus = MockEventBusSubcontract(
            publish_topics=["onex.evt.platform.node-state-changed.v1"],
        )
        self.node._introspection_contract = MockContract(event_bus=event_bus)

        result = self.node._extract_event_bus_config("dev")

        assert (
            result.publish_topics[0].topic == "onex.evt.platform.node-state-changed.v1"
        )

    def test_topic_suffix_with_numbers(self) -> None:
        """Handles topic suffixes with version numbers."""
        event_bus = MockEventBusSubcontract(
            publish_topics=[
                "onex.evt.platform.test-event.v2",
                "onex.evt.platform.test-event.v10",
            ],
        )
        self.node._introspection_contract = MockContract(event_bus=event_bus)

        result = self.node._extract_event_bus_config("dev")

        assert result.publish_topics[0].topic == "onex.evt.platform.test-event.v2"
        assert result.publish_topics[1].topic == "onex.evt.platform.test-event.v10"


__all__ = [
    "TestExtractEventBusConfig",
    "TestEventBusInGetIntrospectionData",
    "TestEventBusEdgeCases",
]
