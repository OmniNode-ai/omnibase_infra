# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Validate orchestrator contracts declare event_bus.subscribe_topics.

OMN-1972 Phase 1 established that the runtime reads subscribe_topics from
orchestrator contracts (the event_bus subcontract) rather than hardcoding
topic strings in the kernel. This test ensures the contract structure is
maintained and that all declared topics are realm-agnostic.

Contract Structure Required:
    event_bus:
      subscribe_topics:
        - "onex.evt.platform.node-introspection.v1"
        - ...
      publish_topics:
        - "onex.evt.platform.node-registration.v1"
        - ...

Topics MUST:
    - Start with "onex." (realm-agnostic)
    - NOT contain environment prefixes (dev., prod., staging., etc.)
"""

from pathlib import Path

import pytest
import yaml


class TestOrchestratorContractEventBus:
    """Validate orchestrator contracts declare event_bus.subscribe_topics."""

    @pytest.fixture
    def contract_path(self) -> Path:
        """Return the path to the registration orchestrator contract."""
        path = (
            Path(__file__).resolve().parents[3]
            / "src/omnibase_infra/nodes/node_registration_orchestrator/contract.yaml"
        )
        return path

    @pytest.fixture
    def contract(self, contract_path: Path) -> dict:
        """Load and return the registration orchestrator contract."""
        assert contract_path.exists(), f"Contract not found: {contract_path}"
        with contract_path.open() as f:
            return yaml.safe_load(f)

    def test_registration_orchestrator_has_event_bus(self, contract: dict) -> None:
        """Registration orchestrator contract MUST declare event_bus subcontract."""
        assert "event_bus" in contract, (
            "Registration orchestrator contract MUST declare 'event_bus' subcontract. "
            "Topics must be the runtime source of truth, not hardcoded in kernel."
        )

    def test_registration_orchestrator_has_subscribe_topics(
        self, contract: dict
    ) -> None:
        """event_bus subcontract MUST declare subscribe_topics."""
        event_bus = contract.get("event_bus", {})
        assert "subscribe_topics" in event_bus, (
            "event_bus subcontract MUST declare 'subscribe_topics'."
        )

    def test_subscribe_topics_not_empty(self, contract: dict) -> None:
        """subscribe_topics must not be empty."""
        event_bus = contract.get("event_bus", {})
        topics = event_bus.get("subscribe_topics", [])
        assert len(topics) > 0, "subscribe_topics must not be empty."

    def test_subscribe_topics_are_realm_agnostic(self, contract: dict) -> None:
        """All subscribe_topics must start with 'onex.' (no env prefix)."""
        event_bus = contract.get("event_bus", {})
        topics = event_bus.get("subscribe_topics", [])

        env_prefixes = ("dev.", "prod.", "staging.", "test.", "local.")

        for topic in topics:
            assert topic.startswith("onex."), (
                f"Topic '{topic}' must be realm-agnostic (start with 'onex.'). "
                "Environment prefix is not allowed in contract topics."
            )
            assert not any(topic.startswith(env) for env in env_prefixes), (
                f"Topic '{topic}' has environment prefix - must be realm-agnostic."
            )

    def test_subscribe_topics_do_not_contain_template_placeholders(
        self, contract: dict
    ) -> None:
        """subscribe_topics must not contain {env} or {namespace} placeholders.

        The event_bus subcontract uses concrete realm-agnostic topics, unlike
        consumed_events which uses template placeholders for documentation.
        """
        event_bus = contract.get("event_bus", {})
        topics = event_bus.get("subscribe_topics", [])

        for topic in topics:
            assert "{env}" not in topic, (
                f"Topic '{topic}' contains '{{env}}' placeholder. "
                "event_bus.subscribe_topics must use concrete realm-agnostic topics."
            )
            assert "{namespace}" not in topic, (
                f"Topic '{topic}' contains '{{namespace}}' placeholder. "
                "event_bus.subscribe_topics must use concrete realm-agnostic topics."
            )

    def test_registration_orchestrator_has_publish_topics(self, contract: dict) -> None:
        """Registration orchestrator contract SHOULD declare publish_topics."""
        event_bus = contract.get("event_bus", {})
        assert "publish_topics" in event_bus, (
            "event_bus subcontract SHOULD declare 'publish_topics'."
        )

    def test_publish_topics_are_realm_agnostic(self, contract: dict) -> None:
        """All publish_topics must start with 'onex.' (no env prefix)."""
        event_bus = contract.get("event_bus", {})
        topics = event_bus.get("publish_topics", [])

        env_prefixes = ("dev.", "prod.", "staging.", "test.", "local.")

        for topic in topics:
            assert topic.startswith("onex."), (
                f"Publish topic '{topic}' must be realm-agnostic (start with 'onex.'). "
                "Environment prefix is not allowed in contract topics."
            )
            assert not any(topic.startswith(env) for env in env_prefixes), (
                f"Publish topic '{topic}' has environment prefix - must be realm-agnostic."
            )

    def test_event_bus_has_version(self, contract: dict) -> None:
        """event_bus subcontract SHOULD declare a version for schema tracking."""
        event_bus = contract.get("event_bus", {})
        assert "version" in event_bus, (
            "event_bus subcontract SHOULD declare 'version' for schema tracking."
        )
