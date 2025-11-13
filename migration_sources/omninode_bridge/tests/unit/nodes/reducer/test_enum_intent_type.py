#!/usr/bin/env python3
"""Unit tests for EnumIntentType.

Tests cover:
- Enum value definitions
- Target node type mapping
- Effect node requirement checking
- Enum validation
"""

import pytest

from omninode_bridge.nodes.reducer.v1_0_0.models.enum_intent_type import EnumIntentType


class TestEnumIntentType:
    """Test suite for EnumIntentType."""

    def test_enum_values(self):
        """Test that all expected enum values are defined."""
        assert EnumIntentType.PUBLISH_EVENT.value == "PublishEvent"
        assert EnumIntentType.PERSIST_STATE.value == "PersistState"
        assert EnumIntentType.PERSIST_FSM_TRANSITION.value == "PersistFSMTransition"
        assert EnumIntentType.RECOVER_FSM_STATES.value == "RecoverFSMStates"

    def test_enum_inheritance(self):
        """Test enum inherits from str."""
        assert isinstance(EnumIntentType.PUBLISH_EVENT, str)

    def test_enum_string_behavior(self):
        """Test enum behaves like string."""
        assert EnumIntentType.PUBLISH_EVENT.value == "PublishEvent"
        assert EnumIntentType.PUBLISH_EVENT == "PublishEvent"

    def test_enum_iteration(self):
        """Test enum iteration."""
        all_intents = list(EnumIntentType)
        assert len(all_intents) == 4  # Total number of intent types
        assert EnumIntentType.PUBLISH_EVENT in all_intents
        assert EnumIntentType.PERSIST_STATE in all_intents

    def test_get_target_node_type_publish_event(self):
        """Test target node type for PUBLISH_EVENT."""
        target = EnumIntentType.PUBLISH_EVENT.get_target_node_type()
        assert target == "event_bus"

    def test_get_target_node_type_persist_state(self):
        """Test target node type for PERSIST_STATE."""
        target = EnumIntentType.PERSIST_STATE.get_target_node_type()
        assert target == "store_effect"

    def test_get_target_node_type_persist_fsm_transition(self):
        """Test target node type for PERSIST_FSM_TRANSITION."""
        target = EnumIntentType.PERSIST_FSM_TRANSITION.get_target_node_type()
        assert target == "store_effect"

    def test_get_target_node_type_recover_fsm_states(self):
        """Test target node type for RECOVER_FSM_STATES."""
        target = EnumIntentType.RECOVER_FSM_STATES.get_target_node_type()
        assert target == "store_effect"

    def test_get_target_node_type_all_intents(self):
        """Test that all intents have a target node type mapping."""
        for intent in EnumIntentType:
            target = intent.get_target_node_type()
            assert target in ("event_bus", "store_effect")

    def test_requires_effect_node(self):
        """Test that all intents require effect nodes."""
        # All intents should require effect nodes
        assert EnumIntentType.PUBLISH_EVENT.requires_effect_node is True
        assert EnumIntentType.PERSIST_STATE.requires_effect_node is True
        assert EnumIntentType.PERSIST_FSM_TRANSITION.requires_effect_node is True
        assert EnumIntentType.RECOVER_FSM_STATES.requires_effect_node is True

    def test_requires_effect_node_all_intents(self):
        """Test that all intents require effect nodes."""
        for intent in EnumIntentType:
            assert intent.requires_effect_node is True

    def test_enum_serialization(self):
        """Test enum serialization to string."""
        intent = EnumIntentType.PUBLISH_EVENT
        # Can be serialized to string via .value
        serialized = intent.value
        assert serialized == "PublishEvent"

        # Can be compared with string
        assert intent == serialized

    def test_enum_comparison(self):
        """Test enum comparison operations."""
        intent1 = EnumIntentType.PUBLISH_EVENT
        intent2 = EnumIntentType.PUBLISH_EVENT
        intent3 = EnumIntentType.PERSIST_STATE

        # Same intents are equal
        assert intent1 == intent2
        assert intent1 is intent2  # Same instance

        # Different intents are not equal
        assert intent1 != intent3

    def test_enum_invalid_value(self):
        """Test accessing invalid enum value raises error."""
        with pytest.raises(ValueError):
            EnumIntentType("InvalidIntent")

    def test_enum_docstring(self):
        """Test enum has docstring."""
        assert EnumIntentType.__doc__ is not None
        assert "Intent types" in EnumIntentType.__doc__

    def test_enum_value_docstrings(self):
        """Test enum values have docstrings."""
        # Check a few key intents have docstrings
        assert EnumIntentType.PUBLISH_EVENT.__doc__ is not None
        assert EnumIntentType.PERSIST_STATE.__doc__ is not None

    def test_enum_membership(self):
        """Test enum membership."""
        assert EnumIntentType.PUBLISH_EVENT == "PublishEvent"
        assert EnumIntentType.PUBLISH_EVENT != "InvalidIntent"

    def test_target_node_type_mapping_consistency(self):
        """Test that target node type mapping is consistent with intent purpose."""
        # Event publishing should target event_bus
        assert EnumIntentType.PUBLISH_EVENT.get_target_node_type() == "event_bus"

        # State persistence should target store_effect
        assert EnumIntentType.PERSIST_STATE.get_target_node_type() == "store_effect"
        assert (
            EnumIntentType.PERSIST_FSM_TRANSITION.get_target_node_type()
            == "store_effect"
        )
        assert (
            EnumIntentType.RECOVER_FSM_STATES.get_target_node_type() == "store_effect"
        )
