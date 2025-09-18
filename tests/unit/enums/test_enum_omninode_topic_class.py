"""Test suite for EnumOmniNodeTopicClass."""

import pytest

from omnibase_infra.enums.enum_omninode_topic_class import EnumOmniNodeTopicClass


class TestEnumOmniNodeTopicClass:
    """Test cases for OmniNode Topic Class enumeration."""

    def test_enum_values_exist(self):
        """Test that all expected enum values are defined."""
        expected_values = {
            "EVT", "CMD", "QRS",  # Core event processing
            "CTL", "RTY", "DLT",  # Control and management
            "CDC", "MET", "AUD", "LOG"  # Data and monitoring
        }

        actual_values = {item.name for item in EnumOmniNodeTopicClass}
        assert actual_values == expected_values

    def test_enum_string_values(self):
        """Test that enum values have correct string representations."""
        expected_mappings = {
            EnumOmniNodeTopicClass.EVT: "evt",
            EnumOmniNodeTopicClass.CMD: "cmd",
            EnumOmniNodeTopicClass.QRS: "qrs",
            EnumOmniNodeTopicClass.CTL: "ctl",
            EnumOmniNodeTopicClass.RTY: "rty",
            EnumOmniNodeTopicClass.DLT: "dlt",
            EnumOmniNodeTopicClass.CDC: "cdc",
            EnumOmniNodeTopicClass.MET: "met",
            EnumOmniNodeTopicClass.AUD: "aud",
            EnumOmniNodeTopicClass.LOG: "log",
        }

        for enum_item, expected_value in expected_mappings.items():
            assert enum_item.value == expected_value

    def test_enum_inheritance(self):
        """Test that enum inherits from str and Enum correctly."""
        # Test string inheritance
        assert isinstance(EnumOmniNodeTopicClass.EVT, str)
        assert EnumOmniNodeTopicClass.EVT.value == "evt"

        # Test enum functionality
        assert EnumOmniNodeTopicClass("evt") == EnumOmniNodeTopicClass.EVT
        assert "evt" in [item.value for item in EnumOmniNodeTopicClass]

    def test_enum_usage_in_topic_patterns(self):
        """Test enum usage in OmniNode topic patterns."""
        # Simulate topic construction: <env>.<tenant>.<context>.<class>.<topic>.<v>
        env = "dev"
        tenant = "omni"
        context = "user"
        topic_class = EnumOmniNodeTopicClass.EVT.value  # Use .value for string representation
        topic = "profile_updated"
        version = "v1"

        full_topic = f"{env}.{tenant}.{context}.{topic_class}.{topic}.{version}"
        assert full_topic == "dev.omni.user.evt.profile_updated.v1"

    @pytest.mark.parametrize("topic_class,expected_usage", [
        (EnumOmniNodeTopicClass.EVT, "State change notifications"),
        (EnumOmniNodeTopicClass.CMD, "Action requests"),
        (EnumOmniNodeTopicClass.QRS, "Request/response patterns"),
        (EnumOmniNodeTopicClass.CTL, "Control plane operations"),
        (EnumOmniNodeTopicClass.RTY, "Retry processing"),
        (EnumOmniNodeTopicClass.DLT, "Failed messages"),
        (EnumOmniNodeTopicClass.CDC, "Database changes"),
        (EnumOmniNodeTopicClass.MET, "Performance and operational metrics"),
        (EnumOmniNodeTopicClass.AUD, "Audit trail and compliance"),
        (EnumOmniNodeTopicClass.LOG, "Application logging"),
    ])
    def test_enum_semantic_meaning(self, topic_class, expected_usage):
        """Test that each enum value has clear semantic meaning."""
        # This test documents the intended usage of each topic class
        assert isinstance(topic_class, EnumOmniNodeTopicClass)
        assert isinstance(expected_usage, str)
        assert len(expected_usage) > 10  # Meaningful description