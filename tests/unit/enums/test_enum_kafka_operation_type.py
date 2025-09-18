"""Test suite for EnumKafkaOperationType."""

import pytest

from omnibase_infra.enums.enum_kafka_operation_type import EnumKafkaOperationType


class TestEnumKafkaOperationType:
    """Test cases for Kafka Operation Type enumeration."""

    def test_enum_values_exist(self):
        """Test that all expected enum values are defined."""
        expected_values = {
            "PRODUCE", "CONSUME", "TOPIC_CREATE",
            "TOPIC_DELETE", "HEALTH_CHECK", "CONNECTION_TEST"
        }

        actual_values = {item.name for item in EnumKafkaOperationType}
        assert actual_values == expected_values

    def test_enum_string_values(self):
        """Test that enum values have correct string representations."""
        expected_mappings = {
            EnumKafkaOperationType.PRODUCE: "produce",
            EnumKafkaOperationType.CONSUME: "consume",
            EnumKafkaOperationType.TOPIC_CREATE: "topic_create",
            EnumKafkaOperationType.TOPIC_DELETE: "topic_delete",
            EnumKafkaOperationType.HEALTH_CHECK: "health_check",
            EnumKafkaOperationType.CONNECTION_TEST: "connection_test",
        }

        for enum_item, expected_value in expected_mappings.items():
            assert enum_item.value == expected_value

    def test_enum_inheritance(self):
        """Test that enum inherits from str and Enum correctly."""
        assert isinstance(EnumKafkaOperationType.PRODUCE, str)
        assert EnumKafkaOperationType.PRODUCE.value == "produce"

        # Test enum functionality
        assert EnumKafkaOperationType("produce") == EnumKafkaOperationType.PRODUCE

    @pytest.mark.parametrize("operation", [
        EnumKafkaOperationType.PRODUCE,
        EnumKafkaOperationType.CONSUME,
        EnumKafkaOperationType.TOPIC_CREATE,
        EnumKafkaOperationType.TOPIC_DELETE,
        EnumKafkaOperationType.HEALTH_CHECK,
        EnumKafkaOperationType.CONNECTION_TEST,
    ])
    def test_all_operations_are_strings(self, operation):
        """Test that all operation types are valid strings."""
        assert isinstance(operation, str)
        assert len(operation.value) > 0
        assert operation.value == operation.value.lower()  # lowercase convention

    def test_operational_categories(self):
        """Test that operations can be categorized logically."""
        data_operations = {
            EnumKafkaOperationType.PRODUCE,
            EnumKafkaOperationType.CONSUME
        }

        admin_operations = {
            EnumKafkaOperationType.TOPIC_CREATE,
            EnumKafkaOperationType.TOPIC_DELETE
        }

        health_operations = {
            EnumKafkaOperationType.HEALTH_CHECK,
            EnumKafkaOperationType.CONNECTION_TEST
        }

        all_operations = data_operations | admin_operations | health_operations
        enum_values = set(EnumKafkaOperationType)

        assert all_operations == enum_values