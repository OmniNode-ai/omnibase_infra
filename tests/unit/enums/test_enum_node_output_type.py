# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for EnumNodeOutputType helper methods.

Tests the to_message_category() and is_routable() methods added to
EnumNodeOutputType for OMN-974.
"""

import pytest

from omnibase_infra.enums import EnumMessageCategory, EnumNodeOutputType
from omnibase_infra.errors import ProtocolConfigurationError


class TestIsRoutable:
    """Tests for EnumNodeOutputType.is_routable() method."""

    def test_event_is_routable(self) -> None:
        """EVENT output type should be routable."""
        assert EnumNodeOutputType.EVENT.is_routable() is True

    def test_command_is_routable(self) -> None:
        """COMMAND output type should be routable."""
        assert EnumNodeOutputType.COMMAND.is_routable() is True

    def test_intent_is_routable(self) -> None:
        """INTENT output type should be routable."""
        assert EnumNodeOutputType.INTENT.is_routable() is True

    def test_projection_is_not_routable(self) -> None:
        """PROJECTION output type should NOT be routable."""
        assert EnumNodeOutputType.PROJECTION.is_routable() is False

    def test_all_routable_types(self) -> None:
        """Verify exactly which types are routable."""
        routable = [t for t in EnumNodeOutputType if t.is_routable()]
        non_routable = [t for t in EnumNodeOutputType if not t.is_routable()]

        assert set(routable) == {
            EnumNodeOutputType.EVENT,
            EnumNodeOutputType.COMMAND,
            EnumNodeOutputType.INTENT,
        }
        assert set(non_routable) == {EnumNodeOutputType.PROJECTION}


class TestToMessageCategory:
    """Tests for EnumNodeOutputType.to_message_category() method."""

    def test_event_to_message_category(self) -> None:
        """EVENT should convert to EnumMessageCategory.EVENT."""
        result = EnumNodeOutputType.EVENT.to_message_category()
        assert result == EnumMessageCategory.EVENT
        assert isinstance(result, EnumMessageCategory)

    def test_command_to_message_category(self) -> None:
        """COMMAND should convert to EnumMessageCategory.COMMAND."""
        result = EnumNodeOutputType.COMMAND.to_message_category()
        assert result == EnumMessageCategory.COMMAND
        assert isinstance(result, EnumMessageCategory)

    def test_intent_to_message_category(self) -> None:
        """INTENT should convert to EnumMessageCategory.INTENT."""
        result = EnumNodeOutputType.INTENT.to_message_category()
        assert result == EnumMessageCategory.INTENT
        assert isinstance(result, EnumMessageCategory)

    def test_projection_raises_configuration_error(self) -> None:
        """PROJECTION should raise ProtocolConfigurationError - no message category equivalent."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            EnumNodeOutputType.PROJECTION.to_message_category()

        assert "PROJECTION has no message category" in str(exc_info.value)
        assert "not routable" in str(exc_info.value)

    def test_all_routable_types_convert_correctly(self) -> None:
        """All routable types should convert to matching message categories."""
        mapping = {
            EnumNodeOutputType.EVENT: EnumMessageCategory.EVENT,
            EnumNodeOutputType.COMMAND: EnumMessageCategory.COMMAND,
            EnumNodeOutputType.INTENT: EnumMessageCategory.INTENT,
        }

        for output_type, expected_category in mapping.items():
            result = output_type.to_message_category()
            assert result == expected_category
            # Verify string values match
            assert output_type.value == expected_category.value


class TestIsRoutableAndToMessageCategoryConsistency:
    """Tests ensuring is_routable() and to_message_category() are consistent."""

    def test_routable_types_can_convert(self) -> None:
        """All routable types should successfully convert to message category."""
        for output_type in EnumNodeOutputType:
            if output_type.is_routable():
                # Should not raise
                result = output_type.to_message_category()
                assert isinstance(result, EnumMessageCategory)

    def test_non_routable_types_raise_on_convert(self) -> None:
        """All non-routable types should raise ProtocolConfigurationError on conversion."""
        for output_type in EnumNodeOutputType:
            if not output_type.is_routable():
                with pytest.raises(ProtocolConfigurationError):
                    output_type.to_message_category()

    def test_consistency_between_methods(self) -> None:
        """is_routable() and to_message_category() should be consistent."""
        for output_type in EnumNodeOutputType:
            if output_type.is_routable():
                # Should succeed
                output_type.to_message_category()
            else:
                # Should fail
                with pytest.raises(ProtocolConfigurationError):
                    output_type.to_message_category()
