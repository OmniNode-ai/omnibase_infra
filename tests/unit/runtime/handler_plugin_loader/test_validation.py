# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for HandlerPluginLoader validation logic.

Part of OMN-1132: Handler Plugin Loader implementation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import (
    VALID_HANDLER_CONTRACT_YAML,
    MockInvalidHandler,
    MockValidHandler,
)


class TestHandlerPluginLoaderValidation:
    """Tests for handler validation logic."""

    def test_validate_handler_implements_protocol(self) -> None:
        """Test protocol validation for handler with describe method."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        # MockValidHandler has describe method
        assert loader._validate_handler_protocol(MockValidHandler) is True

    def test_validate_handler_without_describe_method(self) -> None:
        """Test protocol validation for handler without describe method."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        # MockInvalidHandler doesn't have describe method
        assert loader._validate_handler_protocol(MockInvalidHandler) is False

    def test_validate_non_callable_describe_rejected(self) -> None:
        """Test that non-callable describe attribute is rejected."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        # Create a class with describe as an attribute, not a method
        class HandlerWithNonCallableDescribe:
            describe = "not a method"

        result = loader._validate_handler_protocol(HandlerWithNonCallableDescribe)
        assert result is False

    def test_capability_tags_extracted_correctly(self, tmp_path: Path) -> None:
        """Test that capability tags are extracted from contract."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            VALID_HANDLER_CONTRACT_YAML.format(
                handler_name="tagged.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
                handler_type="compute",
                tag1="database",
                tag2="caching",
            )
        )

        loader = HandlerPluginLoader()
        handler = loader.load_from_contract(contract_file)

        assert "database" in handler.capability_tags
        assert "caching" in handler.capability_tags
        assert len(handler.capability_tags) == 2

    def test_handler_type_categories_parsed_correctly(self, tmp_path: Path) -> None:
        """Test that handler_type values map to correct enum values."""
        from omnibase_infra.enums import EnumHandlerTypeCategory
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        test_cases = [
            ("compute", EnumHandlerTypeCategory.COMPUTE),
            ("effect", EnumHandlerTypeCategory.EFFECT),
            (
                "nondeterministic_compute",
                EnumHandlerTypeCategory.NONDETERMINISTIC_COMPUTE,
            ),
            # Case insensitive
            ("COMPUTE", EnumHandlerTypeCategory.COMPUTE),
            ("Effect", EnumHandlerTypeCategory.EFFECT),
        ]

        for handler_type_str, expected_enum in test_cases:
            contract_file = tmp_path / f"handler_{handler_type_str}.yaml"
            contract_file.write_text(
                f"""
handler_name: test.handler.{handler_type_str}
handler_class: tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler
handler_type: {handler_type_str}
"""
            )

            handler = loader.load_from_contract(contract_file)
            assert handler.handler_type == expected_enum, (
                f"Expected {expected_enum} for '{handler_type_str}', "
                f"got {handler.handler_type}"
            )

    def test_invalid_handler_type_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid handler_type value raises error."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            """
handler_name: invalid.type.handler
handler_class: test.handlers.TestHandler
handler_type: invalid_type
"""
        )

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        # Verify error message indicates invalid handler type
        error_msg = str(exc_info.value).lower()
        assert "invalid" in error_msg or "type" in error_msg

    def test_missing_handler_type_raises_error(self, tmp_path: Path) -> None:
        """Test that missing handler_type raises error (handler_type is required)."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            """
handler_name: no.type.handler
handler_class: tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler
"""
        )

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        # Verify error message indicates missing handler_type
        error_msg = str(exc_info.value).lower()
        assert "handler_type" in error_msg

    def test_whitespace_only_handler_type_raises_error(self, tmp_path: Path) -> None:
        """Test that whitespace-only handler_type raises error.

        PR #134 feedback: Whitespace-only strings like '   ' would pass
        isinstance check but fail with unclear error during lookup.
        """
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            """
handler_name: whitespace.type.handler
handler_class: tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler
handler_type: "   "
"""
        )

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        # Verify error message indicates non-empty requirement
        error_msg = str(exc_info.value).lower()
        assert "non-empty" in error_msg

    def test_empty_string_handler_type_raises_error(self, tmp_path: Path) -> None:
        """Test that empty string handler_type raises error."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            """
handler_name: empty.type.handler
handler_class: tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler
handler_type: ""
"""
        )

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        # Verify error message indicates non-empty requirement
        error_msg = str(exc_info.value).lower()
        assert "non-empty" in error_msg
