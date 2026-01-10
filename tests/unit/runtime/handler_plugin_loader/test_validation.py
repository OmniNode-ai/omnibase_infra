# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for HandlerPluginLoader validation logic.

Part of OMN-1132: Handler Plugin Loader implementation.

Design Note (PR #134 review):
    Contract file creation is intentionally inline rather than extracted to a helper.
    Each test requires slightly different YAML content (missing fields, invalid values,
    whitespace-only values, etc.), and the 2-line creation pattern (path + write_text)
    is clear and compact. Inline YAML makes each test's validation target immediately
    visible without requiring readers to trace through helper abstractions.
    pytest's tmp_path fixture handles cleanup automatically.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import (
    VALID_HANDLER_CONTRACT_YAML,
    MockInvalidHandler,
    MockPartialHandler,
    MockValidHandler,
)


class TestHandlerPluginLoaderValidation:
    """Tests for handler validation logic.

    Protocol Validation Requirements:
        The handler plugin loader validates handlers against ProtocolHandler
        from omnibase_spi.protocols.handlers.protocol_handler. A valid handler
        must implement all 5 required methods:

        - handler_type (property): Returns handler type identifier string
        - initialize(config): Async method to initialize connections/pools
        - shutdown(timeout_seconds): Async method to release resources
        - execute(request, operation_config): Async method for operations
        - describe(): Sync method returning handler metadata/capabilities

        Note: health_check() is part of the protocol but is optional because
        existing handlers (HandlerHttp, HandlerDb, etc.) do not implement it.
    """

    def test_validate_handler_implements_protocol(self) -> None:
        """Test protocol validation for handler with all required methods.

        MockValidHandler implements all 5 required protocol methods:
        handler_type, initialize, shutdown, execute, describe.
        """
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        # MockValidHandler has all 5 required methods
        is_valid, missing_methods = loader._validate_handler_protocol(MockValidHandler)
        assert is_valid is True
        assert missing_methods == []

    def test_validate_handler_without_any_protocol_methods(self) -> None:
        """Test protocol validation for handler without any protocol methods."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        # MockInvalidHandler has no protocol methods at all
        is_valid, missing_methods = loader._validate_handler_protocol(
            MockInvalidHandler
        )
        assert is_valid is False
        # Should be missing all 5 required methods
        assert set(missing_methods) == {
            "handler_type",
            "initialize",
            "shutdown",
            "execute",
            "describe",
        }

    def test_validate_partial_handler_rejected(self) -> None:
        """Test that handler with only describe() is rejected.

        Ensures validation checks for ALL required methods, not just describe().
        This prevents false positives where a class has describe() but is
        missing other essential handler methods.
        """
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        # MockPartialHandler only has describe(), missing other 4 methods
        is_valid, missing_methods = loader._validate_handler_protocol(
            MockPartialHandler
        )
        assert is_valid is False
        # Should be missing 4 methods (has describe, missing the rest)
        assert set(missing_methods) == {
            "handler_type",
            "initialize",
            "shutdown",
            "execute",
        }

    def test_validate_non_callable_describe_rejected(self) -> None:
        """Test that non-callable describe attribute is rejected."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        # Create a class with describe as an attribute, not a method
        class HandlerWithNonCallableDescribe:
            describe = "not a method"

        is_valid, missing_methods = loader._validate_handler_protocol(
            HandlerWithNonCallableDescribe
        )
        assert is_valid is False
        # describe should be in missing_methods because it's not callable
        assert "describe" in missing_methods

    def test_validate_missing_individual_methods(self) -> None:
        """Test that each missing required method causes validation to fail.

        Verifies that validation is comprehensive - missing ANY of the 5
        required methods should cause rejection, and the specific missing
        method is reported in the returned missing_methods list.
        """
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        # Missing handler_type
        class MissingHandlerType:
            async def initialize(self, config: object) -> None:
                pass

            async def shutdown(self, timeout_seconds: float = 30.0) -> None:
                pass

            async def execute(self, request: object, config: object) -> object:
                return {}

            def describe(self) -> dict[str, object]:
                return {}

        is_valid, missing = loader._validate_handler_protocol(MissingHandlerType)
        assert is_valid is False
        assert "handler_type" in missing

        # Missing initialize
        class MissingInitialize:
            @property
            def handler_type(self) -> str:
                return "test"

            async def shutdown(self, timeout_seconds: float = 30.0) -> None:
                pass

            async def execute(self, request: object, config: object) -> object:
                return {}

            def describe(self) -> dict[str, object]:
                return {}

        is_valid, missing = loader._validate_handler_protocol(MissingInitialize)
        assert is_valid is False
        assert "initialize" in missing

        # Missing shutdown
        class MissingShutdown:
            @property
            def handler_type(self) -> str:
                return "test"

            async def initialize(self, config: object) -> None:
                pass

            async def execute(self, request: object, config: object) -> object:
                return {}

            def describe(self) -> dict[str, object]:
                return {}

        is_valid, missing = loader._validate_handler_protocol(MissingShutdown)
        assert is_valid is False
        assert "shutdown" in missing

        # Missing execute
        class MissingExecute:
            @property
            def handler_type(self) -> str:
                return "test"

            async def initialize(self, config: object) -> None:
                pass

            async def shutdown(self, timeout_seconds: float = 30.0) -> None:
                pass

            def describe(self) -> dict[str, object]:
                return {}

        is_valid, missing = loader._validate_handler_protocol(MissingExecute)
        assert is_valid is False
        assert "execute" in missing

        # Missing describe (already tested but included for completeness)
        class MissingDescribe:
            @property
            def handler_type(self) -> str:
                return "test"

            async def initialize(self, config: object) -> None:
                pass

            async def shutdown(self, timeout_seconds: float = 30.0) -> None:
                pass

            async def execute(self, request: object, config: object) -> object:
                return {}

        is_valid, missing = loader._validate_handler_protocol(MissingDescribe)
        assert is_valid is False
        assert "describe" in missing

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

        # Verify error message indicates invalid handler_type
        # Pydantic validation produces "Invalid handler_type '   '. Valid types: ..."
        error_msg = str(exc_info.value).lower()
        assert "invalid" in error_msg and "handler_type" in error_msg

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

        # Verify error message indicates invalid handler_type
        # Pydantic validation produces "Invalid handler_type ''. Valid types: ..."
        error_msg = str(exc_info.value).lower()
        assert "invalid" in error_msg and "handler_type" in error_msg
