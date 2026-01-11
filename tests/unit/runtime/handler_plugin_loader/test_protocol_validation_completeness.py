# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests verifying _validate_handler_protocol() checks match ProtocolHandler.

This module ensures that the HandlerPluginLoader's protocol validation
correctly reflects the ProtocolHandler interface requirements. The validation
must be kept in sync with the actual protocol definition.

Test Coverage:
- Verify all 5 required methods are checked
- Verify validation order matches documentation
- Cross-validate against real handler implementations
- Verify optional methods (health_check) are NOT required

Related:
    - OMN-1132: Handler Plugin Loader implementation
    - PR #134: Security enhancements and protocol validation
    - docs/patterns/handler_plugin_loader.md

Protocol Compliance Requirements (from handler_plugin_loader.py docstring):
    ProtocolHandler requires these 5 methods:
    1. handler_type (property): Returns handler type identifier string
    2. initialize(config): Async method to initialize connections/pools
    3. shutdown(timeout_seconds): Async method to release resources
    4. execute(request, operation_config): Async method for operations
    5. describe(): Sync method returning handler metadata/capabilities

    Note: health_check() is part of ProtocolHandler but is OPTIONAL because
    existing handlers (HandlerHttp, HandlerDb, etc.) do not implement it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader


class TestProtocolValidationCompleteness:
    """Tests ensuring _validate_handler_protocol checks all required methods.

    These tests verify that the validation logic in HandlerPluginLoader
    correctly identifies all required ProtocolHandler methods.
    """

    # The 5 required methods that MUST be checked
    REQUIRED_METHODS = {
        "handler_type",
        "initialize",
        "shutdown",
        "execute",
        "describe",
    }

    # Methods that are optional (part of protocol but not validated)
    OPTIONAL_METHODS = {"health_check"}

    def test_validation_checks_exactly_five_methods(self) -> None:
        """Verify validation checks exactly 5 required methods.

        The _validate_handler_protocol method should check for exactly
        the 5 methods documented in the protocol requirements.
        """
        loader = HandlerPluginLoader()

        # Create a class with no methods
        class EmptyHandler:
            pass

        is_valid, missing_methods = loader._validate_handler_protocol(EmptyHandler)

        assert is_valid is False
        assert len(missing_methods) == 5
        assert set(missing_methods) == self.REQUIRED_METHODS

    def test_validation_reports_correct_method_names(self) -> None:
        """Verify validation reports the exact method names from protocol.

        Missing method names should match the protocol definition exactly.
        """
        loader = HandlerPluginLoader()

        class EmptyHandler:
            pass

        _, missing = loader._validate_handler_protocol(EmptyHandler)

        # Verify exact method names
        assert "handler_type" in missing
        assert "initialize" in missing
        assert "shutdown" in missing
        assert "execute" in missing
        assert "describe" in missing

        # Verify no extra methods are required
        extra_methods = set(missing) - self.REQUIRED_METHODS
        assert extra_methods == set(), f"Unexpected required methods: {extra_methods}"

    def test_health_check_is_not_required(self) -> None:
        """Verify health_check() is NOT required by validation.

        Per the protocol definition, health_check() is optional because
        existing handlers (HandlerHttp, HandlerDb, HandlerVault, HandlerConsul)
        do not implement it.
        """
        loader = HandlerPluginLoader()

        # Create a handler with all 5 required methods but WITHOUT health_check
        class HandlerWithoutHealthCheck:
            @property
            def handler_type(self) -> str:
                return "test"

            async def initialize(self, config: object) -> None:
                pass

            async def shutdown(self, timeout_seconds: float = 30.0) -> None:
                pass

            async def execute(self, request: object, config: object) -> object:
                return {}

            def describe(self) -> dict[str, object]:
                return {}

        is_valid, missing = loader._validate_handler_protocol(HandlerWithoutHealthCheck)

        # Should pass validation even without health_check
        assert is_valid is True
        assert missing == []

    def test_handler_type_can_be_property_or_method(self) -> None:
        """Verify handler_type can be a property (preferred) or method.

        The validation uses hasattr() for handler_type to accommodate both
        property decorators and regular attributes.
        """
        loader = HandlerPluginLoader()

        # Test with property decorator
        class HandlerWithProperty:
            @property
            def handler_type(self) -> str:
                return "test"

            async def initialize(self, config: object) -> None:
                pass

            async def shutdown(self, timeout_seconds: float = 30.0) -> None:
                pass

            async def execute(self, request: object, config: object) -> object:
                return {}

            def describe(self) -> dict[str, object]:
                return {}

        is_valid, _ = loader._validate_handler_protocol(HandlerWithProperty)
        assert is_valid is True

        # Test with method (less common but valid)
        class HandlerWithMethod:
            def handler_type(self) -> str:
                return "test"

            async def initialize(self, config: object) -> None:
                pass

            async def shutdown(self, timeout_seconds: float = 30.0) -> None:
                pass

            async def execute(self, request: object, config: object) -> object:
                return {}

            def describe(self) -> dict[str, object]:
                return {}

        is_valid, _ = loader._validate_handler_protocol(HandlerWithMethod)
        assert is_valid is True

    def test_callable_check_rejects_non_callable_attributes(self) -> None:
        """Verify validation rejects non-callable attributes.

        Methods must be callable, not just attributes with matching names.
        """
        loader = HandlerPluginLoader()

        class HandlerWithAttributes:
            handler_type = "test"  # Attribute, not method
            initialize = "not callable"
            shutdown = 123
            execute = None
            describe = "also not callable"

        is_valid, missing = loader._validate_handler_protocol(HandlerWithAttributes)

        assert is_valid is False
        # handler_type passes hasattr check (it's a property-like attribute)
        # But initialize, shutdown, execute, describe fail callable() check
        assert "initialize" in missing
        assert "shutdown" in missing
        assert "execute" in missing
        assert "describe" in missing

    def test_each_missing_method_reported_individually(self) -> None:
        """Verify each missing method is individually reported.

        When multiple methods are missing, each should appear in the
        missing_methods list exactly once.
        """
        loader = HandlerPluginLoader()

        # Missing handler_type only
        class MissingHandlerType:
            async def initialize(self, config: object) -> None:
                pass

            async def shutdown(self, timeout_seconds: float = 30.0) -> None:
                pass

            async def execute(self, request: object, config: object) -> object:
                return {}

            def describe(self) -> dict[str, object]:
                return {}

        _, missing = loader._validate_handler_protocol(MissingHandlerType)
        assert missing == ["handler_type"]

        # Missing multiple methods
        class MissingMultiple:
            @property
            def handler_type(self) -> str:
                return "test"

            def describe(self) -> dict[str, object]:
                return {}

        _, missing = loader._validate_handler_protocol(MissingMultiple)
        assert set(missing) == {"initialize", "shutdown", "execute"}


class TestProtocolValidationAgainstRealHandlers:
    """Cross-validate protocol requirements against real handler implementations.

    These tests import actual handler classes from omnibase_infra.handlers
    and verify they pass protocol validation. This ensures the validation
    logic stays in sync with real implementations.
    """

    # Real handler class paths to validate
    REAL_HANDLERS = [
        "omnibase_infra.handlers.handler_http.HttpRestHandler",
        "omnibase_infra.handlers.handler_db.HandlerDb",
        "omnibase_infra.handlers.handler_vault.HandlerVault",
        "omnibase_infra.handlers.handler_consul.HandlerConsul",
    ]

    @pytest.mark.parametrize(
        "handler_class_path",
        REAL_HANDLERS,
        ids=["http", "db", "vault", "consul"],
    )
    def test_real_handlers_pass_validation(self, handler_class_path: str) -> None:
        """Verify all real handlers pass protocol validation.

        If validation requirements change, real handlers should still pass.
        This test catches regressions where validation becomes too strict.
        """
        import importlib

        loader = HandlerPluginLoader()

        # Import the handler class
        module_path, class_name = handler_class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        handler_class = getattr(module, class_name)

        is_valid, missing = loader._validate_handler_protocol(handler_class)

        assert is_valid, (
            f"Real handler {handler_class_path} failed validation. "
            f"Missing methods: {missing}. "
            f"This may indicate protocol validation is too strict."
        )
        assert missing == []

    @pytest.mark.parametrize(
        "handler_class_path",
        REAL_HANDLERS,
        ids=["http", "db", "vault", "consul"],
    )
    def test_real_handlers_have_all_required_methods(
        self, handler_class_path: str
    ) -> None:
        """Verify real handlers implement all 5 required methods.

        This tests the methods directly on the class, independent of
        the validation logic, to ensure our test assumptions are correct.
        """
        import importlib

        module_path, class_name = handler_class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        handler_class = getattr(module, class_name)

        # Check each required method exists
        assert hasattr(handler_class, "handler_type")
        assert callable(getattr(handler_class, "initialize", None))
        assert callable(getattr(handler_class, "shutdown", None))
        assert callable(getattr(handler_class, "execute", None))
        assert callable(getattr(handler_class, "describe", None))


class TestProtocolValidationDocumentation:
    """Tests ensuring validation matches documented requirements.

    The documentation in handler_plugin_loader.py and CLAUDE.md specifies
    exactly which methods are required. These tests verify the implementation
    matches the documentation.
    """

    def test_documented_required_methods_are_validated(self) -> None:
        """Verify all documented required methods are validated.

        From handler_plugin_loader.py docstring:
        - handler_type (property)
        - initialize(config)
        - shutdown(timeout_seconds)
        - execute(request, operation_config)
        - describe()
        """
        loader = HandlerPluginLoader()

        # Create class missing all methods
        class EmptyHandler:
            pass

        _, missing = loader._validate_handler_protocol(EmptyHandler)

        # All documented methods should be in missing list
        documented_methods = {
            "handler_type",
            "initialize",
            "shutdown",
            "execute",
            "describe",
        }

        for method in documented_methods:
            assert method in missing, (
                f"Documented method '{method}' is not being validated. "
                f"Update _validate_handler_protocol to check for this method."
            )

    def test_optional_health_check_documented_correctly(self) -> None:
        """Verify health_check is NOT in the required methods list.

        From handler_plugin_loader.py docstring:
        'Note: health_check() is part of ProtocolHandler but is NOT validated
        because existing handlers (HandlerHttp, HandlerDb, etc.) do not
        implement it.'
        """
        loader = HandlerPluginLoader()

        class EmptyHandler:
            pass

        _, missing = loader._validate_handler_protocol(EmptyHandler)

        # health_check should NOT be required
        assert "health_check" not in missing, (
            "health_check should be optional, not required. "
            "Existing handlers do not implement it."
        )
