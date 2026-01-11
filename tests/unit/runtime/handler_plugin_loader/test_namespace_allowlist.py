# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for HandlerPluginLoader namespace allowlisting feature.

This module tests the defense-in-depth namespace allowlisting feature that
restricts which Python packages can be dynamically imported via handler contracts.

Part of OMN-1132: Handler Plugin Loader implementation - Security Enhancement.

Test Coverage:
    - Allowed namespace passes validation
    - Disallowed namespace raises NAMESPACE_NOT_ALLOWED error
    - None (no restriction) allows any namespace
    - Empty list blocks all namespaces
    - Prefix matching behavior (with and without trailing period)
    - Error message contains allowed namespaces
    - Correlation ID is included in error context
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumHandlerLoaderError
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

from .conftest import VALID_HANDLER_CONTRACT_YAML, MockValidHandler


class TestNamespaceAllowlistValidation:
    """Tests for namespace allowlist validation behavior."""

    def test_allowed_namespace_passes_validation(self, tmp_path: Path) -> None:
        """Namespace in allowed list should pass validation and load successfully."""
        # Create a contract that uses the test module namespace
        contract_dir = tmp_path / "handler"
        contract_dir.mkdir(parents=True)
        contract_file = contract_dir / "handler_contract.yaml"
        contract_file.write_text(
            VALID_HANDLER_CONTRACT_YAML.format(
                handler_name="test.handler",
                handler_class=f"{__name__.rsplit('.', 1)[0]}.conftest.MockValidHandler",
                handler_type="compute",
                tag1="test",
                tag2="namespace",
            )
        )

        # Create loader with namespace that matches the test module
        loader = HandlerPluginLoader(
            allowed_namespaces=[
                "tests.unit.runtime.handler_plugin_loader.",
            ]
        )

        # Should load successfully
        handler = loader.load_from_contract(contract_file)
        assert handler.handler_name == "test.handler"

    def test_disallowed_namespace_raises_error(self, tmp_path: Path) -> None:
        """Namespace not in allowed list should raise NAMESPACE_NOT_ALLOWED error."""
        # Create a contract that uses the test module namespace
        contract_dir = tmp_path / "handler"
        contract_dir.mkdir(parents=True)
        contract_file = contract_dir / "handler_contract.yaml"
        contract_file.write_text(
            VALID_HANDLER_CONTRACT_YAML.format(
                handler_name="test.handler",
                handler_class=f"{__name__.rsplit('.', 1)[0]}.conftest.MockValidHandler",
                handler_type="compute",
                tag1="test",
                tag2="namespace",
            )
        )

        # Create loader with namespace that does NOT match
        loader = HandlerPluginLoader(
            allowed_namespaces=[
                "omnibase_infra.",
                "omnibase_core.",
                "mycompany.handlers.",
            ]
        )

        # Should raise ProtocolConfigurationError with NAMESPACE_NOT_ALLOWED
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.NAMESPACE_NOT_ALLOWED.value
        )
        assert "namespace not allowed" in str(error).lower()
        # Error message should list allowed namespaces
        assert "omnibase_infra." in str(error)

    def test_none_allows_any_namespace(self, tmp_path: Path) -> None:
        """When allowed_namespaces is None, any namespace should be allowed."""
        # Create a contract that uses the test module namespace
        contract_dir = tmp_path / "handler"
        contract_dir.mkdir(parents=True)
        contract_file = contract_dir / "handler_contract.yaml"
        contract_file.write_text(
            VALID_HANDLER_CONTRACT_YAML.format(
                handler_name="test.handler",
                handler_class=f"{__name__.rsplit('.', 1)[0]}.conftest.MockValidHandler",
                handler_type="compute",
                tag1="test",
                tag2="namespace",
            )
        )

        # Create loader with no namespace restriction (default)
        loader = HandlerPluginLoader()  # allowed_namespaces defaults to None

        # Should load successfully
        handler = loader.load_from_contract(contract_file)
        assert handler.handler_name == "test.handler"

    def test_empty_list_blocks_all_namespaces(self, tmp_path: Path) -> None:
        """When allowed_namespaces is empty list, ALL namespaces should be blocked."""
        # Create a contract
        contract_dir = tmp_path / "handler"
        contract_dir.mkdir(parents=True)
        contract_file = contract_dir / "handler_contract.yaml"
        contract_file.write_text(
            VALID_HANDLER_CONTRACT_YAML.format(
                handler_name="test.handler",
                handler_class=f"{__name__.rsplit('.', 1)[0]}.conftest.MockValidHandler",
                handler_type="compute",
                tag1="test",
                tag2="namespace",
            )
        )

        # Create loader with empty allowlist
        loader = HandlerPluginLoader(allowed_namespaces=[])

        # Should raise ProtocolConfigurationError
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.NAMESPACE_NOT_ALLOWED.value
        )
        # Error message should indicate empty allowlist
        assert "empty allowlist" in str(error).lower()


class TestNamespacePrefixMatching:
    """Tests for namespace prefix matching behavior."""

    def test_exact_prefix_match_with_period(self) -> None:
        """Prefix with trailing period should match exact package boundaries."""
        loader = HandlerPluginLoader(allowed_namespaces=["omnibase_infra."])

        # Direct call to _validate_namespace for fine-grained testing
        # This should pass (exact prefix match)
        loader._validate_namespace(
            "omnibase_infra.handlers.HandlerAuth",
            Path("contract.yaml"),
        )

        # This should also pass
        loader._validate_namespace(
            "omnibase_infra.runtime.handler_plugin_loader.HandlerPluginLoader",
            Path("contract.yaml"),
        )

    def test_prefix_without_period_matches_unintended_packages(self) -> None:
        """Prefix without trailing period may match unintended packages."""
        loader = HandlerPluginLoader(allowed_namespaces=["omnibase"])

        # This matches (expected)
        loader._validate_namespace(
            "omnibase_infra.handlers.HandlerAuth",
            Path("contract.yaml"),
        )

        # This also matches! (potentially unexpected)
        loader._validate_namespace(
            "omnibase_other.malicious.Handler",
            Path("contract.yaml"),
        )

    def test_multiple_allowed_namespaces(self) -> None:
        """Multiple allowed namespaces should all be checked."""
        loader = HandlerPluginLoader(
            allowed_namespaces=[
                "omnibase_infra.",
                "omnibase_core.",
                "mycompany.handlers.",
            ]
        )

        # All of these should pass
        loader._validate_namespace("omnibase_infra.handlers.Auth", Path("c.yaml"))
        loader._validate_namespace("omnibase_core.models.Base", Path("c.yaml"))
        loader._validate_namespace("mycompany.handlers.Custom", Path("c.yaml"))

        # This should fail
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader._validate_namespace("malicious.package.Evil", Path("c.yaml"))

        assert (
            exc_info.value.model.context.get("loader_error")
            == EnumHandlerLoaderError.NAMESPACE_NOT_ALLOWED.value
        )


class TestNamespaceValidationErrorDetails:
    """Tests for error message and context details."""

    def test_error_contains_class_path(self) -> None:
        """Error should contain the class path that was rejected."""
        loader = HandlerPluginLoader(allowed_namespaces=["allowed."])
        class_path = "malicious.package.EvilHandler"

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader._validate_namespace(class_path, Path("contract.yaml"))

        assert class_path in str(exc_info.value)

    def test_error_contains_allowed_namespaces(self) -> None:
        """Error should contain the list of allowed namespaces."""
        allowed = ["omnibase_infra.", "omnibase_core.", "mycompany."]
        loader = HandlerPluginLoader(allowed_namespaces=allowed)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader._validate_namespace("evil.Handler", Path("contract.yaml"))

        error_message = str(exc_info.value)
        for ns in allowed:
            assert ns in error_message or repr(ns) in error_message

    def test_error_context_contains_allowed_namespaces_list(self) -> None:
        """Error context should contain allowed_namespaces as a list."""
        allowed = ["omnibase_infra.", "omnibase_core."]
        loader = HandlerPluginLoader(allowed_namespaces=allowed)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader._validate_namespace("evil.Handler", Path("contract.yaml"))

        context = exc_info.value.model.context
        assert context.get("allowed_namespaces") == allowed

    def test_correlation_id_is_included_in_error(self) -> None:
        """Correlation ID should be included in error model."""
        loader = HandlerPluginLoader(allowed_namespaces=["allowed."])
        correlation_id = uuid4()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader._validate_namespace(
                "evil.Handler",
                Path("contract.yaml"),
                correlation_id=correlation_id,
            )

        # Correlation ID should be on the error model directly
        assert exc_info.value.model.correlation_id == correlation_id


class TestNamespaceValidationIntegration:
    """Integration tests verifying namespace validation in the full load flow."""

    def test_validation_occurs_before_import(self, tmp_path: Path) -> None:
        """Namespace validation should occur BEFORE importlib.import_module().

        This is critical for security - we must reject disallowed namespaces
        before any module-level code can execute.
        """
        # Create a contract pointing to a non-existent but disallowed module
        contract_dir = tmp_path / "handler"
        contract_dir.mkdir(parents=True)
        contract_file = contract_dir / "handler_contract.yaml"
        contract_file.write_text(
            VALID_HANDLER_CONTRACT_YAML.format(
                handler_name="test.handler",
                # This module doesn't exist, but namespace validation should
                # reject it BEFORE we try to import it
                handler_class="malicious_nonexistent.package.EvilHandler",
                handler_type="compute",
                tag1="test",
                tag2="security",
            )
        )

        loader = HandlerPluginLoader(allowed_namespaces=["omnibase_infra."])

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        # Should be NAMESPACE_NOT_ALLOWED, NOT MODULE_NOT_FOUND
        # This proves validation happened before import attempt
        assert (
            exc_info.value.model.context.get("loader_error")
            == EnumHandlerLoaderError.NAMESPACE_NOT_ALLOWED.value
        )

    def test_directory_load_respects_namespace_restriction(
        self, tmp_path: Path
    ) -> None:
        """load_from_directory should respect namespace restrictions."""
        # Create valid handler contract
        handler_dir = tmp_path / "handler1"
        handler_dir.mkdir(parents=True)
        (handler_dir / "handler_contract.yaml").write_text(
            VALID_HANDLER_CONTRACT_YAML.format(
                handler_name="handler.one",
                handler_class=f"{__name__.rsplit('.', 1)[0]}.conftest.MockValidHandler",
                handler_type="compute",
                tag1="test",
                tag2="namespace",
            )
        )

        # Create loader with namespace that matches
        loader = HandlerPluginLoader(
            allowed_namespaces=["tests.unit.runtime.handler_plugin_loader."]
        )

        # Should load successfully
        handlers = loader.load_from_directory(tmp_path)
        assert len(handlers) == 1

        # Now with a namespace that doesn't match
        loader_restricted = HandlerPluginLoader(allowed_namespaces=["omnibase_infra."])

        # Should fail to load (graceful mode - returns empty list)
        handlers = loader_restricted.load_from_directory(tmp_path)
        assert len(handlers) == 0  # Failed handlers don't appear in result


class TestNamespaceAllowlistInitialization:
    """Tests for HandlerPluginLoader initialization with allowed_namespaces."""

    def test_init_stores_allowed_namespaces(self) -> None:
        """Allowed namespaces should be stored during initialization."""
        allowed = ["omnibase_infra.", "omnibase_core."]
        loader = HandlerPluginLoader(allowed_namespaces=allowed)

        # Internal attribute check
        assert loader._allowed_namespaces == allowed

    def test_init_with_none_stores_none(self) -> None:
        """None value should be stored as-is (no restriction)."""
        loader = HandlerPluginLoader(allowed_namespaces=None)
        assert loader._allowed_namespaces is None

    def test_init_default_is_none(self) -> None:
        """Default value for allowed_namespaces should be None."""
        loader = HandlerPluginLoader()
        assert loader._allowed_namespaces is None

    def test_init_with_empty_list_stores_empty_list(self) -> None:
        """Empty list should be stored as-is (block all)."""
        loader = HandlerPluginLoader(allowed_namespaces=[])
        assert loader._allowed_namespaces == []
