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
from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumHandlerLoaderError
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

from .conftest import VALID_HANDLER_CONTRACT_YAML


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

    def test_prefix_without_period_enforces_package_boundary(self) -> None:
        """Prefix without trailing period should enforce package boundary.

        The namespace allowlist now enforces proper package boundaries. When
        a namespace like "foo" is allowed (without trailing period), it will:
        - Match "foo.handlers.Auth" (followed by ".")
        - Match "foo" exactly (though unlikely for class paths)
        - NOT match "foobar.malicious.Handler" (no boundary after "foo")

        This prevents package-boundary bypass vulnerabilities where allowing
        "omnibase" would accidentally also allow "omnibase_other".
        """
        loader = HandlerPluginLoader(allowed_namespaces=["omnibase"])

        # This should match - "omnibase" followed by "."
        loader._validate_namespace(
            "omnibase.handlers.HandlerAuth",
            Path("contract.yaml"),
        )

        # This should NOT match - "omnibase_infra" is a different package
        # "omnibase" is a prefix but not at a package boundary
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader._validate_namespace(
                "omnibase_infra.handlers.HandlerAuth",
                Path("contract.yaml"),
            )
        assert (
            exc_info.value.model.context.get("loader_error")
            == EnumHandlerLoaderError.NAMESPACE_NOT_ALLOWED.value
        )

        # This should also NOT match - "omnibase_other" is a different package
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader._validate_namespace(
                "omnibase_other.malicious.Handler",
                Path("contract.yaml"),
            )
        assert (
            exc_info.value.model.context.get("loader_error")
            == EnumHandlerLoaderError.NAMESPACE_NOT_ALLOWED.value
        )

    def test_prefix_with_underscore_boundary_not_confused(self) -> None:
        """Package names with underscores should not be confused.

        "foo_bar" allowlist should not match "foo_baz" or "foo_bar_extra".
        """
        loader = HandlerPluginLoader(allowed_namespaces=["foo_bar"])

        # This should match - "foo_bar" followed by "."
        loader._validate_namespace(
            "foo_bar.handlers.Handler",
            Path("contract.yaml"),
        )

        # This should NOT match - "foo_baz" is a different package
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader._validate_namespace(
                "foo_baz.handlers.Handler",
                Path("contract.yaml"),
            )
        assert (
            exc_info.value.model.context.get("loader_error")
            == EnumHandlerLoaderError.NAMESPACE_NOT_ALLOWED.value
        )

        # This should NOT match - "foo_bar_extra" is a different package
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader._validate_namespace(
                "foo_bar_extra.handlers.Handler",
                Path("contract.yaml"),
            )
        assert (
            exc_info.value.model.context.get("loader_error")
            == EnumHandlerLoaderError.NAMESPACE_NOT_ALLOWED.value
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


class TestNamespaceBoundaryBypassPrevention:
    """Tests to verify namespace matching prevents package-boundary bypass attacks.

    These tests verify that namespace allowlisting properly enforces package
    boundaries to prevent malicious packages from bypassing the allowlist
    by using similar-looking package names.

    Security Concern:
        If "omnibase" is allowed without trailing period, it could accidentally
        allow "omnibase_evil" or "omnibase_malicious" - packages that are NOT
        part of the trusted omnibase ecosystem.
    """

    def test_omnibase_does_not_match_omnibase_evil(self) -> None:
        """Verify 'omnibase' allowlist does NOT match 'omnibase_evil' package.

        This is a critical security test: allowing "omnibase" should NOT
        accidentally allow malicious packages like "omnibase_evil".
        """
        loader = HandlerPluginLoader(allowed_namespaces=["omnibase"])

        # "omnibase.handlers.Auth" should be allowed (proper package boundary)
        loader._validate_namespace(
            "omnibase.handlers.Auth",
            Path("contract.yaml"),
        )

        # "omnibase_evil" should be BLOCKED (different package despite similar prefix)
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader._validate_namespace(
                "omnibase_evil.malicious.Payload",
                Path("contract.yaml"),
            )
        assert (
            exc_info.value.model.context.get("loader_error")
            == EnumHandlerLoaderError.NAMESPACE_NOT_ALLOWED.value
        )

    def test_omnibase_does_not_match_omnibase_malicious(self) -> None:
        """Verify 'omnibase' allowlist does NOT match 'omnibase_malicious'."""
        loader = HandlerPluginLoader(allowed_namespaces=["omnibase"])

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader._validate_namespace(
                "omnibase_malicious.backdoor.Handler",
                Path("contract.yaml"),
            )
        assert (
            exc_info.value.model.context.get("loader_error")
            == EnumHandlerLoaderError.NAMESPACE_NOT_ALLOWED.value
        )

    def test_mycompany_does_not_match_mycompany_fake(self) -> None:
        """Verify 'mycompany' allowlist does NOT match 'mycompany_fake'."""
        loader = HandlerPluginLoader(allowed_namespaces=["mycompany"])

        # Legitimate package should work
        loader._validate_namespace(
            "mycompany.auth.Handler",
            Path("contract.yaml"),
        )

        # Typosquatting/malicious package should be blocked
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader._validate_namespace(
                "mycompany_fake.phishing.Handler",
                Path("contract.yaml"),
            )
        assert (
            exc_info.value.model.context.get("loader_error")
            == EnumHandlerLoaderError.NAMESPACE_NOT_ALLOWED.value
        )

    def test_trailing_period_is_explicit_boundary(self) -> None:
        """Verify trailing period explicitly enforces package boundary.

        When allowlist uses "omnibase_infra." (with period), only that exact
        package and its subpackages should be allowed.
        """
        loader = HandlerPluginLoader(allowed_namespaces=["omnibase_infra."])

        # Should allow subpackages
        loader._validate_namespace(
            "omnibase_infra.handlers.Auth",
            Path("contract.yaml"),
        )
        loader._validate_namespace(
            "omnibase_infra.runtime.loader.Handler",
            Path("contract.yaml"),
        )

        # Should block similar-named packages
        with pytest.raises(ProtocolConfigurationError):
            loader._validate_namespace(
                "omnibase_infra_evil.Handler",
                Path("contract.yaml"),
            )

        # Should block packages that share partial prefix
        with pytest.raises(ProtocolConfigurationError):
            loader._validate_namespace(
                "omnibase_infrastructure.Handler",
                Path("contract.yaml"),
            )

    def test_underscore_in_package_name_not_confused_with_boundary(self) -> None:
        """Verify underscores in package names don't confuse boundary matching.

        Package names like "my_package" should not match "my" or "my_package_extra".
        """
        loader = HandlerPluginLoader(allowed_namespaces=["my_package"])

        # Exact package match should work
        loader._validate_namespace(
            "my_package.module.Handler",
            Path("contract.yaml"),
        )

        # Different package starting with same prefix should fail
        with pytest.raises(ProtocolConfigurationError):
            loader._validate_namespace(
                "my_package_extension.Handler",
                Path("contract.yaml"),
            )

        # Shorter package should also fail
        with pytest.raises(ProtocolConfigurationError):
            loader._validate_namespace(
                "my.different.Handler",
                Path("contract.yaml"),
            )

    def test_numeric_suffix_in_package_name(self) -> None:
        """Verify package names with numeric suffixes are handled correctly."""
        loader = HandlerPluginLoader(allowed_namespaces=["package_v1"])

        # Exact match should work
        loader._validate_namespace(
            "package_v1.module.Handler",
            Path("contract.yaml"),
        )

        # Different version should fail
        with pytest.raises(ProtocolConfigurationError):
            loader._validate_namespace(
                "package_v2.module.Handler",
                Path("contract.yaml"),
            )

        # Extended version should fail
        with pytest.raises(ProtocolConfigurationError):
            loader._validate_namespace(
                "package_v10.module.Handler",
                Path("contract.yaml"),
            )
