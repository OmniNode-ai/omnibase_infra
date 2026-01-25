# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Security tests for HandlerPluginLoader namespace allowlisting.

This module provides explicit security documentation tests that verify the
handler namespace allowlisting feature correctly prevents loading handlers
from untrusted namespaces. These tests serve as security regression tests
and documentation for the defense-in-depth namespace restriction feature.

Security Context:
    The HandlerPluginLoader dynamically imports Python classes specified in YAML
    contracts. Without namespace restrictions, a malicious YAML contract could
    specify an untrusted handler class path (e.g., `malicious_external_package.handlers.EvilHandler`)
    and cause arbitrary code execution during import.

    The `allowed_namespaces` parameter provides defense-in-depth by restricting
    which module namespaces can be imported. Namespace validation occurs BEFORE
    `importlib.import_module()` is called, preventing any module-level side effects
    from untrusted packages.

Error Codes:
    - NAMESPACE_NOT_ALLOWED (HANDLER_LOADER_013): Raised when handler module
      namespace is not in the allowed list.

Related:
    - OMN-1132: Handler Plugin Loader implementation
    - CLAUDE.md: Handler Plugin Loader Patterns section
    - docs/patterns/handler_plugin_loader.md
    - docs/decisions/adr-handler-plugin-loader-security.md

Test Categories:
    - TestHandlerPluginLoaderNamespaceSecurity: Core namespace security tests
    - TestNamespaceAllowlistSecurityBoundary: Package boundary bypass prevention
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumHandlerLoaderError
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

from .conftest import VALID_HANDLER_CONTRACT_YAML

# =============================================================================
# Test Contract Templates for Security Testing
# =============================================================================

MALICIOUS_HANDLER_CONTRACT_YAML = """
handler_name: "{handler_name}"
handler_class: "{handler_class}"
handler_type: "effect"
capability_tags:
  - external
  - untrusted
"""


class TestHandlerPluginLoaderNamespaceSecurity:
    """Security tests for handler namespace allowlisting.

    These tests verify that the allowed_namespaces parameter correctly
    prevents loading handlers from untrusted namespaces. This is a critical
    security control that prevents malicious YAML contracts from loading
    arbitrary code.

    Security Guarantees:
        1. Handlers outside allowed namespaces are REJECTED before import
        2. Rejection occurs with NAMESPACE_NOT_ALLOWED error code
        3. No module-level side effects execute from rejected namespaces
        4. Handlers inside allowed namespaces load successfully
    """

    @pytest.mark.asyncio
    async def test_rejects_handler_outside_allowed_namespace(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify handlers outside allowed namespaces are rejected.

        Given:
            - HandlerPluginLoader configured with allowed_namespaces
            - Contract referencing handler in disallowed namespace

        When:
            - load_from_contract() is called

        Then:
            - ProtocolConfigurationError raised with NAMESPACE_NOT_ALLOWED

        Security Note:
            This test verifies that an attacker cannot bypass namespace
            restrictions by creating a YAML contract that references a
            malicious external package. The namespace validation occurs
            BEFORE importlib.import_module(), preventing any module-level
            code execution from the untrusted package.
        """
        # Create malicious contract pointing to untrusted namespace
        contract_dir = tmp_path / "malicious_handler"
        contract_dir.mkdir(parents=True)
        contract_file = contract_dir / "handler_contract.yaml"
        contract_file.write_text(
            MALICIOUS_HANDLER_CONTRACT_YAML.format(
                handler_name="malicious.handler",
                handler_class="malicious_external_package.handlers.EvilHandler",
            )
        )

        # Create loader with restricted namespaces
        loader = HandlerPluginLoader(
            allowed_namespaces=["omnibase_infra.", "omnibase_core."]
        )

        # Attempt to load malicious contract - should be rejected
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        # Verify error details
        error = exc_info.value
        error_context = error.model.context

        # Must be NAMESPACE_NOT_ALLOWED error
        assert (
            error_context.get("loader_error")
            == EnumHandlerLoaderError.NAMESPACE_NOT_ALLOWED.value
        ), (
            f"Expected NAMESPACE_NOT_ALLOWED (HANDLER_LOADER_013) error, "
            f"got: {error_context.get('loader_error')}"
        )

        # Error message should indicate namespace rejection
        error_message = str(error).lower()
        assert "namespace not allowed" in error_message, (
            f"Error message should indicate namespace rejection: {error}"
        )

        # Error should include the rejected class path
        assert "malicious_external_package" in str(error), (
            f"Error should include rejected namespace: {error}"
        )

    @pytest.mark.asyncio
    async def test_accepts_handler_inside_allowed_namespace(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify handlers inside allowed namespaces are accepted.

        Given:
            - HandlerPluginLoader configured with allowed_namespaces
            - Contract referencing handler in ALLOWED namespace

        When:
            - load_from_contract() is called

        Then:
            - Handler loads successfully without error

        This positive test confirms that legitimate handlers from trusted
        namespaces continue to work when namespace restrictions are enabled.
        """
        # Create contract pointing to allowed namespace (test module)
        contract_dir = tmp_path / "valid_handler"
        contract_dir.mkdir(parents=True)
        contract_file = contract_dir / "handler_contract.yaml"
        contract_file.write_text(
            VALID_HANDLER_CONTRACT_YAML.format(
                handler_name="test.valid.handler",
                # Use the test conftest's MockValidHandler
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
                handler_type="compute",
                tag1="test",
                tag2="security",
            )
        )

        # Create loader with namespace that includes test modules
        loader = HandlerPluginLoader(
            allowed_namespaces=[
                "tests.unit.runtime.handler_plugin_loader.",
                "omnibase_infra.",
            ]
        )

        # Should load successfully
        result = loader.load_from_contract(contract_file)

        assert result.handler_name == "test.valid.handler"
        assert result.handler_class == (
            "tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler"
        )

    @pytest.mark.asyncio
    async def test_rejects_before_import_prevents_side_effects(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify namespace validation occurs BEFORE module import.

        This is critical for security: we must reject disallowed namespaces
        BEFORE calling importlib.import_module() to prevent any module-level
        code from executing.

        Given:
            - Contract pointing to non-existent malicious module
            - Loader with namespace restrictions

        When:
            - load_from_contract() is called

        Then:
            - NAMESPACE_NOT_ALLOWED error (not MODULE_NOT_FOUND)
            - Proves validation happened before import attempt

        Security Note:
            If we got MODULE_NOT_FOUND, it would mean we attempted to import
            the module and only failed because it doesn't exist. In a real
            attack, the module WOULD exist, and its module-level code would
            have already executed. NAMESPACE_NOT_ALLOWED proves we never
            attempted the import.
        """
        # Create contract pointing to non-existent but disallowed module
        contract_dir = tmp_path / "nonexistent_malicious"
        contract_dir.mkdir(parents=True)
        contract_file = contract_dir / "handler_contract.yaml"
        contract_file.write_text(
            MALICIOUS_HANDLER_CONTRACT_YAML.format(
                handler_name="nonexistent.malicious.handler",
                # This module doesn't exist, but namespace validation
                # should reject it BEFORE we try to import it
                handler_class="nonexistent_malicious_package.backdoor.Payload",
            )
        )

        loader = HandlerPluginLoader(allowed_namespaces=["omnibase_infra."])

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        error_code = exc_info.value.model.context.get("loader_error")

        # CRITICAL: Must be NAMESPACE_NOT_ALLOWED, NOT MODULE_NOT_FOUND
        assert error_code == EnumHandlerLoaderError.NAMESPACE_NOT_ALLOWED.value, (
            f"Expected NAMESPACE_NOT_ALLOWED (validation before import), "
            f"got {error_code}. If this is MODULE_NOT_FOUND, namespace "
            f"validation is happening AFTER import attempt, which is a "
            f"security vulnerability."
        )

    @pytest.mark.asyncio
    async def test_correlation_id_included_in_security_error(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify correlation ID is included in namespace rejection errors.

        Correlation IDs are essential for security incident tracking and
        forensic analysis. When a namespace rejection occurs, the correlation
        ID should be preserved for audit trail purposes.
        """
        contract_dir = tmp_path / "handler"
        contract_dir.mkdir(parents=True)
        contract_file = contract_dir / "handler_contract.yaml"
        contract_file.write_text(
            MALICIOUS_HANDLER_CONTRACT_YAML.format(
                handler_name="test.handler",
                handler_class="untrusted_package.handlers.Handler",
            )
        )

        loader = HandlerPluginLoader(allowed_namespaces=["trusted_only."])
        correlation_id = uuid4()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file, correlation_id=correlation_id)

        # Correlation ID should be preserved in error model
        assert exc_info.value.model.correlation_id == correlation_id, (
            "Correlation ID should be preserved in security error for audit trail"
        )


class TestNamespaceAllowlistSecurityBoundary:
    """Tests verifying namespace matching prevents package-boundary bypass attacks.

    These tests ensure that namespace allowlisting properly enforces package
    boundaries to prevent attackers from bypassing restrictions by using
    similar-looking package names (typosquatting/namespace confusion attacks).

    Attack Scenario:
        If "omnibase_infra" is allowed without proper boundary checking, an
        attacker could create a malicious package named "omnibase_infra_evil"
        or "omnibase_infrastructure" that would incorrectly pass validation.
    """

    def test_prevents_underscore_boundary_bypass(self) -> None:
        """Verify namespace boundary prevents underscore-based bypass.

        Attack vector: If "omnibase_infra." is allowed, an attacker might
        try "omnibase_infra_backdoor." expecting the prefix match to succeed.
        """
        loader = HandlerPluginLoader(allowed_namespaces=["omnibase_infra."])

        # Legitimate namespace should pass
        loader._validate_namespace(
            "omnibase_infra.handlers.HandlerAuth",
            Path("contract.yaml"),
        )

        # Similar but different namespace should fail
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader._validate_namespace(
                "omnibase_infra_backdoor.handlers.EvilHandler",
                Path("contract.yaml"),
            )

        assert (
            exc_info.value.model.context.get("loader_error")
            == EnumHandlerLoaderError.NAMESPACE_NOT_ALLOWED.value
        )

    def test_prevents_similar_name_typosquatting(self) -> None:
        """Verify namespace matching prevents typosquatting attacks.

        Attack vector: Attacker creates package with similar name hoping
        for accidental inclusion (e.g., "omnibase_infr" vs "omnibase_infra").
        """
        loader = HandlerPluginLoader(allowed_namespaces=["omnibase_infra."])

        # Typosquat attempts should all fail
        typosquat_attempts = [
            "omnibase_infr.handlers.Handler",  # Missing 'a'
            "omnibase_infras.handlers.Handler",  # Extra 's'
            "omnibase_infrastructure.handlers.Handler",  # Extended name
            "omnibasee_infra.handlers.Handler",  # Typo in first part
        ]

        for malicious_path in typosquat_attempts:
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                loader._validate_namespace(malicious_path, Path("contract.yaml"))

            assert (
                exc_info.value.model.context.get("loader_error")
                == EnumHandlerLoaderError.NAMESPACE_NOT_ALLOWED.value
            ), f"Typosquat attempt '{malicious_path}' should have been rejected"

    def test_prevents_namespace_without_period_from_matching_extended_names(
        self,
    ) -> None:
        """Verify namespace without trailing period enforces package boundary.

        When allowlist specifies "omnibase" (without period), it should match:
        - "omnibase.handlers.Auth" (exact package + subpackage)

        But NOT:
        - "omnibase_other.handlers.Evil" (different package)
        """
        loader = HandlerPluginLoader(allowed_namespaces=["omnibase"])

        # Exact package with subpackage should work
        loader._validate_namespace(
            "omnibase.handlers.Auth",
            Path("contract.yaml"),
        )

        # Different package starting with same letters should fail
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader._validate_namespace(
                "omnibase_core.handlers.Handler",
                Path("contract.yaml"),
            )

        assert (
            exc_info.value.model.context.get("loader_error")
            == EnumHandlerLoaderError.NAMESPACE_NOT_ALLOWED.value
        )


class TestNamespaceSecurityErrorMessages:
    """Tests verifying security error messages don't leak sensitive information.

    Error messages for security failures should be informative for debugging
    but should not leak sensitive implementation details that could help
    an attacker.
    """

    def test_error_includes_rejected_namespace(self) -> None:
        """Verify error message includes the rejected namespace for debugging."""
        loader = HandlerPluginLoader(allowed_namespaces=["trusted."])
        malicious_path = "attacker_package.malware.Payload"

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader._validate_namespace(malicious_path, Path("contract.yaml"))

        # Should include rejected path for debugging
        assert malicious_path in str(exc_info.value)

    def test_error_includes_allowed_namespaces_for_remediation(self) -> None:
        """Verify error message includes allowed namespaces for remediation guidance."""
        allowed = ["omnibase_infra.", "omnibase_core."]
        loader = HandlerPluginLoader(allowed_namespaces=allowed)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader._validate_namespace(
                "untrusted.Handler",
                Path("contract.yaml"),
            )

        error_message = str(exc_info.value)
        # Error should include allowed namespaces so user knows what IS allowed
        for namespace in allowed:
            assert namespace in error_message or repr(namespace) in error_message

    def test_error_context_contains_structured_security_data(self) -> None:
        """Verify error context contains structured data for security logging."""
        allowed = ["trusted."]
        loader = HandlerPluginLoader(allowed_namespaces=allowed)
        rejected_path = "untrusted.Handler"

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader._validate_namespace(rejected_path, Path("test.yaml"))

        context = exc_info.value.model.context

        # Should contain structured data for security logging/alerting
        assert (
            context.get("loader_error")
            == EnumHandlerLoaderError.NAMESPACE_NOT_ALLOWED.value
        )
        assert context.get("class_path") == rejected_path
        assert context.get("allowed_namespaces") == allowed
        assert "contract_path" in context


__all__ = [
    "TestHandlerPluginLoaderNamespaceSecurity",
    "TestNamespaceAllowlistSecurityBoundary",
    "TestNamespaceSecurityErrorMessages",
]
