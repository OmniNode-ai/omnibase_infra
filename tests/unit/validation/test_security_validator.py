# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for security validation (OMN-1091).

Tests the integration of ModelHandlerValidationError in security validation paths.
"""

import pytest

from omnibase_infra.enums import EnumHandlerErrorType, EnumHandlerSourceType
from omnibase_infra.models.errors import ModelHandlerValidationError
from omnibase_infra.models.handlers import ModelHandlerIdentifier
from omnibase_infra.validation.security_validator import (
    SENSITIVE_METHOD_PATTERNS,
    SENSITIVE_PARAMETER_NAMES,
    SecurityRuleId,
    has_sensitive_parameters,
    is_sensitive_method_name,
    validate_handler_security,
    validate_method_exposure,
)


class TestSensitiveMethodDetection:
    """Test sensitive method name detection."""

    @pytest.mark.parametrize(
        ("method_name", "expected"),
        [
            ("get_password", True),
            ("get_secret", True),
            ("get_api_key", True),
            ("decrypt_data", True),
            ("admin_delete", True),
            ("process_request", False),
            ("execute_query", False),
            ("handle_event", False),
        ],
    )
    def test_is_sensitive_method_name(self, method_name: str, expected: bool) -> None:
        """Test detection of sensitive method names."""
        assert is_sensitive_method_name(method_name) == expected


class TestSensitiveParameterDetection:
    """Test sensitive parameter detection in signatures."""

    @pytest.mark.parametrize(
        ("signature", "expected_params"),
        [
            ("(username: str, password: str)", ["password"]),
            ("(api_key: str, data: dict)", ["api_key"]),
            ("(user_id: UUID, token: str)", ["token"]),
            ("(user_id: UUID, data: dict)", []),
            ("(secret: str, value: int)", ["secret"]),
        ],
    )
    def test_has_sensitive_parameters(
        self, signature: str, expected_params: list[str]
    ) -> None:
        """Test detection of sensitive parameters in method signatures."""
        found_params = has_sensitive_parameters(signature)
        assert set(found_params) == set(expected_params)


class TestValidateMethodExposure:
    """Test method exposure validation."""

    def test_sensitive_method_exposed(self) -> None:
        """Test validation detects exposed sensitive methods."""
        handler_identity = ModelHandlerIdentifier.from_handler_id("auth-handler")

        errors = validate_method_exposure(
            method_names=["get_api_key", "process_request"],
            handler_identity=handler_identity,
        )

        assert len(errors) == 1
        error = errors[0]
        assert error.error_type == EnumHandlerErrorType.SECURITY_VALIDATION_ERROR
        assert error.rule_id == SecurityRuleId.SENSITIVE_METHOD_EXPOSED
        assert "get_api_key" in error.message
        assert "_get_api_key" in error.remediation_hint

    def test_admin_method_exposed(self) -> None:
        """Test validation detects exposed admin methods."""
        handler_identity = ModelHandlerIdentifier.from_handler_id("admin-handler")

        errors = validate_method_exposure(
            method_names=["admin_delete_user", "process_request"],
            handler_identity=handler_identity,
        )

        # admin_delete_user triggers 2 violations:
        # 1. SECURITY-001 (matches sensitive pattern ^admin_)
        # 2. SECURITY-003 (admin method public)
        assert len(errors) == 2
        rule_ids = {error.rule_id for error in errors}
        assert SecurityRuleId.SENSITIVE_METHOD_EXPOSED in rule_ids
        assert SecurityRuleId.ADMIN_METHOD_PUBLIC in rule_ids
        assert all("admin_delete_user" in error.message for error in errors)

    def test_credential_in_signature(self) -> None:
        """Test validation detects credentials in method signatures."""
        handler_identity = ModelHandlerIdentifier.from_handler_id("auth-handler")

        errors = validate_method_exposure(
            method_names=["authenticate"],
            handler_identity=handler_identity,
            method_signatures={
                "authenticate": "(username: str, password: str) -> bool"
            },
        )

        assert len(errors) == 1
        error = errors[0]
        assert error.rule_id == SecurityRuleId.CREDENTIAL_IN_SIGNATURE
        assert "password" in error.message
        assert "data" in error.remediation_hint

    def test_no_violations(self) -> None:
        """Test validation passes for safe method exposure."""
        handler_identity = ModelHandlerIdentifier.from_handler_id("compute-handler")

        errors = validate_method_exposure(
            method_names=["process_request", "execute_query"],
            handler_identity=handler_identity,
            method_signatures={
                "process_request": "(data: dict) -> Result",
                "execute_query": "(query: str) -> list[dict]",
            },
        )

        assert len(errors) == 0


class TestValidateHandlerSecurity:
    """Test handler security validation."""

    def test_validates_capabilities_dict(self) -> None:
        """Test validation of handler capabilities."""
        handler_identity = ModelHandlerIdentifier.from_handler_id("test-handler")

        capabilities = {
            "operations": ["get_password", "process_data"],
            "protocols": ["ProtocolDatabaseAdapter"],
            "has_fsm": False,
            "method_signatures": {
                "get_password": "() -> str",
                "process_data": "(data: dict) -> Result",
            },
        }

        errors = validate_handler_security(
            handler_identity=handler_identity,
            capabilities=capabilities,
        )

        assert len(errors) == 1
        assert errors[0].rule_id == SecurityRuleId.SENSITIVE_METHOD_EXPOSED
        assert "get_password" in errors[0].message


class TestErrorFormatting:
    """Test error output formatting."""

    def test_format_for_logging(self) -> None:
        """Test structured logging format."""
        error = ModelHandlerValidationError.from_security_violation(
            rule_id=SecurityRuleId.SENSITIVE_METHOD_EXPOSED,
            message="Handler exposes 'get_api_key' method",
            remediation_hint="Prefix with underscore: '_get_api_key'",
            handler_identity=ModelHandlerIdentifier.from_handler_id("auth-handler"),
            file_path="nodes/auth/handlers/handler_authenticate.py",
            line_number=42,
        )

        formatted = error.format_for_logging()
        assert "SECURITY-001" in formatted
        assert "security_validation_error" in formatted
        assert "static_analysis" in formatted
        assert "auth-handler" in formatted
        assert "line_number=42" in formatted or ":42" in formatted
        assert "get_api_key" in formatted

    def test_format_for_ci(self) -> None:
        """Test GitHub Actions annotation format."""
        error = ModelHandlerValidationError.from_security_violation(
            rule_id=SecurityRuleId.SENSITIVE_METHOD_EXPOSED,
            message="Handler exposes 'get_api_key' method",
            remediation_hint="Prefix with underscore: '_get_api_key'",
            handler_identity=ModelHandlerIdentifier.from_handler_id("auth-handler"),
            file_path="nodes/auth/handlers/handler_authenticate.py",
            line_number=42,
        )

        formatted = error.format_for_ci()
        assert formatted.startswith("::error")
        assert "file=nodes/auth/handlers/handler_authenticate.py" in formatted
        assert "line=42" in formatted
        assert "[SECURITY-001]" in formatted

    def test_to_structured_dict(self) -> None:
        """Test JSON serialization."""
        error = ModelHandlerValidationError.from_security_violation(
            rule_id=SecurityRuleId.SENSITIVE_METHOD_EXPOSED,
            message="Handler exposes 'get_api_key' method",
            remediation_hint="Prefix with underscore: '_get_api_key'",
            handler_identity=ModelHandlerIdentifier.from_handler_id("auth-handler"),
        )

        structured = error.to_structured_dict()
        assert structured["error_type"] == "security_validation_error"
        assert structured["rule_id"] == "SECURITY-001"
        assert structured["source_type"] == "static_analysis"
        assert structured["message"] == "Handler exposes 'get_api_key' method"
        assert structured["severity"] == "error"


class TestSecurityRuleIds:
    """Test security rule ID definitions."""

    def test_rule_ids_are_unique(self) -> None:
        """Test all rule IDs are unique."""
        rule_ids = [
            SecurityRuleId.SENSITIVE_METHOD_EXPOSED,
            SecurityRuleId.CREDENTIAL_IN_SIGNATURE,
            SecurityRuleId.ADMIN_METHOD_PUBLIC,
            SecurityRuleId.DECRYPT_METHOD_PUBLIC,
            SecurityRuleId.CREDENTIAL_IN_CONFIG,
            SecurityRuleId.HARDCODED_SECRET,
            SecurityRuleId.INSECURE_CONNECTION,
            SecurityRuleId.INSECURE_PATTERN,
            SecurityRuleId.MISSING_AUTH_CHECK,
            SecurityRuleId.MISSING_INPUT_VALIDATION,
        ]

        assert len(rule_ids) == len(set(rule_ids))

    def test_rule_ids_follow_pattern(self) -> None:
        """Test rule IDs follow naming convention."""
        rule_ids = [
            SecurityRuleId.SENSITIVE_METHOD_EXPOSED,
            SecurityRuleId.CREDENTIAL_IN_SIGNATURE,
            SecurityRuleId.ADMIN_METHOD_PUBLIC,
        ]

        for rule_id in rule_ids:
            assert rule_id.startswith("SECURITY-")
            assert rule_id[9:].isdigit()  # After "SECURITY-"
