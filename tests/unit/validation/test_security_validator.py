# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for ValidatorSecurity (OMN-1277).

Tests the contract-driven security validator that extends ValidatorBase.
"""

from pathlib import Path
from textwrap import dedent

import pytest

from omnibase_core.enums import EnumSeverity
from omnibase_core.models.contracts.subcontracts.model_validator_subcontract import (
    ModelValidatorSubcontract,
)
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.validation.validator_security import ValidatorSecurity


class TestValidatorSecurity:
    """Tests for the contract-driven ValidatorSecurity class."""

    @pytest.fixture
    def contract(self) -> ModelValidatorSubcontract:
        """Create a test contract with all rules enabled.

        NOTE: Patterns are now read from contract.rules[].parameters instead of
        being hardcoded in the validator (OMN-1277). All rules must include their
        patterns in the parameters field.
        """
        return ModelValidatorSubcontract(
            version=ModelSemVer(major=1, minor=0, patch=0),
            validator_id="security",
            validator_name="ONEX Security Validator",
            validator_description="Test security validator",
            target_patterns=["**/*.py"],
            exclude_patterns=[],
            rules=[
                {
                    "rule_id": "sensitive_method_exposed",
                    "description": "Detects sensitive methods",
                    "severity": EnumSeverity.ERROR,
                    "enabled": True,
                    "parameters": {
                        "patterns": [
                            "^get_password$",
                            "^get_secret$",
                            "^get_token$",
                            "^get_api_key$",
                            "^get_credential",
                            "^fetch_password$",
                            "^fetch_secret$",
                            "^fetch_token$",
                            "^validate_password$",
                            "^check_password$",
                            "^verify_password$",
                        ],
                    },
                },
                {
                    "rule_id": "credential_in_signature",
                    "description": "Detects credential in signatures",
                    "severity": EnumSeverity.ERROR,
                    "enabled": True,
                    "parameters": {
                        "sensitive_params": [
                            "password",
                            "secret",
                            "token",
                            "api_key",
                            "apikey",
                            "access_key",
                            "private_key",
                            "credential",
                            "auth_token",
                            "bearer_token",
                            "decrypt_key",
                            "encryption_key",
                        ],
                    },
                },
                {
                    "rule_id": "admin_method_public",
                    "description": "Detects admin methods",
                    "severity": EnumSeverity.WARNING,
                    "enabled": True,
                    "parameters": {
                        "patterns": [
                            "^admin_",
                            "^internal_",
                        ],
                    },
                },
                {
                    "rule_id": "decrypt_method_public",
                    "description": "Detects decrypt methods",
                    "severity": EnumSeverity.WARNING,
                    "enabled": True,
                    "parameters": {
                        "patterns": [
                            "^decrypt_",
                        ],
                    },
                },
            ],
            suppression_comments=["# ONEX_EXCLUDE: security", "# security-ok:"],
            severity_default=EnumSeverity.ERROR,
            fail_on_error=True,
            fail_on_warning=False,
        )

    def test_detects_sensitive_method(
        self, contract: ModelValidatorSubcontract, tmp_path: Path
    ) -> None:
        """Test that ValidatorSecurity detects sensitive method names."""
        test_file = tmp_path / "handler.py"
        test_file.write_text(
            dedent('''
            class DataHandler:
                def get_password(self) -> str:
                    """Sensitive method that should be private."""
                    return "secret"

                def process_request(self, data: dict) -> dict:
                    """Safe method."""
                    return data
            ''').strip()
        )

        validator = ValidatorSecurity(contract=contract)
        result = validator.validate_file(test_file)

        assert not result.is_valid
        assert len(result.issues) == 1
        issue = result.issues[0]
        assert issue.code == "sensitive_method_exposed"
        assert issue.severity == EnumSeverity.ERROR
        assert issue.file_path == test_file
        assert issue.line_number == 2
        assert "get_password" in issue.message
        assert "DataHandler" in issue.message

    def test_detects_credential_in_signature(
        self, contract: ModelValidatorSubcontract, tmp_path: Path
    ) -> None:
        """Test that ValidatorSecurity detects credentials in method signatures."""
        test_file = tmp_path / "handler.py"
        test_file.write_text(
            dedent('''
            class DataHandler:
                def authenticate(self, username: str, password: str) -> bool:
                    """Method with sensitive parameter."""
                    return True
            ''').strip()
        )

        validator = ValidatorSecurity(contract=contract)
        result = validator.validate_file(test_file)

        assert not result.is_valid
        assert len(result.issues) == 1
        issue = result.issues[0]
        assert issue.code == "credential_in_signature"
        assert "password" in issue.message
        assert issue.line_number == 2

    def test_detects_credential_in_kwonly_args(
        self, contract: ModelValidatorSubcontract, tmp_path: Path
    ) -> None:
        """Test detection of sensitive keyword-only parameters."""
        test_file = tmp_path / "handler.py"
        test_file.write_text(
            dedent('''
            class DataHandler:
                def authenticate(self, *, username: str, password: str) -> bool:
                    """Method with sensitive kwonly parameter."""
                    return True
            ''').strip()
        )

        validator = ValidatorSecurity(contract=contract)
        result = validator.validate_file(test_file)

        assert not result.is_valid
        assert len(result.issues) == 1
        assert result.issues[0].code == "credential_in_signature"
        assert "password" in result.issues[0].message

    def test_detects_credential_in_posonly_args(
        self, contract: ModelValidatorSubcontract, tmp_path: Path
    ) -> None:
        """Test detection of sensitive positional-only parameters."""
        test_file = tmp_path / "handler.py"
        test_file.write_text(
            dedent('''
            class DataHandler:
                def authenticate(self, password: str, /) -> bool:
                    """Method with sensitive posonly parameter."""
                    return True
            ''').strip()
        )

        validator = ValidatorSecurity(contract=contract)
        result = validator.validate_file(test_file)

        assert not result.is_valid
        assert len(result.issues) == 1
        assert result.issues[0].code == "credential_in_signature"
        assert "password" in result.issues[0].message

    def test_detects_credential_in_vararg(
        self, contract: ModelValidatorSubcontract, tmp_path: Path
    ) -> None:
        """Test detection of sensitive *args parameter name."""
        test_file = tmp_path / "handler.py"
        test_file.write_text(
            dedent('''
            class DataHandler:
                def collect_secrets(self, *secret) -> list:
                    """Method with sensitive vararg name."""
                    return list(secret)
            ''').strip()
        )

        validator = ValidatorSecurity(contract=contract)
        result = validator.validate_file(test_file)

        assert not result.is_valid
        assert len(result.issues) == 1
        assert result.issues[0].code == "credential_in_signature"
        assert "secret" in result.issues[0].message

    def test_detects_credential_in_kwarg(
        self, contract: ModelValidatorSubcontract, tmp_path: Path
    ) -> None:
        """Test detection of sensitive **kwargs parameter name."""
        test_file = tmp_path / "handler.py"
        test_file.write_text(
            dedent('''
            class DataHandler:
                def process_auth(self, **password) -> dict:
                    """Method with sensitive kwarg name."""
                    return dict(password)
            ''').strip()
        )

        validator = ValidatorSecurity(contract=contract)
        result = validator.validate_file(test_file)

        assert not result.is_valid
        assert len(result.issues) == 1
        assert result.issues[0].code == "credential_in_signature"
        assert "password" in result.issues[0].message

    def test_detects_admin_method(
        self, contract: ModelValidatorSubcontract, tmp_path: Path
    ) -> None:
        """Test that ValidatorSecurity detects admin methods."""
        test_file = tmp_path / "handler.py"
        test_file.write_text(
            dedent('''
            class AdminHandler:
                def admin_delete_user(self, user_id: str) -> None:
                    """Admin method exposed publicly."""
                    pass
            ''').strip()
        )

        validator = ValidatorSecurity(contract=contract)
        result = validator.validate_file(test_file)

        # admin_method_public is WARNING, so result is valid (fail_on_warning=False)
        assert result.is_valid
        assert len(result.issues) == 1
        issue = result.issues[0]
        assert issue.code == "admin_method_public"
        assert issue.severity == EnumSeverity.WARNING

    def test_suppression_comment_works(
        self, contract: ModelValidatorSubcontract, tmp_path: Path
    ) -> None:
        """Test that suppression comments suppress violations."""
        test_file = tmp_path / "handler.py"
        test_file.write_text(
            dedent('''
            class DataHandler:
                def get_password(self) -> str:  # ONEX_EXCLUDE: security
                    """Suppressed sensitive method."""
                    return "secret"

                def get_secret(self) -> str:  # security-ok: required for API
                    """Also suppressed."""
                    return "secret"

                def get_token(self) -> str:
                    """Not suppressed - should be detected."""
                    return "token"
            ''').strip()
        )

        validator = ValidatorSecurity(contract=contract)
        result = validator.validate_file(test_file)

        # Only get_token should be detected (others are suppressed)
        assert len(result.issues) == 1
        assert "get_token" in result.issues[0].message

    def test_fail_on_error_behavior(
        self, contract: ModelValidatorSubcontract, tmp_path: Path
    ) -> None:
        """Test that fail_on_error controls result validity."""
        test_file = tmp_path / "handler.py"
        test_file.write_text(
            dedent("""
            class DataHandler:
                def get_password(self) -> str:
                    return "secret"
            """).strip()
        )

        # With fail_on_error=True (default)
        validator = ValidatorSecurity(contract=contract)
        result = validator.validate_file(test_file)
        assert not result.is_valid

        # With fail_on_error=False
        contract_no_fail = ModelValidatorSubcontract(
            version=ModelSemVer(major=1, minor=0, patch=0),
            validator_id="security",
            validator_name="ONEX Security Validator",
            validator_description="Test security validator",
            rules=[
                {
                    "rule_id": "sensitive_method_exposed",
                    "description": "Detects sensitive methods",
                    "severity": EnumSeverity.ERROR,
                    "enabled": True,
                    "parameters": {
                        "patterns": [
                            "^get_password$",
                            "^get_secret$",
                            "^get_token$",
                        ],
                    },
                },
            ],
            fail_on_error=False,
            fail_on_warning=False,
        )
        validator_no_fail = ValidatorSecurity(contract=contract_no_fail)
        result_no_fail = validator_no_fail.validate_file(test_file)
        # Still finds issues, but result is valid
        assert result_no_fail.is_valid
        assert len(result_no_fail.issues) == 1

    def test_skips_private_methods(
        self, contract: ModelValidatorSubcontract, tmp_path: Path
    ) -> None:
        """Test that private methods (underscore prefix) are skipped."""
        test_file = tmp_path / "handler.py"
        test_file.write_text(
            dedent('''
            class DataHandler:
                def _get_password(self) -> str:
                    """Private method - should not trigger violation."""
                    return "secret"

                def __internal_token(self) -> str:
                    """Protected method - should not trigger violation."""
                    return "token"
            ''').strip()
        )

        validator = ValidatorSecurity(contract=contract)
        result = validator.validate_file(test_file)

        assert result.is_valid
        assert len(result.issues) == 0

    def test_disabled_rule_not_reported(self, tmp_path: Path) -> None:
        """Test that disabled rules don't produce issues."""
        contract = ModelValidatorSubcontract(
            version=ModelSemVer(major=1, minor=0, patch=0),
            validator_id="security",
            validator_name="ONEX Security Validator",
            validator_description="Test security validator",
            rules=[
                {
                    "rule_id": "sensitive_method_exposed",
                    "description": "Detects sensitive methods",
                    "severity": EnumSeverity.ERROR,
                    "enabled": False,
                    "parameters": {
                        "patterns": [
                            "^get_password$",
                            "^get_secret$",
                            "^get_token$",
                        ],
                    },
                },
            ],
        )

        test_file = tmp_path / "handler.py"
        test_file.write_text(
            dedent("""
            class DataHandler:
                def get_password(self) -> str:
                    return "secret"
            """).strip()
        )

        validator = ValidatorSecurity(contract=contract)
        result = validator.validate_file(test_file)

        assert result.is_valid
        assert len(result.issues) == 0

    def test_issue_has_correct_structure(
        self, contract: ModelValidatorSubcontract, tmp_path: Path
    ) -> None:
        """Test that issues have all required fields correctly populated."""
        test_file = tmp_path / "handler.py"
        test_file.write_text(
            dedent("""
            class DataHandler:
                def get_password(self) -> str:
                    return "secret"
            """).strip()
        )

        validator = ValidatorSecurity(contract=contract)
        result = validator.validate_file(test_file)

        assert len(result.issues) == 1
        issue = result.issues[0]

        # Check all required fields
        assert issue.severity == EnumSeverity.ERROR
        assert issue.message is not None and len(issue.message) > 0
        assert issue.code == "sensitive_method_exposed"
        assert issue.file_path == test_file
        assert issue.line_number == 2
        assert issue.rule_name == "sensitive_method_exposed"
        assert issue.suggestion is not None and "underscore" in issue.suggestion.lower()
        assert issue.context is not None
        assert issue.context.get("class_name") == "DataHandler"
        assert issue.context.get("violation_type") == "sensitive_method_exposed"

    def test_multiple_violations_in_one_file(
        self, contract: ModelValidatorSubcontract, tmp_path: Path
    ) -> None:
        """Test detection of multiple violations in a single file."""
        test_file = tmp_path / "handler.py"
        test_file.write_text(
            dedent("""
            class DataHandler:
                def get_password(self) -> str:
                    return "secret"

                def authenticate(self, username: str, password: str) -> bool:
                    return True

                def admin_reset(self) -> None:
                    pass
            """).strip()
        )

        validator = ValidatorSecurity(contract=contract)
        result = validator.validate_file(test_file)

        # Should find: sensitive_method_exposed, credential_in_signature, admin_method_public
        assert len(result.issues) == 3

        codes = {issue.code for issue in result.issues}
        assert "sensitive_method_exposed" in codes
        assert "credential_in_signature" in codes
        assert "admin_method_public" in codes

    def test_syntax_error_file_skipped(
        self, contract: ModelValidatorSubcontract, tmp_path: Path
    ) -> None:
        """Test that files with syntax errors are skipped gracefully."""
        test_file = tmp_path / "bad_syntax.py"
        test_file.write_text("class Broken(\n    def method self):  # Invalid syntax")

        validator = ValidatorSecurity(contract=contract)
        result = validator.validate_file(test_file)

        assert result.is_valid
        assert len(result.issues) == 0

    def test_decrypt_method_detected(
        self, contract: ModelValidatorSubcontract, tmp_path: Path
    ) -> None:
        """Test that decrypt methods are detected."""
        test_file = tmp_path / "handler.py"
        test_file.write_text(
            dedent("""
            class CryptoHandler:
                def decrypt_message(self, data: bytes) -> str:
                    return data.decode()
            """).strip()
        )

        validator = ValidatorSecurity(contract=contract)
        result = validator.validate_file(test_file)

        assert len(result.issues) == 1
        assert result.issues[0].code == "decrypt_method_public"
        assert result.issues[0].severity == EnumSeverity.WARNING

    def test_file_with_no_classes(
        self, contract: ModelValidatorSubcontract, tmp_path: Path
    ) -> None:
        """Test that files with only module-level functions are handled gracefully.

        ValidatorSecurity only checks class methods, so module-level functions
        should not trigger violations even if they have sensitive names.
        """
        test_file = tmp_path / "module_only.py"
        test_file.write_text(
            dedent('''
            def get_password() -> str:
                """Module-level function - not checked by validator."""
                return "secret"

            def process_data(data: dict) -> dict:
                """Another module-level function."""
                return data
            ''').strip()
        )

        validator = ValidatorSecurity(contract=contract)
        result = validator.validate_file(test_file)

        # Module-level functions are not checked (only class methods)
        assert result.is_valid
        assert len(result.issues) == 0
