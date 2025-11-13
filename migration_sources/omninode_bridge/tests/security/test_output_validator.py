#!/usr/bin/env python3
"""
Tests for Output Validator.

Comprehensive security testing for generated code validation using AST analysis.
"""

import pytest

from omninode_bridge.security.output_validator import OutputValidator


class TestOutputValidator:
    """Test suite for OutputValidator."""

    @pytest.fixture
    def validator(self):
        """Create output validator instance."""
        return OutputValidator(strict_mode=False)

    @pytest.fixture
    def strict_validator(self):
        """Create strict mode validator instance."""
        return OutputValidator(strict_mode=True)

    # ===== Safe Code Tests =====

    def test_safe_simple_code(self, validator):
        """Test validation of simple safe code."""
        code = """
def hello_world():
    return "Hello, World!"
"""
        report = validator.validate_generated_code(code)

        assert report.is_safe is True
        assert len(report.dangerous_patterns) == 0
        assert report.quality_score == 1.0

    def test_safe_onex_node_code(self, validator):
        """Test validation of ONEX node code."""
        code = """
from typing import Any
from pydantic import BaseModel

class NodeDatabaseEffect:
    async def execute_effect(self, contract: BaseModel) -> Any:
        result = await self._perform_query(contract)
        return {"status": "success", "data": result}
"""
        report = validator.validate_generated_code(code)

        assert report.is_safe is True
        assert report.quality_score >= 0.9

    # ===== Dangerous Import Tests =====

    def test_dangerous_import_os_system(self, validator):
        """Test detection of os.system import."""
        code = """
import os

os.system('ls -la')
"""
        report = validator.validate_generated_code(code)

        assert report.is_safe is False
        assert len(report.dangerous_patterns) > 0
        assert any("os.system" in issue.message for issue in report.dangerous_patterns)

    def test_dangerous_import_subprocess(self, validator):
        """Test detection of subprocess imports."""
        dangerous_codes = [
            "import subprocess\nsubprocess.call(['ls', '-la'])",
            "from subprocess import run\nrun(['whoami'])",
            "from subprocess import Popen\nPopen(['cat', '/etc/passwd'])",
        ]

        for code in dangerous_codes:
            report = validator.validate_generated_code(code)
            assert len(report.dangerous_patterns) > 0
            assert any(
                "subprocess" in issue.message for issue in report.dangerous_patterns
            )

    def test_dangerous_import_exec_eval(self, validator):
        """Test detection of exec/eval imports."""
        code = """
# Dangerous dynamic code execution
exec("print('hello')")
result = eval("2 + 2")
"""
        report = validator.validate_generated_code(code)

        assert report.is_safe is False  # Critical issues
        assert len(report.dangerous_patterns) >= 2
        critical_issues = [
            i for i in report.dangerous_patterns if i.severity == "critical"
        ]
        assert len(critical_issues) >= 2

    def test_dangerous_import_pickle(self, validator):
        """Test detection of pickle import (can execute code)."""
        code = """
import pickle

data = pickle.loads(untrusted_data)
"""
        report = validator.validate_generated_code(code)

        assert len(report.dangerous_patterns) > 0
        assert any("pickle" in issue.message for issue in report.dangerous_patterns)

    # ===== Dangerous Function Call Tests =====

    def test_eval_function_detected(self, validator):
        """Test detection of eval() function calls."""
        code = """
def process_input(user_input):
    result = eval(user_input)  # Dangerous!
    return result
"""
        report = validator.validate_generated_code(code)

        assert report.is_safe is False
        critical_issues = [
            i for i in report.dangerous_patterns if i.severity == "critical"
        ]
        assert len(critical_issues) > 0

    def test_exec_function_detected(self, validator):
        """Test detection of exec() function calls."""
        code = """
def run_code(code_string):
    exec(code_string)  # Dangerous!
"""
        report = validator.validate_generated_code(code)

        assert report.is_safe is False
        assert any("exec" in issue.message for issue in report.dangerous_patterns)

    def test_compile_function_detected(self, validator):
        """Test detection of compile() function calls."""
        code = """
code_obj = compile("print('hello')", '<string>', 'exec')
"""
        report = validator.validate_generated_code(code)

        assert len(report.dangerous_patterns) > 0

    def test_dynamic_import_detected(self, validator):
        """Test detection of __import__() calls."""
        code = """
module = __import__('os')
module.system('ls')
"""
        report = validator.validate_generated_code(code)

        assert len(report.dangerous_patterns) >= 2  # __import__ + os.system

    # ===== File Operation Tests =====

    def test_file_read_allowed(self, validator):
        """Test file read operations are allowed (low severity)."""
        code = """
def read_config():
    with open('config.yaml', 'r') as f:
        return f.read()
"""
        report = validator.validate_generated_code(code)

        # Read operations should not trigger high-severity issues
        high_severity = [
            i for i in report.dangerous_patterns if i.severity in ["high", "critical"]
        ]
        assert len(high_severity) == 0

    def test_file_write_detected(self, validator):
        """Test file write operations are detected."""
        code = """
def save_data(data):
    with open('output.txt', 'w') as f:
        f.write(data)
"""
        report = validator.validate_generated_code(code)

        # Should detect file write (low severity - may be legitimate)
        assert len(report.dangerous_patterns) > 0
        assert any(
            "write" in issue.message.lower() for issue in report.dangerous_patterns
        )

    def test_file_append_detected(self, validator):
        """Test file append operations are detected."""
        code = """
with open('log.txt', 'a') as f:
    f.write('log entry')
"""
        report = validator.validate_generated_code(code)

        assert len(report.dangerous_patterns) > 0

    # ===== Hardcoded Credentials Tests =====

    def test_hardcoded_password_detected(self, validator):
        """Test detection of hardcoded passwords."""
        code = """
password = "SuperSecret123!"
db_password = 'admin'
"""
        report = validator.validate_generated_code(code)

        assert len(report.dangerous_patterns) >= 2
        assert any(
            "password" in issue.message.lower() for issue in report.dangerous_patterns
        )

    def test_hardcoded_api_key_detected(self, validator):
        """Test detection of hardcoded API keys."""
        code = """
api_key = "sk-proj-abc123def456"
API_KEY = 'production-key-xyz789'
"""
        report = validator.validate_generated_code(code)

        assert len(report.dangerous_patterns) >= 2
        assert any(
            "api key" in issue.message.lower() for issue in report.dangerous_patterns
        )

    def test_hardcoded_secret_detected(self, validator):
        """Test detection of hardcoded secrets."""
        code = """
secret = "my-secret-token"
SECRET_KEY = 'django-secret-key-production'
"""
        report = validator.validate_generated_code(code)

        assert len(report.dangerous_patterns) >= 2

    def test_hardcoded_token_detected(self, validator):
        """Test detection of hardcoded tokens."""
        code = """
token = "bearer-token-abc123"
auth_token = 'github-pat-xyz789'
"""
        report = validator.validate_generated_code(code)

        assert len(report.dangerous_patterns) >= 2

    def test_hardcoded_aws_credentials_detected(self, validator):
        """Test detection of hardcoded AWS credentials."""
        code = """
aws_access_key_id = "AKIAIOSFODNN7EXAMPLE"
aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
"""
        report = validator.validate_generated_code(code)

        assert len(report.dangerous_patterns) >= 2

    # ===== Sensitive Path Tests =====

    def test_sensitive_path_etc_passwd(self, validator):
        """Test detection of /etc/passwd access."""
        code = """
with open('/etc/passwd', 'r') as f:
    users = f.read()
"""
        report = validator.validate_generated_code(code)

        assert len(report.dangerous_patterns) > 0
        assert any(
            "/etc/passwd" in issue.message for issue in report.dangerous_patterns
        )

    def test_sensitive_path_ssh(self, validator):
        """Test detection of SSH directory access."""
        code = """
ssh_key = open('~/.ssh/id_rsa').read()
"""
        report = validator.validate_generated_code(code)

        assert len(report.dangerous_patterns) > 0

    def test_sensitive_path_aws(self, validator):
        """Test detection of AWS config directory access."""
        code = """
credentials = open('~/.aws/credentials').read()
"""
        report = validator.validate_generated_code(code)

        assert len(report.dangerous_patterns) > 0

    # ===== Quality Score Tests =====

    def test_quality_score_decreases_with_issues(self, validator):
        """Test quality score decreases with security issues."""
        # Safe code
        safe_code = "def hello(): return 'hello'"
        safe_report = validator.validate_generated_code(safe_code)

        # Code with low severity issues
        low_severity_code = """
with open('file.txt', 'w') as f:
    f.write('data')
"""
        low_report = validator.validate_generated_code(low_severity_code)

        # Code with critical issues
        critical_code = """
exec(user_input)
eval(malicious_code)
"""
        critical_report = validator.validate_generated_code(critical_code)

        assert safe_report.quality_score > low_report.quality_score
        assert low_report.quality_score > critical_report.quality_score

    def test_quality_score_bounded(self, validator):
        """Test quality score is bounded between 0.0 and 1.0."""
        # Many critical issues
        code = """
exec('bad')
eval('bad')
compile('bad', '', 'exec')
password = 'hardcoded'
api_key = 'hardcoded'
"""
        report = validator.validate_generated_code(code)

        assert 0.0 <= report.quality_score <= 1.0

    # ===== Syntax Error Tests =====

    def test_syntax_error_handled(self, validator):
        """Test syntax errors are handled gracefully."""
        code = """
def broken_function(
    # Missing closing parenthesis
"""
        report = validator.validate_generated_code(code)

        assert report.is_safe is False
        assert len(report.warnings) > 0
        assert any("syntax" in w.lower() for w in report.warnings)
        assert report.quality_score == 0.0

    # ===== Needs Review Tests =====

    def test_needs_review_for_high_severity(self, validator):
        """Test needs_review flag for high severity issues."""
        code = """
import subprocess
subprocess.call(['ls', '-la'])
"""
        report = validator.validate_generated_code(code)

        assert report.needs_review is True

    def test_needs_review_for_critical_severity(self, validator):
        """Test needs_review flag for critical severity issues."""
        code = """
exec(user_input)
"""
        report = validator.validate_generated_code(code)

        assert report.needs_review is True

    def test_no_review_for_safe_code(self, validator):
        """Test no review needed for safe code."""
        code = """
def process_data(data):
    return {"status": "success", "data": data}
"""
        report = validator.validate_generated_code(code)

        assert report.needs_review is False

    # ===== Strict Mode Tests =====

    def test_strict_mode_fails_on_critical(self, strict_validator):
        """Test strict mode fails on critical issues."""
        code = """
exec(malicious_code)
"""
        report = strict_validator.validate_generated_code(code)

        assert report.is_safe is False

    # ===== Report Generation Tests =====

    def test_generate_report_format(self, validator):
        """Test report generation produces readable output."""
        code = """
password = 'hardcoded'
exec('dangerous')
"""
        report = validator.validate_generated_code(code)
        report_text = validator.generate_report(report)

        assert "Security Validation Report" in report_text
        assert "Status:" in report_text
        assert "Quality Score:" in report_text
        assert "Security Issues" in report_text

    # ===== Unsupported Language Tests =====

    def test_unsupported_language_warning(self, validator):
        """Test validation of unsupported language."""
        code = "console.log('JavaScript');"
        report = validator.validate_generated_code(code, language="javascript")

        assert len(report.warnings) > 0
        assert "No validation available" in report.warnings[0]
        assert report.quality_score < 1.0  # Lower score for unsupported
        assert report.needs_review is True

    # ===== Edge Cases =====

    def test_empty_code(self, validator):
        """Test validation of empty code."""
        code = ""
        report = validator.validate_generated_code(code)

        assert report.is_safe is True
        assert len(report.dangerous_patterns) == 0

    def test_comments_only_code(self, validator):
        """Test validation of comments-only code."""
        code = """
# This is a comment
# Another comment
"""
        report = validator.validate_generated_code(code)

        assert report.is_safe is True
