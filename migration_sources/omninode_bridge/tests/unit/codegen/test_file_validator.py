#!/usr/bin/env python3
"""
Unit tests for FileValidator.

Tests post-generation file validation to ensure:
- Syntax errors are detected
- Stub patterns are detected
- Missing files are detected
- Valid files pass validation
"""

import tempfile
from pathlib import Path

import pytest

from omninode_bridge.codegen.file_validator import FileValidator


@pytest.mark.asyncio
class TestFileValidator:
    """Test suite for FileValidator."""

    async def test_valid_file_passes(self):
        """Test that valid Python file passes validation."""
        # Create valid Python file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''#!/usr/bin/env python3
"""Valid module."""

def hello():
    """Say hello."""
    return "Hello, world!"
'''
            )
            temp_file = Path(f.name)

        try:
            validator = FileValidator()
            result = await validator.validate_generated_files([temp_file])

            assert result.passed
            assert result.files_validated == 1
            assert result.files_passed == 1
            assert result.files_failed == 0
            assert len(result.issues) == 0
        finally:
            temp_file.unlink()

    async def test_syntax_error_detected(self):
        """Test that syntax errors are detected."""
        # Create file with syntax error
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''#!/usr/bin/env python3
"""Invalid module."""

def broken():
    try:
        # Missing indented block after try
    except Exception:
        pass
'''
            )
            temp_file = Path(f.name)

        try:
            validator = FileValidator()
            result = await validator.validate_generated_files([temp_file])

            assert not result.passed
            assert result.files_validated == 1
            assert result.files_passed == 0
            assert result.files_failed == 1
            assert len(result.syntax_errors) > 0
            assert any(
                "IndentationError" in issue.message or "SyntaxError" in issue.message
                for issue in result.syntax_errors
            )
        finally:
            temp_file.unlink()

    async def test_implementation_required_detected(self):
        """Test that IMPLEMENTATION REQUIRED markers are detected."""
        # Create file with stub marker
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''#!/usr/bin/env python3
"""Module with stub."""

def stub_method():
    """Method with stub."""
    # IMPLEMENTATION REQUIRED
    pass
'''
            )
            temp_file = Path(f.name)

        try:
            validator = FileValidator()
            result = await validator.validate_generated_files([temp_file])

            assert not result.passed
            assert len(result.stub_issues) > 0
            assert any(
                "IMPLEMENTATION REQUIRED" in issue.message
                for issue in result.stub_issues
            )
        finally:
            temp_file.unlink()

    async def test_todo_comment_detected(self):
        """Test that TODO comments are detected."""
        # Create file with TODO
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''#!/usr/bin/env python3
"""Module with TODO."""

def incomplete():
    """Incomplete method."""
    # TODO: Implement this
    pass
'''
            )
            temp_file = Path(f.name)

        try:
            validator = FileValidator()
            result = await validator.validate_generated_files([temp_file])

            assert not result.passed
            assert len(result.stub_issues) > 0
            assert any("TODO" in issue.message for issue in result.stub_issues)
        finally:
            temp_file.unlink()

    async def test_bare_pass_method_detected(self):
        """Test that methods with only pass statement are detected."""
        # Create file with bare pass method
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''#!/usr/bin/env python3
"""Module with stub method."""

class MyClass:
    """Example class."""

    def stub_method(self):
        """Method with only pass."""
        pass
'''
            )
            temp_file = Path(f.name)

        try:
            validator = FileValidator()
            result = await validator.validate_generated_files([temp_file])

            assert not result.passed
            assert len(result.stub_issues) > 0
            assert any(
                "contains only 'pass' statement" in issue.message
                for issue in result.stub_issues
            )
        finally:
            temp_file.unlink()

    async def test_not_implemented_error_detected(self):
        """Test that NotImplementedError is detected."""
        # Create file with NotImplementedError
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''#!/usr/bin/env python3
"""Module with NotImplementedError."""

def stub():
    """Stub method."""
    raise NotImplementedError("Not implemented yet")
'''
            )
            temp_file = Path(f.name)

        try:
            validator = FileValidator()
            result = await validator.validate_generated_files([temp_file])

            assert not result.passed
            assert len(result.stub_issues) > 0
            assert any(
                "NotImplementedError" in issue.message for issue in result.stub_issues
            )
        finally:
            temp_file.unlink()

    async def test_missing_file_detected(self):
        """Test that missing files are detected."""
        # Use non-existent file
        missing_file = Path("/tmp/nonexistent_file_12345.py")

        validator = FileValidator()
        result = await validator.validate_generated_files([missing_file])

        assert not result.passed
        assert len(result.missing_files) == 1
        assert "does not exist" in result.missing_files[0].message

    async def test_multiple_files_validation(self):
        """Test validation of multiple files."""
        # Create two files: one valid, one with stub
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as valid_file:
            valid_file.write(
                """#!/usr/bin/env python3
def valid():
    return "valid"
"""
            )
            valid_file_path = valid_file.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as stub_file:
            stub_file.write(
                """#!/usr/bin/env python3
def stub():
    # IMPLEMENTATION REQUIRED
    pass
"""
            )
            stub_file_path = stub_file.name

        try:
            validator = FileValidator()
            result = await validator.validate_generated_files(
                [Path(valid_file_path), Path(stub_file_path)]
            )

            assert not result.passed  # Should fail due to stub file
            assert result.files_validated == 2
            assert result.files_passed == 1
            assert result.files_failed == 1
            assert len(result.stub_issues) > 0
        finally:
            Path(valid_file_path).unlink()
            Path(stub_file_path).unlink()

    async def test_format_validation_report(self):
        """Test validation report formatting."""
        # Create file with issue
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """#!/usr/bin/env python3
def stub():
    # IMPLEMENTATION REQUIRED
    pass
"""
            )
            temp_file = Path(f.name)

        try:
            validator = FileValidator()
            result = await validator.validate_generated_files([temp_file])

            # Format report
            report = validator.format_validation_report(result)

            assert "POST-GENERATION FILE VALIDATION REPORT" in report
            assert "Overall Status: FAILED" in report
            assert "Files validated: 1" in report
            assert "IMPLEMENTATION REQUIRED" in report
        finally:
            temp_file.unlink()

    async def test_strict_mode(self):
        """Test that strict mode fails on warnings."""
        # Create file with TODO (warning-level issue)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """#!/usr/bin/env python3
def incomplete():
    # TODO: Implement
    pass
"""
            )
            temp_file = Path(f.name)

        try:
            validator = FileValidator()

            # Strict mode should fail
            result_strict = await validator.validate_generated_files(
                [temp_file], strict_mode=True
            )
            assert not result_strict.passed

            # Non-strict mode should also fail (TODO is still an issue)
            result_non_strict = await validator.validate_generated_files(
                [temp_file], strict_mode=False
            )
            assert not result_non_strict.passed
        finally:
            temp_file.unlink()
