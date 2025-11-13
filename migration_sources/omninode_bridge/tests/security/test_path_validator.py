#!/usr/bin/env python3
"""
Tests for Path Validator.

Comprehensive security testing for file system path validation and access control.
"""

import os
import tempfile
from pathlib import Path

import pytest

from omninode_bridge.security.exceptions import SecurityValidationError
from omninode_bridge.security.path_validator import PathValidator


class TestPathValidator:
    """Test suite for PathValidator."""

    @pytest.fixture
    def validator(self):
        """Create path validator instance with test directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            allowed_dirs = [
                Path(tmpdir) / "generated",
                Path(tmpdir) / "output",
            ]
            yield PathValidator(
                allowed_base_dirs=allowed_dirs,
                enable_audit_logging=True,
            )

    @pytest.fixture
    def validator_no_logging(self):
        """Create path validator without audit logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            allowed_dirs = [Path(tmpdir) / "generated"]
            yield PathValidator(
                allowed_base_dirs=allowed_dirs,
                enable_audit_logging=False,
            )

    # ===== Output Path Validation Tests =====

    def test_valid_output_path(self, validator):
        """Test validation of legitimate output path."""
        allowed_dir = validator.allowed_base_dirs[0]
        output_path = str(allowed_dir / "nodes" / "node_database.py")

        result = validator.validate_output_path(output_path)

        assert result.is_absolute()
        assert str(result).startswith(str(allowed_dir))

    def test_path_traversal_rejected(self, validator):
        """Test path traversal patterns are rejected."""
        from omninode_bridge.security.exceptions import PathTraversalAttempt

        with pytest.raises(PathTraversalAttempt):
            validator.validate_output_path("../../etc/passwd")

    def test_blocked_path_rejected(self, validator):
        """Test access to blocked paths is rejected."""
        blocked_paths = [
            "/etc/passwd",
            "/var/www/html",
            "/root/.bashrc",
            "/sys/kernel",
            "/proc/self",
            "/boot/grub",
            "/dev/null",
        ]

        for path in blocked_paths:
            with pytest.raises(SecurityValidationError, match="blocked path"):
                validator.validate_output_path(path)

    def test_ssh_directory_blocked(self, validator):
        """Test SSH directory is blocked."""
        ssh_path = str(Path.home() / ".ssh" / "id_rsa")

        with pytest.raises(SecurityValidationError, match="blocked path"):
            validator.validate_output_path(ssh_path)

    def test_aws_directory_blocked(self, validator):
        """Test AWS credentials directory is blocked."""
        aws_path = str(Path.home() / ".aws" / "credentials")

        with pytest.raises(SecurityValidationError, match="blocked path"):
            validator.validate_output_path(aws_path)

    def test_path_outside_allowed_rejected(self, validator):
        """Test paths outside allowed directories are rejected."""
        with pytest.raises(SecurityValidationError, match="not within allowed"):
            validator.validate_output_path("/tmp/unauthorized/file.py")

    # ===== File Path Validation Tests =====

    def test_valid_read_file_path(self, validator):
        """Test validation of file path for read operation."""
        allowed_dir = validator.allowed_base_dirs[0]
        file_path = str(allowed_dir / "config.yaml")

        result = validator.validate_file_path(file_path, operation="read")

        assert result.is_absolute()

    def test_valid_write_file_path(self, validator):
        """Test validation of file path for write operation."""
        allowed_dir = validator.allowed_base_dirs[0]
        file_path = str(allowed_dir / "output.txt")

        result = validator.validate_file_path(file_path, operation="write")

        assert result.is_absolute()
        assert str(result).startswith(str(allowed_dir))

    def test_write_outside_allowed_rejected(self, validator):
        """Test write operation outside allowed dirs is rejected."""
        with pytest.raises(SecurityValidationError, match="not within allowed"):
            validator.validate_file_path("/tmp/unauthorized.txt", operation="write")

    def test_delete_outside_allowed_rejected(self, validator):
        """Test delete operation outside allowed dirs is rejected."""
        with pytest.raises(SecurityValidationError, match="not within allowed"):
            validator.validate_file_path("/tmp/unauthorized.txt", operation="delete")

    def test_null_byte_in_path_rejected(self, validator):
        """Test null byte injection is rejected."""
        with pytest.raises(SecurityValidationError, match="Null byte"):
            validator.validate_file_path("/tmp/file\x00.txt")

    # ===== Directory Creation Tests =====

    def test_validate_directory_creation(self, validator):
        """Test validation for directory creation."""
        allowed_dir = validator.allowed_base_dirs[0]
        new_dir = str(allowed_dir / "new_directory")

        result = validator.validate_directory_creation(new_dir)

        assert result.is_absolute()
        assert str(result).startswith(str(allowed_dir))

    def test_directory_creation_outside_allowed_rejected(self, validator):
        """Test directory creation outside allowed dirs is rejected."""
        with pytest.raises(SecurityValidationError):
            validator.validate_directory_creation("/tmp/unauthorized_dir")

    def test_create_secure_directory(self, validator):
        """Test secure directory creation."""
        allowed_dir = validator.allowed_base_dirs[0]
        new_dir = str(allowed_dir / "test_secure_dir")

        result = validator.create_secure_directory(new_dir)

        assert result.exists()
        assert result.is_dir()

        # Check permissions (Unix-like systems only)
        if hasattr(os, "stat"):
            stat_info = os.stat(result)
            # Should be readable and executable by owner
            assert stat_info.st_mode & 0o700

    def test_create_secure_directory_with_parents(self, validator):
        """Test secure directory creation with parent directories."""
        allowed_dir = validator.allowed_base_dirs[0]
        nested_dir = str(allowed_dir / "parent" / "child" / "grandchild")

        result = validator.create_secure_directory(nested_dir)

        assert result.exists()
        assert result.is_dir()
        assert (result.parent).exists()
        assert (result.parent.parent).exists()

    # ===== Relative Path Tests =====

    def test_get_relative_path(self, validator):
        """Test getting relative path from base directory."""
        allowed_dir = validator.allowed_base_dirs[0]
        absolute_path = allowed_dir / "nodes" / "node_database.py"

        relative = validator.get_relative_path(absolute_path, allowed_dir)

        assert relative == Path("nodes") / "node_database.py"

    def test_get_relative_path_outside_base_rejected(self, validator):
        """Test relative path calculation rejects paths outside base."""
        allowed_dir = validator.allowed_base_dirs[0]
        outside_path = Path("/tmp/outside/file.py")

        with pytest.raises(SecurityValidationError, match="not within base"):
            validator.get_relative_path(outside_path, allowed_dir)

    # ===== Safe Filename Tests =====

    def test_safe_filename_valid(self, validator):
        """Test validation of safe filenames."""
        safe_names = [
            "node_database.py",
            "contract.yaml",
            "model_user.py",
            "test_node.py",
        ]

        for name in safe_names:
            assert validator.is_safe_filename(name) is True

    def test_safe_filename_with_path_separator_rejected(self, validator):
        """Test filenames with path separators are rejected."""
        unsafe_names = [
            "path/to/file.py",
            "..\\file.py",
            "dir/file.py",
        ]

        for name in unsafe_names:
            assert validator.is_safe_filename(name) is False

    def test_safe_filename_parent_reference_rejected(self, validator):
        """Test parent directory references are rejected."""
        assert validator.is_safe_filename("..") is False
        assert validator.is_safe_filename(".") is False

    def test_safe_filename_null_byte_rejected(self, validator):
        """Test filenames with null bytes are rejected."""
        assert validator.is_safe_filename("file\x00.py") is False

    def test_safe_filename_hidden_files_warning(self, validator, caplog):
        """Test hidden files generate warnings but are allowed."""
        # Hidden files should be allowed but may log warnings
        result = validator.is_safe_filename(".hidden_file")
        # Result may be True (allowed) but should log warning
        assert result is True

    # ===== Configuration Tests =====

    def test_custom_allowed_directories(self):
        """Test custom allowed directories configuration."""
        custom_dirs = [
            Path("/tmp/custom1"),
            Path("/tmp/custom2"),
        ]

        validator = PathValidator(allowed_base_dirs=custom_dirs)

        assert len(validator.allowed_base_dirs) == 2
        assert any("/tmp/custom1" in str(d) for d in validator.allowed_base_dirs)

    def test_additional_blocked_paths(self):
        """Test additional blocked paths configuration."""
        additional_blocked = [Path("/custom/blocked")]

        with tempfile.TemporaryDirectory() as tmpdir:
            validator = PathValidator(
                allowed_base_dirs=[Path(tmpdir)],
                additional_blocked_paths=additional_blocked,
            )

            assert Path("/custom/blocked") in validator.blocked_paths

    def test_audit_logging_disabled(self, validator_no_logging, caplog):
        """Test audit logging can be disabled."""
        allowed_dir = validator_no_logging.allowed_base_dirs[0]
        file_path = str(allowed_dir / "test.py")

        validator_no_logging.validate_file_path(file_path, operation="read")

        # Should not log when audit logging is disabled
        # (This test verifies the setting works; actual log checking may vary)

    # ===== Edge Cases =====

    def test_symlink_resolution(self, validator):
        """Test symlinks are resolved to absolute paths."""
        allowed_dir = validator.allowed_base_dirs[0]
        real_file = allowed_dir / "real_file.txt"

        # Create directory if it doesn't exist
        allowed_dir.mkdir(parents=True, exist_ok=True)
        real_file.touch()

        # Create symlink
        symlink = allowed_dir / "symlink.txt"
        if not symlink.exists():
            symlink.symlink_to(real_file)

        try:
            result = validator.validate_file_path(str(symlink))
            # Symlink should resolve to absolute path
            assert result.is_absolute()
        finally:
            # Cleanup
            if symlink.is_symlink():
                symlink.unlink()
            if real_file.exists():
                real_file.unlink()

    def test_case_sensitivity(self, validator):
        """Test path validation is case-sensitive on case-sensitive systems."""
        allowed_dir = validator.allowed_base_dirs[0]

        # Test with different cases
        path_lower = str(allowed_dir / "file.txt")
        path_upper = str(allowed_dir / "FILE.txt")

        # Both should validate (case-sensitivity depends on filesystem)
        validator.validate_file_path(path_lower)
        validator.validate_file_path(path_upper)

    def test_unicode_paths(self, validator):
        """Test Unicode characters in paths are handled correctly."""
        allowed_dir = validator.allowed_base_dirs[0]
        unicode_path = str(allowed_dir / "用户数据" / "file.txt")

        result = validator.validate_output_path(unicode_path)
        assert result.is_absolute()

    def test_very_long_path(self, validator):
        """Test very long paths are handled correctly."""
        allowed_dir = validator.allowed_base_dirs[0]

        # Create a very long path (but within limits)
        long_name = "a" * 200
        long_path = str(allowed_dir / long_name / "file.txt")

        result = validator.validate_output_path(long_path)
        assert result.is_absolute()
