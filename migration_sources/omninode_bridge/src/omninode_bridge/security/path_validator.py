#!/usr/bin/env python3
"""
Path Validation for Security Hardening.

Validates file system paths to prevent path traversal attacks and unauthorized access.

ONEX v2.0 Compliance:
- Path traversal prevention
- Allowlist-based access control
- Audit logging for file operations
- Configurable security policies
"""

import logging
from pathlib import Path
from typing import ClassVar, Optional

from omninode_bridge.security.exceptions import (
    BlockedPathAccessError,
    NullByteInPathError,
    PathNotAllowedError,
    PathTraversalAttempt,
    PathValidationError,
)

# Configure logger
logger = logging.getLogger(__name__)


class PathValidator:
    """
    Validates file system paths for security.

    Features:
    - Path traversal prevention
    - Allowlist-based access control
    - Blocklist for sensitive directories
    - Path resolution and canonicalization
    - Audit logging for all operations
    """

    # Default allowed base directories
    DEFAULT_ALLOWED_BASE_DIRS: ClassVar[list[Path]] = [
        Path("/tmp/omninode_generated"),
        Path("./generated_nodes"),
        Path("./output"),
        Path("./workspace"),
    ]

    # Blocked paths (system-critical directories)
    BLOCKED_PATHS: ClassVar[list[Path]] = [
        Path("/etc"),
        Path("/var"),
        Path("/root"),
        Path("/sys"),
        Path("/proc"),
        Path("/boot"),
        Path("/dev"),
    ]

    # Blocked user directories
    BLOCKED_USER_DIRS: ClassVar[list[str]] = [
        ".ssh",
        ".aws",
        ".config",
        ".gnupg",
    ]

    def __init__(
        self,
        allowed_base_dirs: Optional[list[Path]] = None,
        additional_blocked_paths: Optional[list[Path]] = None,
        enable_audit_logging: bool = True,
    ):
        """
        Initialize path validator.

        Args:
            allowed_base_dirs: List of allowed base directories. If None, uses defaults.
            additional_blocked_paths: Additional paths to block beyond defaults.
            enable_audit_logging: Enable audit logging for file operations.
        """
        self.allowed_base_dirs = (
            allowed_base_dirs if allowed_base_dirs else self.DEFAULT_ALLOWED_BASE_DIRS
        )
        self.blocked_paths = self.BLOCKED_PATHS.copy()
        if additional_blocked_paths:
            self.blocked_paths.extend(additional_blocked_paths)

        self.enable_audit_logging = enable_audit_logging

        # Resolve allowed base directories to absolute paths
        self.allowed_base_dirs = [path.resolve() for path in self.allowed_base_dirs]

        # Resolve blocked paths to absolute paths (handles symlinks like /etc -> /private/etc)
        self.blocked_paths = [path.resolve() for path in self.blocked_paths]

        # Add user home blocked directories
        home = Path.home()
        for blocked_dir in self.BLOCKED_USER_DIRS:
            self.blocked_paths.append((home / blocked_dir).resolve())

        # Log configuration
        if self.enable_audit_logging:
            logger.info(
                "PathValidator initialized",
                extra={
                    "allowed_base_dirs": [str(p) for p in self.allowed_base_dirs],
                    "blocked_paths_count": len(self.blocked_paths),
                },
            )

    def validate_output_path(self, output_dir: str) -> Path:
        """
        Validate and resolve output directory path.

        Args:
            output_dir: Output directory path to validate

        Returns:
            Resolved absolute Path object

        Raises:
            SecurityException: If path is invalid or blocked
        """
        try:
            # Check for path traversal patterns (extra safety check)
            if ".." in output_dir:
                raise PathTraversalAttempt(
                    "Path traversal pattern detected in output directory",
                    path=output_dir,
                )

            # Convert to Path and resolve
            path = Path(output_dir).resolve()

            # Check if path is within allowed base directories (check first for explicitly allowed paths)
            if not self._is_path_allowed(path):
                # Only check blocked paths if not explicitly allowed
                self._check_blocked_paths(path)
                raise PathNotAllowedError(
                    f"Output path {path} is not within allowed directories",
                    path=str(path),
                    allowed_dirs=[str(p) for p in self.allowed_base_dirs],
                )

            # Audit log
            if self.enable_audit_logging:
                logger.info(
                    "Output path validated",
                    extra={
                        "path": str(path),
                        "original_input": output_dir,
                    },
                )

            return path

        except Exception as e:
            if self.enable_audit_logging:
                logger.error(
                    "Path validation failed",
                    extra={
                        "path": output_dir,
                        "error": str(e),
                    },
                )
            raise

    def validate_file_path(self, file_path: str, operation: str = "read") -> Path:
        """
        Validate file path for read/write operations.

        Args:
            file_path: File path to validate
            operation: Operation type ("read", "write", "delete")

        Returns:
            Resolved absolute Path object

        Raises:
            SecurityException: If path is invalid or blocked
        """
        try:
            # Check for null bytes BEFORE path operations
            if "\x00" in file_path:
                raise NullByteInPathError(
                    "Null byte detected in file path", path=file_path
                )

            # Check for path traversal
            if ".." in file_path:
                raise PathTraversalAttempt(
                    f"Path traversal pattern detected in file path: {file_path}",
                    path=file_path,
                )

            # Convert to Path and resolve
            path = Path(file_path).resolve()

            # For write/delete operations, ensure path is within allowed directories
            if operation in ["write", "delete"]:
                if not self._is_path_allowed(path):
                    # Check blocked paths only if not in allowed directories
                    self._check_blocked_paths(path)
                    raise PathNotAllowedError(
                        f"File path {path} is not within allowed directories for {operation} operation",
                        path=str(path),
                        allowed_dirs=[str(p) for p in self.allowed_base_dirs],
                    )
            # For read operations, just check if path is blocked
            else:
                # Check if path is in allowed directories
                if not self._is_path_allowed(path):
                    # If not in allowed dirs, check if it's blocked
                    self._check_blocked_paths(path)

            # Audit log
            if self.enable_audit_logging:
                logger.info(
                    "File path validated",
                    extra={
                        "path": str(path),
                        "operation": operation,
                        "original_input": file_path,
                    },
                )

            return path

        except Exception as e:
            if self.enable_audit_logging:
                logger.error(
                    "File path validation failed",
                    extra={
                        "path": file_path,
                        "operation": operation,
                        "error": str(e),
                    },
                )
            raise

    def validate_directory_creation(self, directory_path: str) -> Path:
        """
        Validate directory path before creation.

        Args:
            directory_path: Directory path to validate

        Returns:
            Resolved absolute Path object

        Raises:
            SecurityException: If path is invalid or blocked
        """
        path = self.validate_output_path(directory_path)

        # Additional checks for directory creation
        if path.exists() and not path.is_dir():
            raise PathValidationError(
                f"Path exists but is not a directory: {path}",
                path=str(path),
                severity="medium",
            )

        return path

    def _is_path_allowed(self, path: Path) -> bool:
        """
        Check if path is within allowed base directories.

        Args:
            path: Resolved path to check

        Returns:
            True if path is allowed, False otherwise
        """
        # Check if path starts with any allowed base directory
        for base_dir in self.allowed_base_dirs:
            try:
                # Use relative_to to check if path is under base_dir
                path.relative_to(base_dir)
                return True
            except ValueError:
                # Path is not relative to this base dir
                continue

        return False

    def _check_blocked_paths(self, path: Path) -> None:
        """
        Check if path is in blocked paths.

        Args:
            path: Resolved path to check

        Raises:
            SecurityException: If path is blocked
        """
        path_str = str(path)

        for blocked in self.blocked_paths:
            blocked_str = str(blocked)

            # Check if path starts with blocked path
            if path_str.startswith(blocked_str):
                raise BlockedPathAccessError(
                    f"Cannot access blocked path: {blocked}",
                    path=path_str,
                )

    def create_secure_directory(self, directory_path: str) -> Path:
        """
        Validate and create directory securely.

        Args:
            directory_path: Directory path to create

        Returns:
            Created directory path

        Raises:
            SecurityException: If path is invalid or blocked
        """
        path = self.validate_directory_creation(directory_path)

        try:
            # Create directory with restrictive permissions (0o755)
            path.mkdir(parents=True, exist_ok=True, mode=0o755)

            if self.enable_audit_logging:
                logger.info(
                    "Directory created securely",
                    extra={
                        "path": str(path),
                        "mode": "0o755",
                    },
                )

            return path

        except Exception as e:
            if self.enable_audit_logging:
                logger.error(
                    "Failed to create directory",
                    extra={
                        "path": str(path),
                        "error": str(e),
                    },
                )
            raise

    def get_relative_path(self, path: Path, base_dir: Optional[Path] = None) -> Path:
        """
        Get relative path from base directory.

        Args:
            path: Absolute path
            base_dir: Base directory. If None, uses first allowed base dir.

        Returns:
            Relative path

        Raises:
            SecurityException: If path is not within base directory
        """
        if base_dir is None:
            base_dir = self.allowed_base_dirs[0]

        try:
            return path.relative_to(base_dir)
        except ValueError as e:
            raise PathNotAllowedError(
                f"Path {path} is not within base directory {base_dir}",
                path=str(path),
                allowed_dirs=[str(base_dir)],
            ) from e

    def is_safe_filename(self, filename: str) -> bool:
        """
        Check if filename is safe (no path components, no special characters).

        Args:
            filename: Filename to check

        Returns:
            True if filename is safe
        """
        # Check for path separators (forward slash, backslash)
        if "/" in filename or "\\" in filename:
            return False

        # Check for parent directory references
        if filename in [".", ".."]:
            return False

        # Check for parent directory patterns
        if ".." in filename:
            return False

        # Check for null bytes
        if "\x00" in filename:
            return False

        # Check for leading/trailing dots (hidden files)
        if filename.startswith(".") or filename.endswith("."):
            logger.warning(f"Potentially unsafe filename: {filename}")

        return True
