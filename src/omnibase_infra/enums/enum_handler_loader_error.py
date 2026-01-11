# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Loader Error Code Enumeration.

Defines structured error codes for the HandlerPluginLoader system. These codes
enable precise error classification, debugging, and programmatic error handling
for handler contract discovery and loading operations.

Error Code Ranges:
    - 001-009: File-level errors (load_from_contract)
    - 010-019: Import errors (handler class loading)
    - 020-029: Directory-level errors (load_from_directory)
    - 030-039: Pattern errors (discover_and_load)
    - 040-049: Configuration errors (ambiguous configurations)

Usage:
    >>> from omnibase_infra.enums import EnumHandlerLoaderError
    >>> error_code = EnumHandlerLoaderError.FILE_NOT_FOUND
    >>> print(f"Error: {error_code.value}")
    Error: HANDLER_LOADER_001

    >>> # Check error code in exception handling
    >>> try:
    ...     loader.load_from_contract(path)
    ... except ProtocolConfigurationError as e:
    ...     if e.model.context.get("loader_error") == EnumHandlerLoaderError.NOT_A_FILE.value:
    ...         print("Path exists but is not a file")

See Also:
    - HandlerPluginLoader: Implementation using these error codes
    - ProtocolHandlerPluginLoader: Protocol defining error code contracts
"""

from enum import Enum


class EnumHandlerLoaderError(str, Enum):
    """Error codes for handler plugin loader operations.

    These codes provide structured classification for failures during
    handler contract discovery, parsing, validation, and class loading.

    File-Level Errors (001-009):
        FILE_NOT_FOUND: Contract file path does not exist.
            The specified contract file path does not exist on the filesystem.
            Verify the path is correct and the file has not been moved/deleted.

        INVALID_YAML_SYNTAX: Contract file contains invalid YAML.
            The contract file could not be parsed due to YAML syntax errors.
            Check for proper indentation, quoting, and YAML structure.

        SCHEMA_VALIDATION_FAILED: Contract fails schema validation.
            The parsed YAML does not conform to the handler contract schema.
            This includes empty files or files with only comments.

        MISSING_REQUIRED_FIELDS: Contract is missing required fields.
            Required fields (handler_name, handler_class, handler_type) are missing.
            Add the missing fields to the contract file.

        FILE_SIZE_EXCEEDED: Contract file exceeds size limit.
            The contract file exceeds the maximum allowed size (10MB).
            Contract files should be small configuration files.

        PROTOCOL_NOT_IMPLEMENTED: Handler does not implement protocol.
            The handler class does not implement the required ProtocolHandler
            interface (missing required methods like describe, initialize, etc.).

        NOT_A_FILE: Path exists but is not a regular file.
            The specified path exists but is a directory, symlink, or other
            non-regular file type. Only regular files can be loaded as contracts.

        FILE_READ_ERROR: Failed to read contract file.
            An I/O error occurred while reading the contract file contents.
            Check file permissions and filesystem accessibility.

        FILE_STAT_ERROR: Failed to stat contract file.
            An I/O error occurred while checking file metadata (size, type).
            Check file permissions and filesystem accessibility.

    Import Errors (010-019):
        MODULE_NOT_FOUND: Handler module not found.
            The module specified in handler_class cannot be found.
            Verify the module path and ensure it's installed/accessible.

        CLASS_NOT_FOUND: Handler class not found in module.
            The module exists but does not contain the specified class.
            Verify the class name matches exactly (case-sensitive).

        IMPORT_ERROR: Import error loading handler module.
            The module exists but failed to import due to syntax errors,
            missing dependencies, or circular imports.

        NAMESPACE_NOT_ALLOWED: Handler module namespace not in allowed list.
            The handler module's namespace (package prefix) is not in the
            configured allowed_namespaces list. This is a security control to
            prevent loading handlers from untrusted packages. Add the module's
            namespace prefix to allowed_namespaces or remove the restriction.

    Directory Errors (020-029):
        DIRECTORY_NOT_FOUND: Directory does not exist.
            The specified directory path does not exist on the filesystem.

        PERMISSION_DENIED: Permission denied accessing directory.
            Insufficient permissions to read the directory contents.

        NOT_A_DIRECTORY: Path exists but is not a directory.
            The specified path exists but is a file or other non-directory.

    Pattern Errors (030-039):
        EMPTY_PATTERNS_LIST: Patterns list cannot be empty.
            The discover_and_load method requires at least one glob pattern.

        INVALID_GLOB_PATTERN: Invalid glob pattern syntax.
            The provided glob pattern has invalid syntax.

    Configuration Errors (040-049):
        AMBIGUOUS_CONTRACT_CONFIGURATION: Both handler_contract.yaml and contract.yaml
            exist in the same directory. This is an ambiguous configuration that could
            lead to duplicate handler registrations or unexpected behavior. Use only
            ONE contract file per handler directory to avoid this error.
    """

    # File-level errors (001-009)
    FILE_NOT_FOUND = "HANDLER_LOADER_001"
    INVALID_YAML_SYNTAX = "HANDLER_LOADER_002"
    SCHEMA_VALIDATION_FAILED = "HANDLER_LOADER_003"
    MISSING_REQUIRED_FIELDS = "HANDLER_LOADER_004"
    FILE_SIZE_EXCEEDED = "HANDLER_LOADER_005"
    PROTOCOL_NOT_IMPLEMENTED = "HANDLER_LOADER_006"
    NOT_A_FILE = "HANDLER_LOADER_007"
    FILE_READ_ERROR = "HANDLER_LOADER_008"
    FILE_STAT_ERROR = "HANDLER_LOADER_009"

    # Import errors (010-019)
    MODULE_NOT_FOUND = "HANDLER_LOADER_010"
    CLASS_NOT_FOUND = "HANDLER_LOADER_011"
    IMPORT_ERROR = "HANDLER_LOADER_012"
    NAMESPACE_NOT_ALLOWED = "HANDLER_LOADER_013"

    # Directory errors (020-029)
    DIRECTORY_NOT_FOUND = "HANDLER_LOADER_020"
    PERMISSION_DENIED = "HANDLER_LOADER_021"
    NOT_A_DIRECTORY = "HANDLER_LOADER_022"

    # Pattern errors (030-039)
    EMPTY_PATTERNS_LIST = "HANDLER_LOADER_030"
    INVALID_GLOB_PATTERN = "HANDLER_LOADER_031"

    # Configuration errors (040-049)
    AMBIGUOUS_CONTRACT_CONFIGURATION = "HANDLER_LOADER_040"

    @property
    def is_file_error(self) -> bool:
        """Check if this is a file-level error (001-009)."""
        return self.value.startswith("HANDLER_LOADER_00")

    @property
    def is_import_error(self) -> bool:
        """Check if this is an import error (010-019)."""
        return self.value.startswith("HANDLER_LOADER_01")

    @property
    def is_directory_error(self) -> bool:
        """Check if this is a directory error (020-029)."""
        return self.value.startswith("HANDLER_LOADER_02")

    @property
    def is_pattern_error(self) -> bool:
        """Check if this is a pattern error (030-039)."""
        return self.value.startswith("HANDLER_LOADER_03")

    @property
    def is_configuration_error(self) -> bool:
        """Check if this is a configuration error (040-049)."""
        return self.value.startswith("HANDLER_LOADER_04")


__all__ = ["EnumHandlerLoaderError"]
