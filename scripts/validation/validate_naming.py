#!/usr/bin/env python3
"""Naming convention validation for omnibase_infra.

This script validates that Python files and classes follow the ONEX naming conventions
for the omnibase_infra codebase. It leverages omnibase_core's validation infrastructure
while adding omnibase_infra-specific rules.

Infrastructure-Specific Conventions:
    - handlers/ -> handler_*.py, Handler* classes
    - dispatchers/ -> dispatcher_*.py, Dispatcher* classes
    - stores/ -> store_*.py, Store* classes
    - adapters/ -> adapter_*.py, Adapter* classes
    - reducers/ -> reducer_*.py (node-specific)
    - runtime/ -> service_*.py, registry_*.py, handler_*.py

Usage:
    python scripts/validation/validate_naming.py src/omnibase_infra
    python scripts/validation/validate_naming.py src/omnibase_infra --verbose
    python scripts/validation/validate_naming.py src/omnibase_infra --fail-on-warnings

Exit Codes:
    0 - All naming conventions are compliant
    1 - Naming violations detected (errors, or warnings with --fail-on-warnings)

Performance Optimizations:
    - Single directory scan with file caching
    - Pre-compiled regex patterns
    - O(1) set lookups for allowed files and prefixes
    - Batched file reading (each file parsed only once)
    - Cached path component extraction
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

# =============================================================================
# omnibase_core imports (reusing existing validation infrastructure)
# =============================================================================
try:
    from omnibase_core.validation.checker_naming_convention import (
        ALLOWED_FILE_PREFIXES,
        ALLOWED_FILES,
        NamingConventionChecker,
    )

    OMNIBASE_CORE_AVAILABLE = True
except ImportError:
    # Fallback if omnibase_core is not available
    OMNIBASE_CORE_AVAILABLE = False
    ALLOWED_FILES: set[str] = {"__init__.py", "conftest.py", "py.typed"}
    ALLOWED_FILE_PREFIXES: tuple[str, ...] = ("_",)


# =============================================================================
# omnibase_infra-specific directory prefix rules
# =============================================================================
# These rules complement the omnibase_core rules for infrastructure-specific directories.
# NOTE: These are enforced as WARNINGS, not ERRORS, to allow incremental adoption.
#
# PERFORMANCE: Using frozensets for O(1) prefix lookups instead of tuple iteration.
INFRA_DIRECTORY_PREFIX_RULES: dict[str, frozenset[str]] = {
    # Handler directories - primary pattern
    "handlers": frozenset(
        {"handler_", "protocol_", "model_", "enum_", "adapter_", "registry_"}
    ),
    # Dispatcher directories
    "dispatchers": frozenset({"dispatcher_"}),
    # Store directories (idempotency, state storage)
    "stores": frozenset({"store_"}),
    "idempotency": frozenset({"store_", "protocol_"}),
    # Adapter directories
    "adapters": frozenset({"adapter_"}),
    # Runtime directory - accepts many patterns due to diverse component types
    "runtime": frozenset(
        {
            "service_",
            "registry_",
            "handler_",
            "protocol_",
            "chain_",
            "dispatch_",
            "container_",
            "envelope_",
            "introspection_",
            "invocation_",
            "kernel",  # kernel.py is allowed
            "message_",
            "policy_",
            "projector_",
            "runtime_",
            "security_",
            "util_",
            "validation",  # validation.py is allowed
            "wiring",  # wiring.py is allowed
            "contract_",
        }
    ),
    # DLQ directory
    "dlq": frozenset(
        {"service_", "dlq_", "protocol_", "constants_", "model_", "enum_"}
    ),
    # Services directory
    "services": frozenset({"service_"}),
    # Validation directory - accepts suffix patterns (e.g., *_validator.py)
    "validation": frozenset({"validator_", "checker_", "infra_"}),
    # Models directory
    "models": frozenset({"model_", "enum_", "protocol_", "registry_", "types_"}),
    # Enums directory
    "enums": frozenset({"enum_"}),
    # Protocols directory
    "protocols": frozenset({"protocol_"}),
    # Mixins directory
    "mixins": frozenset({"mixin_", "protocol_"}),
    # Registry directory (within nodes)
    "registry": frozenset({"registry_"}),
    # Nodes directory (node.py is the main file, also allow model_, handler_, etc.)
    "nodes": frozenset(
        {
            "node_",
            "node.py",
            "model_",
            "handler_",
            "registry_",
            "enum_",
            "protocol_",
        }
    ),
    # Effects directory
    "effects": frozenset({"store_", "registry_", "effect_"}),
    # Clients directory
    "clients": frozenset({"client_"}),
}

# Additional suffix patterns that are allowed (e.g., *_validator.py)
# PERFORMANCE: Using frozensets for O(1) suffix lookups
INFRA_DIRECTORY_SUFFIX_RULES: dict[str, frozenset[str]] = {
    "validation": frozenset({"_validator.py", "_linter.py", "_aggregator.py"}),
}

# Files that are always allowed regardless of directory
# PERFORMANCE: Using frozenset for immutable O(1) lookups
INFRA_ALLOWED_FILES: frozenset[str] = frozenset(
    {
        "__init__.py",
        "conftest.py",
        "py.typed",
        "node.py",  # Node main files
        "contract.yaml",
        "README.md",
    }
)

# Maximum file size to validate (10 MB) - prevents memory issues with large files
MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB

# PERFORMANCE: Combined set of all allowed files for single O(1) lookup
# Merges both omnibase_core's ALLOWED_FILES and INFRA_ALLOWED_FILES
_ALL_ALLOWED_FILES: frozenset[str] = INFRA_ALLOWED_FILES | frozenset(ALLOWED_FILES)


def _is_in_archived_directory(file_path: Path) -> bool:
    """Check if file is in an archived directory using cross-platform path comparison.

    Args:
        file_path: Path to check.

    Returns:
        True if any parent directory is named "archived".
    """
    return "archived" in file_path.parts


@dataclass
class NamingViolation:
    """Represents a single naming convention violation.

    Attributes:
        file_path: Absolute path to the file containing the violation.
        line_number: Line number where the violation occurs.
        class_name: Name of the class that violates the convention (or '(file name)').
        expected_pattern: The regex pattern or description of expected naming.
        description: Human-readable description of the naming rule.
        severity: Violation severity ('error' or 'warning').
    """

    file_path: str
    line_number: int
    class_name: str
    expected_pattern: str
    description: str
    severity: str = "error"


@dataclass
class ParsedFileInfo:
    """Cached information about a parsed Python file.

    PERFORMANCE: Stores parsed AST and extracted class definitions to avoid
    re-reading and re-parsing the same file for multiple category validations.

    Attributes:
        file_path: Resolved absolute path to the file.
        ast_tree: Parsed AST tree (None if parsing failed).
        class_defs: List of (class_name, line_number) tuples.
        parse_error: Error message if parsing failed, None otherwise.
        relevant_dir: The omnibase_infra subdirectory (e.g., 'handlers', 'models').
    """

    file_path: Path
    ast_tree: ast.Module | None
    class_defs: list[tuple[str, int]] = field(default_factory=list)
    parse_error: str | None = None
    relevant_dir: str | None = None


class InfraNamingConventionValidator:
    """Validates naming conventions for omnibase_infra codebase.

    This validator extends omnibase_core's validation with infrastructure-specific
    rules for handlers, dispatchers, stores, adapters, and other infra components.
    """

    # Class naming patterns for infrastructure components
    NAMING_PATTERNS: ClassVar[dict[str, dict[str, str | None]]] = {
        "handlers": {
            "pattern": r"^Handler[A-Z][A-Za-z0-9]*$",
            "file_prefix": "handler_",
            "description": "Handlers must start with 'Handler' (e.g., HandlerConsul, HandlerDb)",
            "directory": "handlers",
        },
        "dispatchers": {
            "pattern": r"^Dispatcher[A-Z][A-Za-z0-9]*$",
            "file_prefix": "dispatcher_",
            "description": "Dispatchers must start with 'Dispatcher' (e.g., DispatcherNodeIntrospected)",
            "directory": "dispatchers",
        },
        "stores": {
            "pattern": r"^Store[A-Z][A-Za-z0-9]*$",
            "file_prefix": "store_",
            "description": "Stores must start with 'Store' (e.g., StoreInmemory, StorePostgres)",
            "directory": None,  # Stores can be in multiple directories
        },
        "adapters": {
            "pattern": r"^Adapter[A-Z][A-Za-z0-9]*$",
            "file_prefix": "adapter_",
            "description": "Adapters must start with 'Adapter' (e.g., AdapterOnexToMcp)",
            "directory": "adapters",
        },
        "registries": {
            "pattern": r"^Registry[A-Z][A-Za-z0-9]*$",
            "file_prefix": "registry_",
            "description": "Registries must start with 'Registry' (e.g., RegistryCompute, RegistryDispatcher)",
            "directory": None,  # Registries can be in multiple directories
        },
        "services": {
            "pattern": r"^Service[A-Z][A-Za-z0-9]*$",
            "file_prefix": "service_",
            "description": "Services must start with 'Service' (e.g., ServiceHealth, ServiceTimeoutScanner)",
            "directory": "services",
        },
        "models": {
            "pattern": r"^Model[A-Z][A-Za-z0-9]*$",
            "file_prefix": "model_",
            "description": "Models must start with 'Model' (e.g., ModelConsulConfig)",
            "directory": "models",
        },
        "enums": {
            "pattern": r"^Enum[A-Z][A-Za-z0-9]*$",
            "file_prefix": "enum_",
            "description": "Enums must start with 'Enum' (e.g., EnumDispatchStrategy)",
            "directory": "enums",
        },
        "protocols": {
            "pattern": r"^Protocol[A-Z][A-Za-z0-9]*$",
            "file_prefix": "protocol_",
            "description": "Protocols must start with 'Protocol' (e.g., ProtocolHandler)",
            "directory": "protocols",
        },
        "mixins": {
            "pattern": r"^Mixin[A-Z][A-Za-z0-9]*$",
            "file_prefix": "mixin_",
            "description": "Mixins must start with 'Mixin' (e.g., MixinAsyncCircuitBreaker)",
            "directory": "mixins",
        },
    }

    # Exception patterns - classes that don't need to follow strict naming
    # PERFORMANCE: Raw patterns stored here; use _COMPILED_EXCEPTION_PATTERNS for matching
    EXCEPTION_PATTERNS: ClassVar[list[str]] = [
        r"^_.*",  # Private classes
        r".*Test$",  # Test classes
        r".*TestCase$",  # Test case classes
        r"^Test.*",  # Test classes
        r".*Error$",  # Exception classes (end with Error)
        r".*Exception$",  # Exception classes (end with Exception)
        r"^Exception[A-Z].*",  # Exception classes (start with Exception)
        # Common ONEX suffix patterns that may appear across different directories
        r".*Config$",  # Configuration classes (e.g., ModelKafkaEventBusConfig)
        r".*Result$",  # Result model classes (e.g., ModelValidationResult)
        r".*Response$",  # Response model classes (e.g., ModelApiResponse)
    ]

    # PERFORMANCE: Pre-compiled regex patterns (compiled once at class load time)
    # This avoids re-compiling the same patterns on every _is_exception_class() call
    _COMPILED_EXCEPTION_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(r"^_.*"),
        re.compile(r".*Test$"),
        re.compile(r".*TestCase$"),
        re.compile(r"^Test.*"),
        re.compile(r".*Error$"),
        re.compile(r".*Exception$"),
        re.compile(r"^Exception[A-Z].*"),
        # Common ONEX suffix patterns
        re.compile(r".*Config$"),
        re.compile(r".*Result$"),
        re.compile(r".*Response$"),
    ]

    # PERFORMANCE: Pre-compiled naming patterns for fast class name validation
    # Maps category -> compiled regex pattern
    _COMPILED_NAMING_PATTERNS: ClassVar[dict[str, re.Pattern[str]]] = {
        "handlers": re.compile(r"^Handler[A-Z][A-Za-z0-9]*$"),
        "dispatchers": re.compile(r"^Dispatcher[A-Z][A-Za-z0-9]*$"),
        "stores": re.compile(r"^Store[A-Z][A-Za-z0-9]*$"),
        "adapters": re.compile(r"^Adapter[A-Z][A-Za-z0-9]*$"),
        "registries": re.compile(r"^Registry[A-Z][A-Za-z0-9]*$"),
        "services": re.compile(r"^Service[A-Z][A-Za-z0-9]*$"),
        "models": re.compile(r"^Model[A-Z][A-Za-z0-9]*$"),
        "enums": re.compile(r"^Enum[A-Z][A-Za-z0-9]*$"),
        "protocols": re.compile(r"^Protocol[A-Z][A-Za-z0-9]*$"),
        "mixins": re.compile(r"^Mixin[A-Z][A-Za-z0-9]*$"),
    }

    # Architectural exemptions - documented design decisions
    ARCHITECTURAL_EXEMPTIONS: ClassVar[dict[str, list[str]]] = {
        # Runtime directory has multiple valid patterns
        "runtime/": [
            "Handler*",  # Handler classes in runtime (HandlerPluginLoader, etc.)
            "Registry*",  # Registry classes (RegistryCompute, RegistryDispatcher)
            "*Shell",  # Shell classes (ProjectorShell)
            "*Kernel",  # Kernel classes
            "*Engine",  # Engine classes (MessageDispatchEngine)
            "*Enforcer",  # Enforcer classes (DispatchContextEnforcer)
            "*Validator",  # Validator classes
            "*Scheduler",  # Scheduler classes (RuntimeScheduler)
            "*Router",  # Router classes (IntrospectionEventRouter)
            "*Wiring",  # Wiring classes
            "*Process",  # Process classes (RuntimeHostProcess)
            "*Manager",  # Manager classes (ProjectorSchemaManager)
            "*Discovery",  # Discovery classes (ContractHandlerDiscovery)
        ],
        # Handlers directory allows Handler* prefix
        "handlers/": [
            "Handler*",  # All Handler classes
            "Adapter*",  # Adapter classes in handlers/mcp/
            "*Adapter",  # Legacy adapter classes (ONEXToMCPAdapter)
        ],
        # Dispatchers directory allows Dispatcher* prefix
        "dispatchers/": [
            "Dispatcher*",  # All Dispatcher classes
        ],
        # Models directory has multiple valid patterns
        "models/": [
            "Model*",  # Model classes
            "Enum*",  # Enum classes in models
            "Registry*",  # RegistryPayload* classes
            "*Intent",  # Intent classes (RegistryIntent)
        ],
        # Registry directories within nodes
        "registry/": [
            "Registry*",  # All Registry classes
        ],
        # Node handler directories
        "nodes/": [
            "Handler*",  # Handler classes within nodes
            "Node*",  # Node classes
            "Registry*",  # Registry classes
        ],
        # Services directory
        "services/": [
            "Service*",  # Service classes
        ],
        # Idempotency stores
        "idempotency/": [
            "Store*",  # Store classes
        ],
        # Effects directory
        "effects/": [
            "Store*",  # Store effect classes
            "Registry*",  # Registry effect classes
            "InMemory*",  # Legacy InMemory store classes
        ],
        # DLQ directory
        "dlq/": [
            "Service*",  # DLQ service classes
        ],
    }

    def __init__(self, repo_path: Path) -> None:
        """Initialize the naming convention validator.

        Args:
            repo_path: Path to the repository root directory to validate.

        PERFORMANCE: Initializes file cache to avoid re-reading and re-parsing
        files across multiple validation phases.
        """
        self.repo_path = repo_path
        self.violations: list[NamingViolation] = []
        # PERFORMANCE: Cache parsed file info to avoid re-reading files
        # Key: resolved absolute path, Value: ParsedFileInfo
        self._file_cache: dict[Path, ParsedFileInfo] = {}
        # PERFORMANCE: Cache all Python files found during directory scan
        # Avoids multiple rglob calls for different categories
        self._all_python_files: list[Path] | None = None
        # Track files skipped due to size limit (path, size_in_bytes)
        self._skipped_large_files: list[tuple[Path, int]] = []
        # Track which (file, category) pairs have been validated to prevent duplicates
        self._validated_file_categories: set[tuple[Path, str]] = set()

    def check_file_name(self, file_path: Path) -> tuple[str | None, str]:
        """Check if a file name conforms to omnibase_infra naming conventions.

        Args:
            file_path: Path to the file to check.

        Returns:
            Tuple of (error message or None, severity).
            Severity is 'error' for strict violations, 'warning' for style issues.

        PERFORMANCE:
            - Uses combined _ALL_ALLOWED_FILES for single O(1) lookup
            - Pre-cached path.parts avoids repeated property access
            - Early exits for common skip cases (non-.py, allowed files)
        """
        file_name = file_path.name

        # PERFORMANCE: Skip non-Python files early (most common skip case)
        if not file_name.endswith(".py"):
            return None, "info"

        # PERFORMANCE: Single O(1) lookup using combined allowed files set
        if file_name in _ALL_ALLOWED_FILES:
            return None, "info"

        # Skip files with allowed prefixes (private modules)
        # PERFORMANCE: ALLOWED_FILE_PREFIXES is typically small (1-2 items)
        if any(file_name.startswith(prefix) for prefix in ALLOWED_FILE_PREFIXES):
            return None, "info"

        # Find the relevant directory for rule matching
        # PERFORMANCE: Cache parts tuple to avoid repeated property access
        parts = file_path.parts
        try:
            # Find omnibase_infra in the path
            omnibase_idx = parts.index("omnibase_infra")
            if omnibase_idx + 1 < len(parts) - 1:
                relevant_dir = parts[omnibase_idx + 1]

                # Check suffix rules first (e.g., *_validator.py in validation/)
                # PERFORMANCE: O(1) dict lookup + small frozenset iteration
                suffix_rules = INFRA_DIRECTORY_SUFFIX_RULES.get(relevant_dir)
                if suffix_rules:
                    for suffix in suffix_rules:
                        if file_name.endswith(suffix):
                            return None, "info"  # Valid suffix pattern

                # PERFORMANCE: O(1) dict lookup for prefix rules
                required_prefixes = INFRA_DIRECTORY_PREFIX_RULES.get(relevant_dir)
                if required_prefixes:
                    # Check if file name starts with or equals any required prefix
                    # PERFORMANCE: Iterate over frozenset (typically small, 1-20 items)
                    matches_prefix = False
                    for prefix in required_prefixes:
                        if file_name.startswith(prefix) or file_name == prefix:
                            matches_prefix = True
                            break  # Early exit on first match

                    if not matches_prefix:
                        # PERFORMANCE: Only build prefix_str if violation found
                        prefix_list = sorted(required_prefixes)[:3]
                        if len(required_prefixes) == 1:
                            prefix_str = f"'{prefix_list[0]}'"
                        elif len(required_prefixes) > 3:
                            prefix_str = f"one of {prefix_list}..."
                        else:
                            prefix_str = f"one of {sorted(required_prefixes)}"
                        # File naming is a WARNING for gradual adoption
                        return (
                            f"File '{file_name}' in '{relevant_dir}/' directory should start "
                            f"with {prefix_str}",
                            "warning",
                        )
        except ValueError:
            # omnibase_infra not in path, skip validation
            pass

        return None, "info"

    def _get_all_python_files(self) -> list[Path]:
        """Get all Python files in the repository, cached for reuse.

        PERFORMANCE: Single directory scan that caches results. Subsequent calls
        return the cached list instead of re-scanning the filesystem.

        Returns:
            List of resolved absolute paths to all Python files.
        """
        if self._all_python_files is None:
            self._all_python_files = []
            for file_path in self.repo_path.rglob("*.py"):
                # Skip pycache and archived directories (cross-platform path check)
                if "__pycache__" in file_path.parts or _is_in_archived_directory(
                    file_path
                ):
                    continue
                # Skip symbolic links
                if file_path.is_symlink():
                    continue
                # Skip files exceeding size limit to prevent memory issues
                try:
                    file_size = file_path.stat().st_size
                    if file_size > MAX_FILE_SIZE_BYTES:
                        self._skipped_large_files.append(
                            (file_path.resolve(), file_size)
                        )
                        continue
                except OSError:
                    # If we can't stat the file, skip it
                    continue
                # Store resolved paths for proper deduplication
                self._all_python_files.append(file_path.resolve())
        return self._all_python_files

    def _get_parsed_file_info(self, file_path: Path) -> ParsedFileInfo:
        """Get cached parsed file info, parsing on first access.

        PERFORMANCE: Each file is read and parsed only once. Subsequent calls
        for the same file return the cached ParsedFileInfo.

        Args:
            file_path: Resolved absolute path to the Python file.

        Returns:
            ParsedFileInfo with AST tree and extracted class definitions.
        """
        if file_path not in self._file_cache:
            # Parse the file and extract class definitions
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content, filename=str(file_path))
                class_defs: list[tuple[str, int]] = []

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_defs.append((node.name, node.lineno))

                # Extract relevant directory for rule matching
                relevant_dir: str | None = None
                parts = file_path.parts
                try:
                    omnibase_idx = parts.index("omnibase_infra")
                    if omnibase_idx + 1 < len(parts) - 1:
                        relevant_dir = parts[omnibase_idx + 1]
                except ValueError:
                    pass

                self._file_cache[file_path] = ParsedFileInfo(
                    file_path=file_path,
                    ast_tree=tree,
                    class_defs=class_defs,
                    relevant_dir=relevant_dir,
                )

            except (SyntaxError, UnicodeDecodeError) as e:
                self._file_cache[file_path] = ParsedFileInfo(
                    file_path=file_path,
                    ast_tree=None,
                    parse_error=str(e),
                )

        return self._file_cache[file_path]

    def validate_directory(
        self, directory: Path, verbose: bool = False
    ) -> list[tuple[str, str, str]]:
        """Validate all Python files in a directory against naming conventions.

        Args:
            directory: Path to the directory to validate.
            verbose: If True, log each file as it's checked.

        Returns:
            List of tuples (file_path, message, severity).

        PERFORMANCE: Uses cached file list from _get_all_python_files().
        """
        results: list[tuple[str, str, str]] = []

        # PERFORMANCE: Use cached file list instead of re-scanning
        for file_path in self._get_all_python_files():
            # Filter to files within the specified directory
            try:
                file_path.relative_to(directory)
            except ValueError:
                continue  # File not under this directory

            message, severity = self.check_file_name(file_path)
            if message:
                results.append((str(file_path), message, severity))
            elif verbose:
                print(f"Checked: {file_path}")

        return results

    def validate_naming_conventions(self, verbose: bool = False) -> bool:
        """Validate all naming conventions across the repository.

        Args:
            verbose: If True, show detailed output.

        Returns:
            True if no errors were found, False otherwise.
        """
        # Phase 1: File naming validation
        file_results = self.validate_directory(self.repo_path, verbose)
        for file_path, message, severity in file_results:
            self.violations.append(
                NamingViolation(
                    file_path=file_path,
                    line_number=1,
                    class_name="(file name)",
                    expected_pattern="See directory prefix rules",
                    description=message,
                    severity=severity,
                )
            )

        # Phase 2: Class naming validation
        for category, rules in self.NAMING_PATTERNS.items():
            self._validate_category(category, rules, verbose)

        return len([v for v in self.violations if v.severity == "error"]) == 0

    def _validate_category(
        self, category: str, rules: dict[str, str | None], verbose: bool = False
    ) -> None:
        """Validate naming conventions for a specific category.

        Args:
            category: The category to validate (e.g., 'handlers', 'dispatchers').
            rules: Dictionary containing validation rules.
            verbose: If True, show detailed output.

        PERFORMANCE:
            - Uses cached file list from _get_all_python_files() instead of
              separate rglob calls for each category
            - Files are filtered in-memory using cached path info
            - Each file is validated only once per category
        """
        file_prefix = rules.get("file_prefix")
        expected_dir = rules.get("directory")

        # PERFORMANCE: Filter from cached file list instead of multiple rglob calls
        # This is O(n) where n = total files, vs O(n * m) for m glob patterns
        files_to_validate: set[Path] = set()

        for file_path in self._get_all_python_files():
            # Skip __init__.py for directory-based checks
            if file_path.name == "__init__.py":
                continue

            # Check if file matches the prefix pattern
            matches_prefix = file_prefix and file_path.name.startswith(file_prefix)

            # Check if file is in expected directory (using cached info if available)
            matches_dir = False
            if expected_dir:
                # PERFORMANCE: Use path.parts for O(n) directory check vs string search
                matches_dir = expected_dir in file_path.parts

            if matches_prefix or matches_dir:
                files_to_validate.add(file_path)

        # Validate each unique file once
        # PERFORMANCE: Pre-filter against _validated_file_categories to avoid
        # unnecessary method calls for files already validated in this category
        for file_path in files_to_validate:
            validation_key = (file_path, category)
            if validation_key not in self._validated_file_categories:
                self._validate_class_names_in_file(file_path, category, rules, verbose)

    def _validate_class_names_in_file(
        self,
        file_path: Path,
        category: str,
        rules: dict[str, str | None],
        verbose: bool = False,
    ) -> None:
        """Validate class names in a specific file.

        Args:
            file_path: Path to the Python file to validate.
            category: The category being validated.
            rules: Dictionary containing validation rules.
            verbose: If True, show detailed output.

        PERFORMANCE:
            - Uses cached parsed file info from _get_parsed_file_info()
            - Each file is read and parsed only once across all categories
            - Class definitions are pre-extracted during parsing
            - Tracks (file, category) pairs to prevent duplicate validations
        """
        # Prevent duplicate validation of the same file for the same category
        validation_key = (file_path, category)
        if validation_key in self._validated_file_categories:
            return
        self._validated_file_categories.add(validation_key)

        # PERFORMANCE: Use cached parsed info instead of re-reading file
        file_info = self._get_parsed_file_info(file_path)

        if file_info.parse_error:
            if verbose:
                print(f"Warning: Could not parse {file_path}: {file_info.parse_error}")
            return

        # PERFORMANCE: Use pre-extracted class definitions
        for class_name, line_number in file_info.class_defs:
            self._check_class_naming_cached(
                file_path, class_name, line_number, category, rules
            )

    def _check_class_naming_cached(
        self,
        file_path: Path,
        class_name: str,
        line_number: int,
        category: str,
        rules: dict[str, str | None],
    ) -> None:
        """Check if a class name follows conventions (optimized version).

        PERFORMANCE: Works with pre-extracted class info instead of AST nodes.
        Uses pre-compiled regex patterns for fast matching.

        Args:
            file_path: Path to the file containing the class.
            class_name: Name of the class to check.
            line_number: Line number of the class definition.
            category: The category being validated.
            rules: Dictionary containing validation rules.
        """
        description = rules.get("description")

        # PERFORMANCE: Use pre-compiled pattern from class variable
        compiled_pattern = self._COMPILED_NAMING_PATTERNS.get(category)
        if not compiled_pattern:
            return

        # Skip exception patterns (uses pre-compiled patterns)
        if self._is_exception_class(class_name):
            return

        # Skip architectural exemptions
        if self._matches_architectural_exemption(class_name, file_path):
            return

        # Skip classes that already follow a valid naming pattern
        # This prevents flagging Model* classes in handler files, etc.
        if self._matches_any_valid_pattern(class_name):
            return

        # Check if class name matches expected pattern
        file_prefix = rules.get("file_prefix", "")
        expected_dir = rules.get("directory")

        # Only validate classes in files that should contain this category
        # PERFORMANCE: Use path.parts for directory check
        in_relevant_file = file_prefix and file_path.name.startswith(file_prefix)
        in_relevant_dir = expected_dir and expected_dir in file_path.parts

        if not in_relevant_file and not in_relevant_dir:
            return

        # PERFORMANCE: Use pre-compiled pattern
        if not compiled_pattern.match(class_name):
            # Check if it should match based on heuristics
            if self._should_match_pattern(class_name, category):
                self.violations.append(
                    NamingViolation(
                        file_path=str(file_path),
                        line_number=line_number,
                        class_name=class_name,
                        expected_pattern=compiled_pattern.pattern,
                        description=description
                        or f"Must follow {category} naming conventions",
                        severity="error",
                    )
                )

    def _check_class_naming(
        self,
        file_path: Path,
        node: ast.ClassDef,
        category: str,
        rules: dict[str, str | None],
    ) -> None:
        """Check if a class name follows conventions (legacy version).

        Note: Prefer _check_class_naming_cached for better performance.

        Args:
            file_path: Path to the file containing the class.
            node: AST class definition node.
            category: The category being validated.
            rules: Dictionary containing validation rules.
        """
        # Delegate to cached version
        self._check_class_naming_cached(
            file_path, node.name, node.lineno, category, rules
        )

    def _is_exception_class(self, class_name: str) -> bool:
        """Check if class name matches exception patterns.

        PERFORMANCE: Uses pre-compiled regex patterns from _COMPILED_EXCEPTION_PATTERNS
        instead of re-compiling patterns on each call.

        Args:
            class_name: Name of the class to check.

        Returns:
            True if class matches an exception pattern.
        """
        # PERFORMANCE: Use pre-compiled patterns for O(1) regex execution per pattern
        return any(
            pattern.match(class_name) for pattern in self._COMPILED_EXCEPTION_PATTERNS
        )

    def _matches_architectural_exemption(
        self, class_name: str, file_path: Path
    ) -> bool:
        """Check if a class matches documented architectural exemptions.

        Args:
            class_name: Name of the class to check.
            file_path: Path to the file containing the class.

        Returns:
            True if class is architecturally exempt from standard naming rules.
        """
        for directory, exempted_patterns in self.ARCHITECTURAL_EXEMPTIONS.items():
            # Check if file is in the exempted directory
            # Use path.parts for reliable directory matching (not string containment)
            dir_name = directory.rstrip("/")
            if dir_name not in file_path.parts:
                continue

            # Check if class matches any exempted pattern
            for pattern in exempted_patterns:
                if pattern.endswith("*"):
                    # Prefix wildcard pattern
                    prefix = pattern[:-1]
                    if class_name.startswith(prefix):
                        return True
                elif pattern.startswith("*"):
                    # Suffix wildcard pattern
                    suffix = pattern[1:]
                    if class_name.endswith(suffix):
                        return True
                elif class_name == pattern:
                    # Exact match
                    return True

        return False

    # PERFORMANCE: Class variable for category indicators (created once, reused)
    _CATEGORY_INDICATORS: ClassVar[dict[str, tuple[str, ...]]] = {
        "handlers": ("handler",),
        "dispatchers": ("dispatcher", "dispatch"),
        "stores": ("store", "storage"),
        "adapters": ("adapter",),
        "registries": ("registry",),
        "services": ("service",),
        "models": ("model", "data", "schema", "entity"),
        "enums": ("enum", "choice", "status"),
        "protocols": ("protocol", "interface"),
        "mixins": ("mixin",),
    }

    def _should_match_pattern(self, class_name: str, category: str) -> bool:
        """Determine if a class should match the pattern for a category.

        Uses heuristics based on keywords in the class name.

        PERFORMANCE: Uses class-level _CATEGORY_INDICATORS instead of
        recreating the dict on every call.

        Args:
            class_name: Name of the class to check.
            category: The category to check against.

        Returns:
            True if the class name suggests it should follow the category's pattern.
        """
        indicators = self._CATEGORY_INDICATORS.get(category, ())
        class_lower = class_name.lower()

        return any(indicator in class_lower for indicator in indicators)

    def _matches_any_valid_pattern(self, class_name: str) -> bool:
        """Check if a class name matches any of the defined valid naming patterns.

        This prevents false positives when a class follows a different valid pattern
        than the category being checked (e.g., Model* classes in handler files).

        PERFORMANCE: Uses pre-compiled patterns from _COMPILED_NAMING_PATTERNS
        instead of re-compiling on each call.

        Args:
            class_name: Name of the class to check.

        Returns:
            True if the class matches any valid naming pattern.
        """
        # PERFORMANCE: Use pre-compiled patterns for faster matching
        for compiled_pattern in self._COMPILED_NAMING_PATTERNS.values():
            if compiled_pattern.match(class_name):
                return True
        return False

    def generate_report(self) -> str:
        """Generate naming convention validation report.

        Returns:
            Formatted string report with violation details.
        """
        if not self.violations and not self._skipped_large_files:
            return "All naming conventions are compliant!"

        errors = [v for v in self.violations if v.severity == "error"]
        warnings = [v for v in self.violations if v.severity == "warning"]

        report = "Naming Convention Validation Report\n"
        report += "=" * 50 + "\n\n"

        report += f"Summary: {len(errors)} errors, {len(warnings)} warnings"
        if self._skipped_large_files:
            report += f", {len(self._skipped_large_files)} files skipped (too large)"
        report += "\n\n"

        # Report skipped large files
        if self._skipped_large_files:
            report += "SKIPPED FILES (exceeding size limit):\n"
            report += "-" * 40 + "\n"
            for file_path, size in self._skipped_large_files:
                size_mb = size / (1024 * 1024)
                report += f"  {file_path} ({size_mb:.2f} MB)\n"
            report += "\n"

        if errors:
            report += "NAMING ERRORS (Must Fix):\n"
            report += "-" * 40 + "\n"
            for violation in errors:
                report += f"  {violation.class_name} (Line {violation.line_number})\n"
                report += f"    File: {violation.file_path}\n"
                report += f"    Expected: {violation.expected_pattern}\n"
                report += f"    Rule: {violation.description}\n\n"

        if warnings:
            report += "NAMING WARNINGS (Should Fix):\n"
            report += "-" * 40 + "\n"
            for violation in warnings:
                report += f"  {violation.class_name} (Line {violation.line_number})\n"
                report += f"    File: {violation.file_path}\n"
                report += f"    Issue: {violation.description}\n\n"

        # Add quick reference
        report += "NAMING CONVENTION REFERENCE:\n"
        report += "-" * 40 + "\n"
        for category, rules in self.NAMING_PATTERNS.items():
            description = (
                rules.get("description") or f"{category.title()} naming convention"
            )
            file_prefix = rules.get("file_prefix")
            pattern = rules.get("pattern") or "N/A"
            report += f"  {category.title()}:\n"
            report += f"    {description}\n"
            if file_prefix:
                report += f"    File: {file_prefix}*.py\n"
            report += f"    Class: {pattern}\n\n"

        return report


def run_ast_validation(repo_path: Path, verbose: bool = False) -> list[str]:
    """Run AST-based validation using omnibase_core's NamingConventionChecker.

    This validates function naming (snake_case) and detects anti-patterns.

    Args:
        repo_path: Path to validate.
        verbose: If True, show detailed output.

    Returns:
        List of AST-based violations.
    """
    if not OMNIBASE_CORE_AVAILABLE:
        if verbose:
            print("Note: omnibase_core not available, skipping AST validation")
        return []

    ast_issues: list[str] = []

    for file_path in repo_path.rglob("*.py"):
        # Cross-platform path segment check
        if "__pycache__" in file_path.parts or _is_in_archived_directory(file_path):
            continue
        if file_path.is_symlink():
            continue

        # Skip files exceeding size limit to prevent memory issues during AST parsing
        try:
            file_size = file_path.stat().st_size
            if file_size > MAX_FILE_SIZE_BYTES:
                if verbose:
                    size_mb = file_size / (1024 * 1024)
                    print(f"Skipping large file: {file_path} ({size_mb:.2f} MB)")
                continue
        except OSError:
            # If we can't stat the file, skip it
            continue

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))
            checker = NamingConventionChecker(str(file_path))
            checker.visit(tree)

            for issue in checker.issues:
                ast_issues.append(f"{file_path}: {issue}")

        except (SyntaxError, UnicodeDecodeError):
            continue

    return ast_issues


def main() -> int:
    """Main entry point for the naming convention validator.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Validate omnibase_infra naming conventions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validation/validate_naming.py src/omnibase_infra
  python scripts/validation/validate_naming.py src/omnibase_infra --verbose
  python scripts/validation/validate_naming.py src/omnibase_infra --errors-only
  python scripts/validation/validate_naming.py src/omnibase_infra --fail-on-warnings
  python scripts/validation/validate_naming.py src/omnibase_infra --include-ast

Class Naming Conventions:
  Handlers:    Handler*     (e.g., HandlerConsul, HandlerDb)
  Dispatchers: Dispatcher*  (e.g., DispatcherNodeIntrospected)
  Stores:      Store*       (e.g., StorePostgres, StoreInmemory)
  Adapters:    Adapter*     (e.g., AdapterOnexToMcp)
  Registries:  Registry*    (e.g., RegistryCompute, RegistryDispatcher)
  Services:    Service*     (e.g., ServiceHealth)
  Models:      Model*       (e.g., ModelConsulConfig)
  Enums:       Enum*        (e.g., EnumDispatchStrategy)
  Protocols:   Protocol*    (e.g., ProtocolHandler)
  Mixins:      Mixin*       (e.g., MixinAsyncCircuitBreaker)
""",
    )
    parser.add_argument("repo_path", help="Path to validate (e.g., src/omnibase_infra)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--errors-only",
        "-e",
        action="store_true",
        help="Only show errors, hide warnings",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Exit with error code if warnings are found",
    )
    parser.add_argument(
        "--include-ast",
        action="store_true",
        help="Include AST validation (function naming, anti-patterns) from omnibase_core",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format",
    )

    args = parser.parse_args()

    repo_path = Path(args.repo_path).resolve()
    if not repo_path.exists():
        print(f"Error: Path does not exist: {repo_path}")
        return 1

    if not args.json:
        print(f"Validating naming conventions in: {repo_path}")
        print("=" * 60)

    # Run infrastructure naming validation
    validator = InfraNamingConventionValidator(repo_path)
    validator.validate_naming_conventions(args.verbose)

    # Optionally run AST validation from omnibase_core
    ast_issues: list[str] = []
    if args.include_ast:
        if not args.json:
            print("\nRunning AST validation (function naming, anti-patterns)...")
        ast_issues = run_ast_validation(repo_path, args.verbose)
        if ast_issues and not args.json:
            print(f"Found {len(ast_issues)} AST-based issues:")
            for issue in ast_issues[:20]:
                print(f"  - {issue}")
            if len(ast_issues) > 20:
                print(f"  ... and {len(ast_issues) - 20} more")

    # Calculate results
    errors = len([v for v in validator.violations if v.severity == "error"])
    warnings = len([v for v in validator.violations if v.severity == "warning"])
    ast_count = len(ast_issues)

    # Determine pass/fail status (used for both JSON output and exit code)
    has_failures = (
        errors > 0 or ast_count > 0 or (args.fail_on_warnings and warnings > 0)
    )

    if args.json:
        import json

        # Get count of skipped large files
        skipped_files_count = len(validator._skipped_large_files)

        output = {
            "path": str(repo_path),
            "errors": errors,
            "warnings": warnings,
            "ast_issues": ast_count,
            "skipped_files": skipped_files_count,
            "max_file_size_bytes": MAX_FILE_SIZE_BYTES,
            "violations": [
                {
                    "file_path": v.file_path,
                    "line_number": v.line_number,
                    "class_name": v.class_name,
                    "expected_pattern": v.expected_pattern,
                    "description": v.description,
                    "severity": v.severity,
                }
                for v in validator.violations
                if not args.errors_only or v.severity == "error"
            ],
            # IMPORTANT: 'passed' mirrors the script exit code logic exactly.
            # The exit code is determined by has_failures (lines 1133-1135):
            #   has_failures = errors > 0 or ast_count > 0 or
            #                  (args.fail_on_warnings and warnings > 0)
            # Therefore:
            # - passed=True  (exit 0): No errors, no AST issues, and either
            #   no warnings or --fail-on-warnings not set
            # - passed=False (exit 1): Any errors, AST issues, or warnings
            #   with --fail-on-warnings set
            # Note: Skipped large files do NOT affect pass/fail status - they
            # are informational only (the file is simply not validated).
            "passed": not has_failures,
        }
        print(json.dumps(output, indent=2))
    else:
        # Generate and print report
        if args.errors_only:
            # Filter to errors only
            filtered_violations = [
                v for v in validator.violations if v.severity == "error"
            ]
            validator.violations = filtered_violations
        print("\n" + validator.generate_report())

    if not args.json:
        if not has_failures:
            print("SUCCESS: All naming conventions are compliant!")
        elif args.fail_on_warnings and warnings > 0:
            print(
                f"FAILURE: {errors} error(s), {warnings} warning(s), {ast_count} AST issues"
            )
            print("  (--fail-on-warnings flag is set)")
        else:
            print(f"FAILURE: {errors + ast_count} naming violations must be fixed!")

    return 0 if not has_failures else 1


if __name__ == "__main__":
    sys.exit(main())
