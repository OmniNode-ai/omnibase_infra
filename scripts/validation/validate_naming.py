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
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
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
INFRA_DIRECTORY_PREFIX_RULES: dict[str, tuple[str, ...]] = {
    # Handler directories - primary pattern
    "handlers": ("handler_", "protocol_", "model_", "enum_", "adapter_", "registry_"),
    # Dispatcher directories
    "dispatchers": ("dispatcher_",),
    # Store directories (idempotency, state storage)
    "stores": ("store_",),
    "idempotency": ("store_", "protocol_"),
    # Adapter directories
    "adapters": ("adapter_",),
    # Runtime directory - accepts many patterns due to diverse component types
    "runtime": (
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
    ),
    # DLQ directory
    "dlq": ("service_", "dlq_", "protocol_", "constants_", "model_", "enum_"),
    # Services directory
    "services": ("service_",),
    # Validation directory - accepts suffix patterns (e.g., *_validator.py)
    "validation": ("validator_", "checker_", "infra_"),
    # Models directory
    "models": ("model_", "enum_", "protocol_", "registry_", "types_"),
    # Enums directory
    "enums": ("enum_",),
    # Protocols directory
    "protocols": ("protocol_",),
    # Mixins directory
    "mixins": ("mixin_", "protocol_"),
    # Registry directory (within nodes)
    "registry": ("registry_",),
    # Nodes directory (node.py is the main file, also allow model_, handler_, etc.)
    "nodes": (
        "node_",
        "node.py",
        "model_",
        "handler_",
        "registry_",
        "enum_",
        "protocol_",
    ),
    # Effects directory
    "effects": ("store_", "registry_", "effect_"),
    # Clients directory
    "clients": ("client_",),
}

# Additional suffix patterns that are allowed (e.g., *_validator.py)
INFRA_DIRECTORY_SUFFIX_RULES: dict[str, tuple[str, ...]] = {
    "validation": ("_validator.py", "_linter.py", "_aggregator.py"),
}

# Files that are always allowed regardless of directory
INFRA_ALLOWED_FILES: set[str] = {
    "__init__.py",
    "conftest.py",
    "py.typed",
    "node.py",  # Node main files
    "contract.yaml",
    "README.md",
}


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
    EXCEPTION_PATTERNS: ClassVar[list[str]] = [
        r"^_.*",  # Private classes
        r".*Test$",  # Test classes
        r".*TestCase$",  # Test case classes
        r"^Test.*",  # Test classes
        r".*Error$",  # Exception classes (end with Error)
        r".*Exception$",  # Exception classes (end with Exception)
        r"^Exception[A-Z].*",  # Exception classes (start with Exception)
    ]

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
        """
        self.repo_path = repo_path
        self.violations: list[NamingViolation] = []

    def check_file_name(self, file_path: Path) -> tuple[str | None, str]:
        """Check if a file name conforms to omnibase_infra naming conventions.

        Args:
            file_path: Path to the file to check.

        Returns:
            Tuple of (error message or None, severity).
            Severity is 'error' for strict violations, 'warning' for style issues.
        """
        file_name = file_path.name

        # Skip allowed files (both core and infra)
        if file_name in ALLOWED_FILES or file_name in INFRA_ALLOWED_FILES:
            return None, "info"

        # Skip files with allowed prefixes (private modules)
        if any(file_name.startswith(prefix) for prefix in ALLOWED_FILE_PREFIXES):
            return None, "info"

        # Skip non-Python files
        if not file_name.endswith(".py"):
            return None, "info"

        # Find the relevant directory for rule matching
        parts = file_path.parts
        try:
            # Find omnibase_infra in the path
            omnibase_idx = parts.index("omnibase_infra")
            if omnibase_idx + 1 < len(parts) - 1:
                relevant_dir = parts[omnibase_idx + 1]

                # Check suffix rules first (e.g., *_validator.py in validation/)
                if relevant_dir in INFRA_DIRECTORY_SUFFIX_RULES:
                    suffix_rules = INFRA_DIRECTORY_SUFFIX_RULES[relevant_dir]
                    if any(file_name.endswith(suffix) for suffix in suffix_rules):
                        return None, "info"  # Valid suffix pattern

                if relevant_dir in INFRA_DIRECTORY_PREFIX_RULES:
                    required_prefixes = INFRA_DIRECTORY_PREFIX_RULES[relevant_dir]

                    # Check if file name starts with or equals any required prefix
                    matches_prefix = any(
                        file_name.startswith(prefix) or file_name == prefix
                        for prefix in required_prefixes
                    )

                    if not matches_prefix:
                        prefix_str = (
                            f"'{required_prefixes[0]}'"
                            if len(required_prefixes) == 1
                            else f"one of {required_prefixes[:3]}..."
                            if len(required_prefixes) > 3
                            else f"one of {required_prefixes}"
                        )
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

    def validate_directory(
        self, directory: Path, verbose: bool = False
    ) -> list[tuple[str, str, str]]:
        """Validate all Python files in a directory against naming conventions.

        Args:
            directory: Path to the directory to validate.
            verbose: If True, log each file as it's checked.

        Returns:
            List of tuples (file_path, message, severity).
        """
        results: list[tuple[str, str, str]] = []

        for file_path in directory.rglob("*.py"):
            # Skip pycache and archived directories
            if "__pycache__" in str(file_path) or "/archived/" in str(file_path):
                continue

            # Skip symbolic links
            if file_path.is_symlink():
                continue

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
        """
        file_prefix = rules.get("file_prefix")
        expected_dir = rules.get("directory")

        # Scan for files matching the prefix pattern
        if file_prefix:
            for file_path in self.repo_path.rglob(f"{file_prefix}*.py"):
                if "__pycache__" in str(file_path) or "/archived/" in str(file_path):
                    continue
                self._validate_class_names_in_file(file_path, category, rules, verbose)

        # Also check files in expected directories
        if expected_dir:
            for file_path in self.repo_path.rglob(f"*/{expected_dir}/*.py"):
                if file_path.name == "__init__.py":
                    continue
                if "__pycache__" in str(file_path) or "/archived/" in str(file_path):
                    continue
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
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self._check_class_naming(file_path, node, category, rules)

        except (SyntaxError, UnicodeDecodeError) as e:
            if verbose:
                print(f"Warning: Could not parse {file_path}: {e}")

    def _check_class_naming(
        self,
        file_path: Path,
        node: ast.ClassDef,
        category: str,
        rules: dict[str, str | None],
    ) -> None:
        """Check if a class name follows conventions.

        Args:
            file_path: Path to the file containing the class.
            node: AST class definition node.
            category: The category being validated.
            rules: Dictionary containing validation rules.
        """
        class_name = node.name
        pattern = rules.get("pattern")
        description = rules.get("description")

        if not pattern:
            return

        # Skip exception patterns
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
        in_relevant_file = file_prefix and file_path.name.startswith(file_prefix)
        in_relevant_dir = expected_dir and expected_dir in str(file_path)

        if not in_relevant_file and not in_relevant_dir:
            return

        # Check if class matches pattern
        if not re.match(pattern, class_name):
            # Check if it should match based on heuristics
            if self._should_match_pattern(class_name, category):
                self.violations.append(
                    NamingViolation(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        class_name=class_name,
                        expected_pattern=pattern,
                        description=description
                        or f"Must follow {category} naming conventions",
                        severity="error",
                    )
                )

    def _is_exception_class(self, class_name: str) -> bool:
        """Check if class name matches exception patterns.

        Args:
            class_name: Name of the class to check.

        Returns:
            True if class matches an exception pattern.
        """
        return any(re.match(pattern, class_name) for pattern in self.EXCEPTION_PATTERNS)

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
            if directory not in str(file_path):
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

    def _should_match_pattern(self, class_name: str, category: str) -> bool:
        """Determine if a class should match the pattern for a category.

        Uses heuristics based on keywords in the class name.

        Args:
            class_name: Name of the class to check.
            category: The category to check against.

        Returns:
            True if the class name suggests it should follow the category's pattern.
        """
        category_indicators = {
            "handlers": ["handler"],
            "dispatchers": ["dispatcher", "dispatch"],
            "stores": ["store", "storage"],
            "adapters": ["adapter"],
            "registries": ["registry"],
            "services": ["service"],
            "models": ["model", "data", "schema", "entity"],
            "enums": ["enum", "choice", "status"],
            "protocols": ["protocol", "interface"],
            "mixins": ["mixin"],
        }

        indicators = category_indicators.get(category, [])
        class_lower = class_name.lower()

        return any(indicator in class_lower for indicator in indicators)

    def _matches_any_valid_pattern(self, class_name: str) -> bool:
        """Check if a class name matches any of the defined valid naming patterns.

        This prevents false positives when a class follows a different valid pattern
        than the category being checked (e.g., Model* classes in handler files).

        Args:
            class_name: Name of the class to check.

        Returns:
            True if the class matches any valid naming pattern.
        """
        for category, rules in self.NAMING_PATTERNS.items():
            pattern = rules.get("pattern")
            if pattern and re.match(pattern, class_name):
                return True
        return False

    def generate_report(self) -> str:
        """Generate naming convention validation report.

        Returns:
            Formatted string report with violation details.
        """
        if not self.violations:
            return "All naming conventions are compliant!"

        errors = [v for v in self.violations if v.severity == "error"]
        warnings = [v for v in self.violations if v.severity == "warning"]

        report = "Naming Convention Validation Report\n"
        report += "=" * 50 + "\n\n"

        report += f"Summary: {len(errors)} errors, {len(warnings)} warnings\n\n"

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
        if "__pycache__" in str(file_path) or "/archived/" in str(file_path):
            continue
        if file_path.is_symlink():
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
    is_valid = validator.validate_naming_conventions(args.verbose)

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

    if args.json:
        import json

        output = {
            "path": str(repo_path),
            "errors": errors,
            "warnings": warnings,
            "ast_issues": ast_count,
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
            "passed": errors == 0 and ast_count == 0,
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

    has_failures = (
        errors > 0 or ast_count > 0 or (args.fail_on_warnings and warnings > 0)
    )

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
