# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Architecture rule: no direct adapter usage outside handlers and tests.

Validates that ``_internal`` adapters are only imported by their corresponding
handlers and test modules. Application code, nodes, orchestrators, and other
infrastructure components MUST NOT import adapters directly.

This rule enforces the handler-owns-cross-cutting-concerns architecture where:
- Adapters are thin SDK wrappers (no caching, no circuit breaking)
- Handlers own caching, circuit breaking, retry, and audit logging
- Application code accesses secrets only through handlers

.. versionadded:: 0.9.0
    Initial implementation for OMN-2286.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Modules that are ALLOWED to import from adapters._internal
_ALLOWED_IMPORT_PATTERNS: frozenset[str] = frozenset(
    {
        "omnibase_infra.handlers.",  # Handlers can use adapters
        "tests.",  # Test modules can use adapters
    }
)

# The import pattern we're looking for
_INTERNAL_ADAPTER_MODULE = "omnibase_infra.adapters._internal"


@dataclass(frozen=True)
class ModelAdapterViolation:
    """A single violation of the no-direct-adapter-usage rule.

    Attributes:
        file_path: Path to the file containing the violation.
        line_number: Line number of the offending import.
        module_imported: The _internal module being imported.
        message: Human-readable violation description.
    """

    file_path: str
    line_number: int
    module_imported: str
    message: str


def check_no_direct_adapter_usage(
    source_root: Path,
    *,
    exclude_dirs: frozenset[str] | None = None,
) -> list[ModelAdapterViolation]:
    """Scan Python source files for direct _internal adapter imports.

    Args:
        source_root: Root directory to scan (e.g., ``src/omnibase_infra``).
        exclude_dirs: Directory names to skip (default: ``__pycache__``, ``.git``).

    Returns:
        List of ModelAdapterViolation for each offending import found.
    """
    if exclude_dirs is None:
        exclude_dirs = frozenset(
            {"__pycache__", ".git", ".mypy_cache", ".pytest_cache"}
        )

    violations: list[ModelAdapterViolation] = []

    for py_file in source_root.rglob("*.py"):
        # Skip excluded directories
        if any(part in exclude_dirs for part in py_file.parts):
            continue

        # Determine the module path for allowlist checking.
        # Assumes source_root is src/omnibase_infra (or similar), so
        # parent.parent resolves to the ``src/`` directory, yielding a
        # dotted module path like ``omnibase_infra.adapters.foo``.
        try:
            rel_path = py_file.relative_to(source_root.parent.parent).as_posix()
        except ValueError as e:
            raise ValueError(
                f"Cannot compute relative path for {py_file} from "
                f"{source_root.parent.parent}. Ensure source_root is a "
                f"directory like src/omnibase_infra (two levels below the "
                f"project root)."
            ) from e
        module_path = rel_path.replace("/", ".").removesuffix(".py")

        # Check if this module is in the allowlist
        if _is_allowed_importer(module_path, py_file.name):
            continue

        # Parse and check for _internal imports
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _INTERNAL_ADAPTER_MODULE in alias.name:
                        violations.append(
                            ModelAdapterViolation(
                                file_path=str(py_file),
                                line_number=node.lineno,
                                module_imported=alias.name,
                                message=f"Direct import of _internal adapter '{alias.name}' "
                                f"is not allowed. Use the corresponding handler instead.",
                            )
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module and _INTERNAL_ADAPTER_MODULE in node.module:
                    violations.append(
                        ModelAdapterViolation(
                            file_path=str(py_file),
                            line_number=node.lineno,
                            module_imported=node.module,
                            message=f"Direct import from _internal adapter '{node.module}' "
                            f"is not allowed. Use the corresponding handler instead.",
                        )
                    )

    return violations


def _is_allowed_importer(module_path: str, filename: str) -> bool:
    """Check if a module is allowed to import from _internal adapters.

    Args:
        module_path: Dotted module path.
        filename: The file name (for test file detection).

    Returns:
        True if the module is allowed to import adapters directly.
    """
    # Handler modules are allowed
    for pattern in _ALLOWED_IMPORT_PATTERNS:
        if pattern in module_path:
            return True

    # Files named test_* are allowed
    if filename.startswith("test_"):
        return True

    # The _internal package itself is allowed
    if "_internal" in module_path:
        return True

    return False


__all__: list[str] = [
    "ModelAdapterViolation",
    "check_no_direct_adapter_usage",
]
