# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared AST analysis utilities for testing.

This module provides reusable functions for analyzing Python source code ASTs
in tests. These utilities are extracted from orchestrator and reducer purity
tests to enable consistent AST analysis across test modules.

Usage:
    from tests.helpers.ast_analysis import (
        get_imported_root_modules,
        find_datetime_now_calls,
        find_time_module_calls,
        find_io_method_calls,
        is_docstring,
    )

    # Parse source code
    tree = ast.parse(source_code)

    # Find all imported modules
    imports = get_imported_root_modules(tree)

    # Check for forbidden clock calls
    datetime_violations = find_datetime_now_calls(tree)
    time_violations = find_time_module_calls(tree)

Related Tickets:
    - OMN-952: Comprehensive orchestrator tests
    - OMN-914: Reducer purity enforcement

See Also:
    - tests/unit/nodes/test_orchestrator_no_io.py
    - tests/unit/nodes/test_orchestrator_time_injection.py
    - tests/unit/nodes/reducers/test_reducer_purity.py
"""

from __future__ import annotations

import ast

__all__ = [
    "find_datetime_now_calls",
    "find_io_method_calls",
    "find_time_module_calls",
    "get_imported_root_modules",
    "is_docstring",
]


# =============================================================================
# Import Analysis
# =============================================================================


def get_imported_root_modules(tree: ast.AST) -> set[str]:
    """Extract root module names from all import statements in an AST.

    This function walks the AST and extracts the root module name from
    both `import X` and `from X import Y` statements.

    Args:
        tree: The AST tree to analyze.

    Returns:
        A set of root module names (e.g., {"requests", "psycopg2", "pydantic"}).

    Example:
        >>> import ast
        >>> source = '''
        ... import os
        ... import json.decoder
        ... from pathlib import Path
        ... from typing import Optional
        ... '''
        >>> tree = ast.parse(source)
        >>> modules = get_imported_root_modules(tree)
        >>> sorted(modules)
        ['json', 'os', 'pathlib', 'typing']
    """
    imported_modules: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Extract root module (e.g., "consul" from "consul.client")
                imported_modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module.split(".")[0])

    return imported_modules


# =============================================================================
# Clock Function Detection
# =============================================================================


def find_datetime_now_calls(tree: ast.AST) -> list[str]:
    """Find all datetime.now() and datetime.utcnow() calls in the AST.

    This function walks the AST and detects:
    - datetime.datetime.now() calls (attribute access on 'datetime' module)
    - datetime.now() calls (after 'from datetime import datetime')
    - datetime.datetime.utcnow() calls (deprecated but still checked)
    - datetime.utcnow() calls (deprecated but still checked)

    Args:
        tree: The AST tree to analyze.

    Returns:
        List of descriptions of datetime.now/utcnow calls found,
        including line numbers for debugging.

    Example:
        >>> import ast
        >>> source = '''
        ... from datetime import datetime
        ... now = datetime.now()
        ... '''
        >>> tree = ast.parse(source)
        >>> calls = find_datetime_now_calls(tree)
        >>> len(calls)
        1
        >>> 'datetime.now()' in calls[0]
        True
    """
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func

            # Pattern 1: datetime.datetime.now() or datetime.datetime.utcnow()
            if isinstance(func, ast.Attribute):
                attr_name = func.attr
                if attr_name in ("now", "utcnow"):
                    # Check if it's on datetime or datetime.datetime
                    if isinstance(func.value, ast.Attribute):
                        # datetime.datetime.now()
                        if (
                            func.value.attr == "datetime"
                            and isinstance(func.value.value, ast.Name)
                            and func.value.value.id == "datetime"
                        ):
                            violations.append(
                                f"datetime.datetime.{attr_name}() at line {node.lineno}"
                            )
                    elif isinstance(func.value, ast.Name):
                        # datetime.now() (after 'from datetime import datetime')
                        if func.value.id == "datetime":
                            violations.append(
                                f"datetime.{attr_name}() at line {node.lineno}"
                            )

    return violations


def find_time_module_calls(tree: ast.AST) -> list[str]:
    """Find all time.time() and time.monotonic() calls in the AST.

    This function walks the AST and detects:
    - time.time() calls
    - time.monotonic() calls
    - time.perf_counter() calls (also a clock function)

    Args:
        tree: The AST tree to analyze.

    Returns:
        List of descriptions of time module calls found,
        including line numbers for debugging.

    Example:
        >>> import ast
        >>> source = '''
        ... import time
        ... now = time.time()
        ... mono = time.monotonic()
        ... '''
        >>> tree = ast.parse(source)
        >>> calls = find_time_module_calls(tree)
        >>> len(calls)
        2
    """
    violations: list[str] = []

    # Time functions that indicate system clock access
    forbidden_time_functions = {"time", "monotonic", "perf_counter"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func

            # Pattern: time.time(), time.monotonic(), time.perf_counter()
            if isinstance(func, ast.Attribute):
                if (
                    func.attr in forbidden_time_functions
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "time"
                ):
                    violations.append(f"time.{func.attr}() at line {node.lineno}")

    return violations


# =============================================================================
# I/O Method Detection
# =============================================================================


def find_io_method_calls(
    tree: ast.AST,
    method_patterns: frozenset[str],
    class_name: str | None = None,
) -> list[str]:
    """Find method calls matching I/O patterns in the AST.

    This function walks the AST and detects attribute calls (like client.get())
    where the method name matches one of the provided patterns. Optionally
    restricts the search to a specific class.

    Args:
        tree: The AST tree to analyze.
        method_patterns: Set of lowercase method names to look for
            (e.g., {"get", "post", "execute", "query"}).
        class_name: Optional class name to restrict search to.
            If provided, only methods within that class are checked.

    Returns:
        List of descriptions of I/O method calls found,
        including method name and line number.

    Example:
        >>> import ast
        >>> source = '''
        ... class MyService:
        ...     def process(self):
        ...         self.client.get("/api/data")
        ...         self.db.execute("SELECT 1")
        ... '''
        >>> tree = ast.parse(source)
        >>> patterns = frozenset({"get", "execute"})
        >>> calls = find_io_method_calls(tree, patterns, "MyService")
        >>> len(calls)
        2
    """
    io_calls_found: list[str] = []

    def check_node(node: ast.AST) -> None:
        """Check a node for I/O method calls."""
        for item in ast.walk(node):
            if isinstance(item, ast.Call):
                # Check for attribute calls like client.get()
                if isinstance(item.func, ast.Attribute):
                    method_name = item.func.attr.lower()
                    if method_name in method_patterns:
                        io_calls_found.append(
                            f"{item.func.attr}() at line {item.lineno}"
                        )

    if class_name is not None:
        # Search only within the specified class
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                check_node(node)
                break
    else:
        # Search the entire tree
        check_node(tree)

    return io_calls_found


# =============================================================================
# Statement Analysis
# =============================================================================


def is_docstring(stmt: ast.stmt) -> bool:
    """Check if a statement is a docstring (Expr containing a Constant string).

    Docstrings are the first statement in a module, class, or function body
    that is an expression containing only a string constant.

    Args:
        stmt: An AST statement node to check.

    Returns:
        True if the statement is a docstring, False otherwise.

    Example:
        >>> import ast
        >>> source = '''
        ... def foo():
        ...     \"\"\"This is a docstring.\"\"\"
        ...     pass
        ... '''
        >>> tree = ast.parse(source)
        >>> func_def = tree.body[0]
        >>> is_docstring(func_def.body[0])
        True
        >>> is_docstring(func_def.body[1])
        False
    """
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )
