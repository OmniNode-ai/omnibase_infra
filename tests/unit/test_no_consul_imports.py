# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Import-graph validation: no consul library imports in active runtime code.

This test validates that no importable code path in omnibase_infra depends on
the consul (python-consul2) library. It does this via actual import-graph
inspection, not grep/string matching.

Scope:
    - Active runtime source (src/omnibase_infra/)
    - Validated test helpers
    - Shipped config paths

Exclusions:
    - Archive/dormant example modules not part of shipped behavior
    - The virtual environment (.venv)
    - __pycache__ directories

Related:
    - OMN-3995: omnibase_infra full consul removal (stabilization batch)
    - OMN-3540: Remove Consul entirely from omnibase_infra runtime (prior work)
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

# Root of the installed package
SRC_ROOT = Path(__file__).parent.parent.parent / "src" / "omnibase_infra"

# Scripts that are demo/archive and not shipped runtime (excluded from check)
EXCLUDED_PATHS: frozenset[str] = frozenset()


@pytest.mark.unit
class TestNoConsulImports:
    """Verify no consul library imports exist in omnibase_infra source."""

    def _collect_python_files(self) -> list[Path]:
        """Collect all Python source files in omnibase_infra."""
        files = []
        for path in SRC_ROOT.rglob("*.py"):
            # Skip __pycache__
            if "__pycache__" in str(path):
                continue
            # Skip explicitly excluded paths
            rel = str(path.relative_to(SRC_ROOT))
            if any(excl in rel for excl in EXCLUDED_PATHS):
                continue
            files.append(path)
        return files

    def _extract_imports(self, source: str) -> list[str]:
        """Extract all import module names from Python source via AST."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    modules.append(node.module)
        return modules

    def test_no_consul_library_imports_in_source(self) -> None:
        """No Python file in src/omnibase_infra imports the consul library.

        This test inspects the AST of every source file to find import
        statements that reference the 'consul' package (python-consul2).
        It does NOT do string grep — it parses actual import nodes.
        """
        files = self._collect_python_files()
        assert len(files) > 0, "Expected to find Python source files"

        violations: list[str] = []
        for path in files:
            try:
                source = path.read_text(encoding="utf-8")
            except OSError:
                continue

            modules = self._extract_imports(source)
            for mod in modules:
                # Check if the import is from the consul package
                if mod == "consul" or mod.startswith("consul."):
                    violations.append(f"{path}: imports '{mod}'")

        assert not violations, (
            "Found consul library imports in omnibase_infra source. "
            "Consul was removed in OMN-3540. Remove these imports:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_consul_not_importable_via_runtime_modules(self) -> None:
        """The consul package is not available as an importable module.

        After removing python-consul2 from dependencies, importing 'consul'
        should raise ImportError. This guards against the package being
        re-added as a transitive dependency.
        """
        import importlib.util

        spec = importlib.util.find_spec("consul")
        assert spec is None, (
            "The 'consul' package (python-consul2) is still importable. "
            "This means it is still present as a direct or transitive dependency. "
            "Remove it from pyproject.toml and regenerate uv.lock."
        )

    def test_python_consul2_not_in_installed_packages(self) -> None:
        """python-consul2 is not present in the installed package set.

        Checks the importlib metadata to verify python-consul2 is not
        installed in the current environment.
        """
        import importlib.metadata

        try:
            importlib.metadata.version("python-consul2")
            installed = True
        except importlib.metadata.PackageNotFoundError:
            installed = False

        assert not installed, (
            "python-consul2 is still installed in the current environment. "
            "Run 'uv sync' after removing it from pyproject.toml to clean up."
        )

    def test_no_consul_in_sys_modules(self) -> None:
        """The consul module is not loaded in sys.modules.

        If consul was imported as a side effect of test setup or other
        imports, it would appear in sys.modules. This test ensures it
        hasn't been loaded.
        """
        consul_modules = [
            key for key in sys.modules if key == "consul" or key.startswith("consul.")
        ]
        assert not consul_modules, (
            "The consul module is loaded in sys.modules. "
            "Something is importing consul as a side effect:\n"
            + "\n".join(f"  - {m}" for m in consul_modules)
        )
