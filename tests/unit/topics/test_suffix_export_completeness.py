# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""CI guard: every SUFFIX_* constant in platform_topic_suffixes.py must be
exported from topics/__init__.py.

This is the third time a SUFFIX_* was defined but not re-exported, causing
ImportError at runtime in downstream repos. This test prevents recurrence.

Reference: OMN-6814
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_TOPICS_PKG = Path(__file__).resolve().parents[3] / "src" / "omnibase_infra" / "topics"
_SUFFIXES_MODULE = _TOPICS_PKG / "platform_topic_suffixes.py"
_INIT_MODULE = _TOPICS_PKG / "__init__.py"


def _extract_suffix_names(path: Path) -> set[str]:
    """Return all module-level names starting with ``SUFFIX_`` from *path*."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("SUFFIX_"):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id.startswith("SUFFIX_"):
                names.add(node.target.id)
    return names


def _extract_init_imports(path: Path) -> set[str]:
    """Return all names imported from platform_topic_suffixes in __init__.py."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and "platform_topic_suffixes" in node.module
        ):
            for alias in node.names:
                if alias.name.startswith("SUFFIX_"):
                    names.add(alias.name)
    return names


def _extract_init_all(path: Path) -> set[str]:
    """Return all SUFFIX_* entries in __all__ from __init__.py."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()

    def _collect_from_list(value: ast.expr) -> None:
        if isinstance(value, ast.List):
            for elt in value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    if elt.value.startswith("SUFFIX_"):
                        names.add(elt.value)

    for node in ast.walk(tree):
        # Plain assignment: __all__ = [...]
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    _collect_from_list(node.value)
        # Annotated assignment: __all__: list[str] = [...]
        elif isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == "__all__"
                and node.value is not None
            ):
                _collect_from_list(node.value)
    return names


@pytest.mark.unit
class TestSuffixExportCompleteness:
    """Verify that all SUFFIX_* constants are fully exported."""

    def test_all_suffixes_imported_in_init(self) -> None:
        """Every SUFFIX_* in platform_topic_suffixes.py must be imported in __init__.py."""
        defined = _extract_suffix_names(_SUFFIXES_MODULE)
        imported = _extract_init_imports(_INIT_MODULE)
        missing = defined - imported
        assert not missing, (
            f"SUFFIX_* constants defined in platform_topic_suffixes.py but NOT imported "
            f"in topics/__init__.py ({len(missing)} missing):\n"
            + "\n".join(f"  - {n}" for n in sorted(missing))
        )

    def test_all_suffixes_in_all_list(self) -> None:
        """Every SUFFIX_* imported in __init__.py must appear in __all__."""
        imported = _extract_init_imports(_INIT_MODULE)
        in_all = _extract_init_all(_INIT_MODULE)
        missing = imported - in_all
        assert not missing, (
            f"SUFFIX_* constants imported in topics/__init__.py but NOT listed "
            f"in __all__ ({len(missing)} missing):\n"
            + "\n".join(f"  - {n}" for n in sorted(missing))
        )

    def test_no_stale_suffix_in_all(self) -> None:
        """No SUFFIX_* in __all__ that doesn't exist in platform_topic_suffixes.py."""
        defined = _extract_suffix_names(_SUFFIXES_MODULE)
        in_all = _extract_init_all(_INIT_MODULE)
        stale = in_all - defined
        assert not stale, (
            f"SUFFIX_* entries in __all__ that don't exist in "
            f"platform_topic_suffixes.py ({len(stale)} stale):\n"
            + "\n".join(f"  - {n}" for n in sorted(stale))
        )
