# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Architecture invariant tests for handler classification compliance.

Verifies the three architectural invariants from OMN-783 (INFRA-042):

1. All handler classes expose ``handler_type`` property returning ``EnumHandlerType``
2. ``wiring.py`` is the only handler registration location
3. No ``os.getenv`` / ``os.environ`` direct access in handler files

These tests run as CI gates to prevent regressions.

Ticket: OMN-783
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SRC_ROOT = Path(__file__).parent.parent.parent / "src" / "omnibase_infra"

# Handler file directories to scan (all handler_*.py files)
_HANDLER_SEARCH_DIRS: tuple[Path, ...] = (
    _SRC_ROOT / "handlers",
    _SRC_ROOT / "nodes",
    _SRC_ROOT / "observability" / "handlers",
)

# Runtime utility files that are named handler_* but are NOT ONEX protocol
# handlers. They serve as loaders, registries, resolvers, or identity helpers.
# These are excluded from handler_type/handler_category requirements.
_RUNTIME_UTILITY_EXCLUSIONS: frozenset[str] = frozenset(
    {
        # Runtime infrastructure (loaders, registries, resolvers)
        "handler_routing_loader.py",
        "handler_bootstrap_source.py",
        "handler_contract_config_loader.py",
        "handler_contract_source.py",
        "handler_identity.py",
        "handler_registry.py",
        "handler_source_resolver.py",
        "handler_plugin_loader.py",
        # Validation check executors (ABC-based, not ONEX handler protocol)
        "handler_artifact.py",
        "handler_check_executor.py",
        "handler_measurement.py",
        "handler_risk.py",
    }
)

# Patterns that indicate direct environment variable access
_ENV_ACCESS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bos\.getenv\b"),
    re.compile(r"\bos\.environ\b"),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _discover_handler_files() -> list[Path]:
    """Find all handler_*.py files in handler directories.

    Excludes __init__.py and non-handler files.

    Returns:
        Sorted list of handler file paths.
    """
    handler_files: list[Path] = []
    for search_dir in _HANDLER_SEARCH_DIRS:
        if not search_dir.exists():
            continue
        for root_str, _dirs, files in os.walk(search_dir):
            root = Path(root_str)
            for f in files:
                if f.startswith("handler_") and f.endswith(".py"):
                    handler_files.append(root / f)
    return sorted(set(handler_files))


def _is_runtime_utility(filepath: Path) -> bool:
    """Check if a handler file is a runtime utility (not an ONEX handler)."""
    return filepath.name in _RUNTIME_UTILITY_EXCLUSIONS


def _file_defines_class_with_property(filepath: Path, property_name: str) -> bool:
    """Check if a Python file defines a class with the given property.

    Uses AST parsing for accuracy -- does not rely on regex.

    Args:
        filepath: Path to the Python file.
        property_name: Name of the property to search for.

    Returns:
        True if at least one class in the file defines the property.
    """
    try:
        tree = ast.parse(filepath.read_text(encoding="utf-8"))
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == property_name:
                    for decorator in item.decorator_list:
                        if (
                            isinstance(decorator, ast.Name)
                            and decorator.id == "property"
                        ):
                            return True
    return False


def _scan_file_for_env_access(filepath: Path) -> list[tuple[int, str]]:
    """Scan a file for direct os.getenv/os.environ access.

    Skips comments, docstrings (approximated), and TYPE_CHECKING blocks.

    Args:
        filepath: Path to the Python file.

    Returns:
        List of (line_number, line_content) tuples for violations.
    """
    violations: list[tuple[int, str]] = []
    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return violations

    in_multiline_string = False
    multiline_delimiter: str | None = None

    for line_no, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()

        # Track multiline strings (docstrings)
        if not in_multiline_string:
            for delim in ('"""', "'''"):
                count = stripped.count(delim)
                if count % 2 == 1:  # Odd count = entering/exiting multiline
                    in_multiline_string = True
                    multiline_delimiter = delim
                    break
            if in_multiline_string:
                continue
        else:
            if multiline_delimiter and multiline_delimiter in stripped:
                in_multiline_string = False
                multiline_delimiter = None
            continue

        # Skip comment-only lines
        if stripped.startswith("#"):
            continue

        # Skip lines with ONEX_EXCLUDE marker
        if "ONEX_EXCLUDE" in line:
            continue

        # Check for env access patterns
        for pattern in _ENV_ACCESS_PATTERNS:
            if pattern.search(line):
                violations.append((line_no, stripped))
                break

    return violations


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHandlerTypeInvariant:
    """INV-1: All handlers must expose handler_type property returning EnumHandlerType."""

    def test_all_handler_files_discovered(self) -> None:
        """Smoke test: handler discovery finds a reasonable number of files."""
        handler_files = _discover_handler_files()
        # We know there are 60+ handler files in the codebase
        assert len(handler_files) >= 50, (
            f"Expected 50+ handler files, found {len(handler_files)}. "
            "Handler discovery may be broken."
        )

    def test_all_protocol_handlers_have_handler_type(self) -> None:
        """Every handler_*.py file (excluding runtime utilities) must define handler_type."""
        handler_files = _discover_handler_files()
        missing: list[str] = []

        for filepath in handler_files:
            if _is_runtime_utility(filepath):
                continue

            has_handler_type = _file_defines_class_with_property(
                filepath, "handler_type"
            )
            if not has_handler_type:
                rel = filepath.relative_to(_SRC_ROOT.parent.parent)
                missing.append(str(rel))

        assert not missing, (
            f"handler_type property missing in {len(missing)} handler(s):\n"
            + "\n".join(f"  - {f}" for f in sorted(missing))
            + "\n\nAll handlers must define:\n"
            "  @property\n"
            "  def handler_type(self) -> EnumHandlerType: ...\n\n"
            "If this file is NOT an ONEX handler, add it to "
            "_RUNTIME_UTILITY_EXCLUSIONS in this test."
        )

    def test_all_protocol_handlers_have_handler_category(self) -> None:
        """Every handler_*.py file (excluding runtime utilities) must define handler_category."""
        handler_files = _discover_handler_files()
        missing: list[str] = []

        for filepath in handler_files:
            if _is_runtime_utility(filepath):
                continue

            has_category = _file_defines_class_with_property(
                filepath, "handler_category"
            )
            if not has_category:
                rel = filepath.relative_to(_SRC_ROOT.parent.parent)
                missing.append(str(rel))

        assert not missing, (
            f"handler_category property missing in {len(missing)} handler(s):\n"
            + "\n".join(f"  - {f}" for f in sorted(missing))
            + "\n\nAll handlers must define:\n"
            "  @property\n"
            "  def handler_category(self) -> EnumHandlerTypeCategory: ...\n\n"
            "If this file is NOT an ONEX handler, add it to "
            "_RUNTIME_UTILITY_EXCLUSIONS in this test."
        )

    def test_handler_category_returns_valid_enum(self) -> None:
        """handler_category properties must return a valid EnumHandlerTypeCategory member."""
        from omnibase_infra.enums import EnumHandlerTypeCategory

        valid_values = set(EnumHandlerTypeCategory)
        handler_files = _discover_handler_files()
        bad: list[str] = []

        for filepath in handler_files:
            if _is_runtime_utility(filepath):
                continue

            if not _file_defines_class_with_property(filepath, "handler_category"):
                continue

            # Check that the file contains an EnumHandlerTypeCategory return value
            content = filepath.read_text(encoding="utf-8")
            has_valid_return = any(
                f"EnumHandlerTypeCategory.{member.name}" in content
                for member in valid_values
            )
            if not has_valid_return:
                rel = filepath.relative_to(_SRC_ROOT.parent.parent)
                bad.append(str(rel))

        assert not bad, (
            "handler_category does not return a valid EnumHandlerTypeCategory in:\n"
            + "\n".join(f"  - {f}" for f in sorted(bad))
        )

    def test_new_handlers_use_enum_handler_type(self) -> None:
        """New handlers (with EnumHandlerType import) must return enum, not plain string.

        Legacy handlers returning plain strings (e.g., "mock", "consul") are
        tracked separately. This test ensures no NEW handler introduces a plain
        string return value.
        """
        # Pattern matches "EnumHandlerType" but NOT "EnumHandlerTypeCategory"
        import_pattern = re.compile(r"\bEnumHandlerType\b(?!Category)")

        handler_files = _discover_handler_files()
        bad: list[str] = []

        for filepath in handler_files:
            if _is_runtime_utility(filepath):
                continue

            content = filepath.read_text(encoding="utf-8")

            # Only check files that import EnumHandlerType (not just Category)
            if not import_pattern.search(content):
                continue

            if not _file_defines_class_with_property(filepath, "handler_type"):
                continue

            # Verify it returns an enum member, not a plain string
            from omnibase_infra.enums import EnumHandlerType

            has_enum_return = any(
                f"EnumHandlerType.{member.name}" in content
                for member in EnumHandlerType
            )
            if not has_enum_return:
                rel = filepath.relative_to(_SRC_ROOT.parent.parent)
                bad.append(str(rel))

        assert not bad, (
            "handler_type imports EnumHandlerType but does not return a member in:\n"
            + "\n".join(f"  - {f}" for f in sorted(bad))
        )


class TestWiringLocationInvariant:
    """INV-2: wiring.py is the only handler registration location."""

    def test_only_one_wiring_file(self) -> None:
        """Only one wiring.py should exist in the nodes directory."""
        wiring_files: list[Path] = []
        nodes_dir = _SRC_ROOT / "nodes"
        for root_str, _dirs, files in os.walk(nodes_dir):
            root = Path(root_str)
            for f in files:
                if f == "wiring.py":
                    wiring_files.append(root / f)

        assert len(wiring_files) <= 1, (
            f"Found {len(wiring_files)} wiring.py files. "
            "Handler registration should only exist in one wiring.py:\n"
            + "\n".join(f"  - {f}" for f in sorted(wiring_files))
        )

    def test_wiring_exists_in_registration_orchestrator(self) -> None:
        """wiring.py must exist in node_registration_orchestrator."""
        wiring_path = (
            _SRC_ROOT / "nodes" / "node_registration_orchestrator" / "wiring.py"
        )
        assert wiring_path.exists(), (
            f"Expected wiring.py at {wiring_path}. "
            "Handler registration must be centralized in wiring.py."
        )

    def test_no_handler_registration_outside_wiring(self) -> None:
        """No handler registration calls outside wiring.py and util_container_wiring.py.

        Scans for patterns like:
        - container.service_registry.register_instance(interface=Handler...
        - register_instance(interface=Handler...

        These should only appear in:
        - wiring.py (domain-specific wiring)
        - util_container_wiring.py (generic runtime wiring)
        - registry_infra_*.py (node-level registry files)
        - plugin.py (domain plugin lifecycle)
        """
        allowed_files = frozenset(
            {
                "wiring.py",
                "util_container_wiring.py",
                "plugin.py",
            }
        )
        allowed_prefixes = ("registry_infra_",)

        pattern = re.compile(
            r"register_instance\s*\(\s*interface\s*=\s*(?:Handler|ProtocolNode)"
        )

        violations: list[str] = []
        for root_str, _dirs, files in os.walk(_SRC_ROOT):
            root = Path(root_str)
            for f in files:
                if not f.endswith(".py"):
                    continue
                if f in allowed_files:
                    continue
                if any(f.startswith(prefix) for prefix in allowed_prefixes):
                    continue

                filepath = root / f
                try:
                    content = filepath.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue

                for match in pattern.finditer(content):
                    rel = filepath.relative_to(_SRC_ROOT.parent.parent)
                    line_no = content[: match.start()].count("\n") + 1
                    violations.append(f"{rel}:{line_no}")

        assert not violations, (
            "Found handler registration outside allowed files:\n"
            + "\n".join(f"  - {v}" for v in sorted(violations))
            + "\n\nHandler registration must only appear in wiring.py, "
            "util_container_wiring.py, registry_infra_*.py, or plugin.py."
        )


class TestNoDirectEnvAccessInHandlers:
    """INV-3: No os.getenv / os.environ in handler files.

    Handlers should receive configuration through constructor injection
    or container DI, not by reading environment variables directly.

    Known exceptions are tracked with inline ONEX_EXCLUDE markers.
    """

    def test_no_env_access_in_node_handlers(self) -> None:
        """Node-level handler files must not use os.getenv / os.environ."""
        violations: list[str] = []
        nodes_dir = _SRC_ROOT / "nodes"

        for root_str, _dirs, files in os.walk(nodes_dir):
            root = Path(root_str)
            # Only scan files in handlers/ subdirectories
            if "handlers" not in root.parts:
                continue
            for f in files:
                if not f.startswith("handler_") or not f.endswith(".py"):
                    continue
                filepath = root / f
                file_violations = _scan_file_for_env_access(filepath)
                for line_no, line_content in file_violations:
                    rel = filepath.relative_to(_SRC_ROOT.parent.parent)
                    violations.append(f"{rel}:{line_no}: {line_content}")

        # Known pre-existing violations (tracked to prevent growth):
        #   handler_node_registration_acked.py:141 - os.getenv for liveness interval
        #   handler_github_api_poll.py:233 - os.environ.get for GitHub token
        #   handler_runtime_target_collect.py:54-57 - 3x os.environ.get for env/kafka/kubeconfig
        max_allowed = 5
        if len(violations) > max_allowed:
            assert False, (
                f"Found {len(violations)} env access violations in node handlers "
                f"(max allowed: {max_allowed}):\n"
                + "\n".join(f"  - {v}" for v in sorted(violations))
                + "\n\nHandlers should receive config through constructor injection. "
                "Use ONEX_EXCLUDE marker for exceptions."
            )

    def test_no_env_access_in_infra_handlers(self) -> None:
        """Top-level infrastructure handlers should minimize os.getenv usage.

        Tracks known violations to prevent growth. New handlers must not
        introduce os.getenv -- use constructor injection instead.
        """
        violations: list[str] = []
        handlers_dir = _SRC_ROOT / "handlers"

        for root_str, _dirs, files in os.walk(handlers_dir):
            root = Path(root_str)
            for f in files:
                if not f.startswith("handler_") or not f.endswith(".py"):
                    continue
                filepath = root / f
                file_violations = _scan_file_for_env_access(filepath)
                for line_no, line_content in file_violations:
                    rel = filepath.relative_to(_SRC_ROOT.parent.parent)
                    violations.append(f"{rel}:{line_no}: {line_content}")

        # Known violations: handler_gmail_api (3) + handler_slack_webhook (3)
        # These use os.getenv as fallback when constructor args are None.
        max_allowed = 6
        if len(violations) > max_allowed:
            assert False, (
                f"Found {len(violations)} env access violations in infra handlers "
                f"(max allowed: {max_allowed}):\n"
                + "\n".join(f"  - {v}" for v in sorted(violations))
                + "\n\nNew handlers must use constructor injection, not os.getenv."
            )


class TestRuntimeUtilityExclusionsValid:
    """Meta-test: verify that exclusion list entries correspond to real files."""

    def test_all_exclusions_map_to_existing_files(self) -> None:
        """Every entry in _RUNTIME_UTILITY_EXCLUSIONS must correspond to a real file."""
        handler_files = _discover_handler_files()
        all_names = {f.name for f in handler_files}

        # Also check runtime/ and validation/ directories
        for search_dir in (_SRC_ROOT / "runtime", _SRC_ROOT / "validation"):
            if not search_dir.exists():
                continue
            for root_str, _dirs, files in os.walk(search_dir):
                for f in files:
                    if f.startswith("handler_") and f.endswith(".py"):
                        all_names.add(f)

        orphaned = _RUNTIME_UTILITY_EXCLUSIONS - all_names
        assert not orphaned, (
            "Exclusion entries with no matching file:\n"
            + "\n".join(f"  - {f}" for f in sorted(orphaned))
            + "\n\nRemove stale entries from _RUNTIME_UTILITY_EXCLUSIONS."
        )
