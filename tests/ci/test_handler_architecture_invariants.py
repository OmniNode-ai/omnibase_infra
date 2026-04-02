# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Architecture invariant tests for handler classification compliance.

Verifies the four architectural invariants from OMN-783 (INFRA-042):

1. All handler classes expose ``handler_type`` property returning ``EnumHandlerType``
2. ``wiring.py`` is the only handler registration location
3. No ``os.getenv`` / ``os.environ`` direct access in handler files
4. Contract-declared handlers have explicit wiring paths (OMN-5345)

These tests run as CI gates to prevent regressions.

Ticket: OMN-783, OMN-5345
"""

from __future__ import annotations

import ast
import os
import re
import warnings
from pathlib import Path

import yaml

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
        # Onboarding orchestrator utility (module-level async function, not ONEX handler class)
        "handler_onboarding.py",
        # Declarative runtime boot wrappers (lightweight callables, not full ONEX handler protocol)
        "handler_contract_scan.py",
        "handler_event_bus_wiring.py",
        "handler_runtime_lifecycle.py",
        # Delegation pipeline reducers (pure-function delta() pattern, not ONEX handler protocol)
        "handler_delegation_workflow.py",
        "handler_quality_gate.py",
        "handler_delegation_routing.py",
    }
)

# Patterns that indicate direct environment variable access
_ENV_ACCESS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bos\.getenv\b"),
    re.compile(r"\bos\.environ\b"),
)

# Approved wiring file names and prefixes — shared by INV-2 and INV-4.
# Handler instantiation or register_instance() calls are only valid in these files.
_APPROVED_WIRING_NAMES: frozenset[str] = frozenset(
    {
        "wiring.py",
        "plugin.py",
        "util_container_wiring.py",
        "service_kernel.py",
    }
)
_APPROVED_WIRING_PREFIXES: tuple[str, ...] = ("registry_infra_",)

# INV-4: Contract-declared handlers that are known-unwired.
# Each entry is (contract_path_relative_to_repo, handler_class_name).
# The meta-test TestWiringExemptionsValid enforces anti-permanence:
# exemptions for handlers that pass the wiring check will cause test failure.
_INV4_WIRING_EXEMPTIONS: frozenset[tuple[str, str]] = frozenset(
    {
        # OMN-5406: Scope-check canary orchestrator handlers use payload_type_match
        # routing and are dispatched via Kafka events, not through the handler registry.
        (
            "src/omnibase_infra/nodes/node_scope_workflow_orchestrator/contract.yaml",
            "HandlerScopeCheckInitiate",
        ),
        (
            "src/omnibase_infra/nodes/node_scope_workflow_orchestrator/contract.yaml",
            "HandlerScopeFileReadComplete",
        ),
        (
            "src/omnibase_infra/nodes/node_scope_workflow_orchestrator/contract.yaml",
            "HandlerScopeExtractComplete",
        ),
        (
            "src/omnibase_infra/nodes/node_scope_workflow_orchestrator/contract.yaml",
            "HandlerScopeManifestWriteComplete",
        ),
    }
)

# Contract YAML root directory
_CONTRACTS_ROOT = _SRC_ROOT / "nodes"

# Known routing strategies — INV-4 fails on unknown values to catch contract bugs.
_KNOWN_ROUTING_STRATEGIES: frozenset[str] = frozenset(
    {
        "operation_match",
        "payload_type_match",
        "handler_type_match",
    }
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

    # Approved node directories that may contain wiring.py.
    # Each domain orchestrator that registers handlers via the DI container
    # gets its own wiring.py for domain isolation.
    _APPROVED_WIRING_NODES: frozenset[str] = frozenset(
        {
            "node_registration_orchestrator",
            "node_delegation_orchestrator",  # OMN-7040: delegation pipeline
        }
    )

    def test_only_approved_wiring_files(self) -> None:
        """wiring.py files should only exist in approved node directories."""
        wiring_files: list[Path] = []
        nodes_dir = _SRC_ROOT / "nodes"
        for root_str, _dirs, files in os.walk(nodes_dir):
            root = Path(root_str)
            for f in files:
                if f == "wiring.py":
                    wiring_files.append(root / f)

        unapproved = [
            f for f in wiring_files if f.parent.name not in self._APPROVED_WIRING_NODES
        ]
        assert not unapproved, (
            "Found wiring.py in unapproved location(s). "
            "Add to _APPROVED_WIRING_NODES if intentional:\n"
            + "\n".join(f"  - {f}" for f in sorted(unapproved))
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
        allowed_files = _APPROVED_WIRING_NAMES
        allowed_prefixes = _APPROVED_WIRING_PREFIXES

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
        #   handler_upsert_merge_gate.py:250 - LINEAR_API_KEY, no config injection path (OMN-3140)
        #   handler_upsert_merge_gate.py:253 - LINEAR_TEAM_ID, no config injection path (OMN-3140)
        #   handler_delegation_routing.py:103,123 - os.environ.get for LLM endpoint URLs (OMN-7040)
        max_allowed = 9
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


# ---------------------------------------------------------------------------
# INV-4 Helpers
# ---------------------------------------------------------------------------


def _discover_contracts_with_handler_routing() -> list[Path]:
    """Find all contract.yaml files under _CONTRACTS_ROOT that contain handler_routing."""
    contracts: list[Path] = []
    for root_str, _dirs, files in os.walk(_CONTRACTS_ROOT):
        root = Path(root_str)
        for f in files:
            if f == "contract.yaml":
                filepath = root / f
                try:
                    content = filepath.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                if "handler_routing:" in content:
                    contracts.append(filepath)
    return sorted(contracts)


def _extract_declared_handlers(
    contract_path: Path,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Extract handler class names declared in a contract's handler_routing section.

    Handles three schema variants:
    - ``operation_match``: all handlers required
    - ``payload_type_match``: all handlers required
    - ``handler_type_match``: only ``default_handler`` type required

    Args:
        contract_path: Absolute path to the contract.yaml.

    Returns:
        Tuple of (handlers, errors) where handlers is a list of
        (handler_class_name, handler_module) tuples and errors is a list of
        error messages for malformed contracts.

    """
    repo_root = _SRC_ROOT.parent.parent
    rel_path = str(contract_path.relative_to(repo_root))
    errors: list[str] = []

    try:
        data = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return [], [f"{rel_path}: YAML parse error: {exc}"]

    if data is None:
        return [], [f"{rel_path}: empty YAML"]

    routing = data.get("handler_routing")
    if routing is None:
        return [], []
    if not isinstance(routing, dict):
        return [], [f"{rel_path}: handler_routing is not a dict"]

    strategy = routing.get("routing_strategy")
    if strategy is None:
        return [], [f"{rel_path}: handler_routing missing routing_strategy"]
    if strategy not in _KNOWN_ROUTING_STRATEGIES:
        return [], [f"{rel_path}: unknown routing_strategy '{strategy}'"]

    handlers_list = routing.get("handlers")
    if handlers_list is None or not isinstance(handlers_list, list):
        return [], [f"{rel_path}: handler_routing.handlers missing or not a list"]

    default_handler = routing.get("default_handler")
    seen_names: set[str] = set()
    result: list[tuple[str, str]] = []

    for entry in handlers_list:
        if not isinstance(entry, dict):
            errors.append(f"{rel_path}: handler entry is not a dict")
            continue

        # For handler_type_match with default_handler, skip non-default types
        if strategy == "handler_type_match" and default_handler is not None:
            handler_type = entry.get("handler_type")
            if handler_type != default_handler:
                continue

        # Extract handler class name — three possible locations
        handler_name: str | None = None
        handler_module: str = ""

        # Pattern 1: handler.name (nested dict)
        handler_dict = entry.get("handler")
        if isinstance(handler_dict, dict):
            handler_name = handler_dict.get("name")
            handler_module = handler_dict.get("module", "")

        # Pattern 2: handler_class (flat string)
        if handler_name is None:
            handler_name = entry.get("handler_class")
            handler_module = entry.get("handler_module", "")

        if handler_name is None:
            errors.append(
                f"{rel_path}: handler entry missing both handler.name and handler_class"
            )
            continue

        if handler_name in seen_names:
            warnings.warn(
                f"{rel_path}: duplicate handler declaration '{handler_name}'",
                stacklevel=1,
            )
        else:
            seen_names.add(handler_name)
            result.append((handler_name, handler_module))

    # Validate default_handler references an existing handler_type
    if strategy == "handler_type_match" and default_handler is not None:
        declared_types = {
            e.get("handler_type")
            for e in handlers_list
            if isinstance(e, dict) and e.get("handler_type")
        }
        if default_handler not in declared_types:
            errors.append(
                f"{rel_path}: default_handler '{default_handler}' "
                f"not found in declared handler_types {sorted(declared_types)}"
            )

    return result, errors


def _collect_approved_wiring_file_contents() -> dict[Path, str]:
    """Read all approved wiring files and return {path: content} map.

    Approved wiring files are defined by ``_APPROVED_WIRING_NAMES`` and
    ``_APPROVED_WIRING_PREFIXES``.
    """
    result: dict[Path, str] = {}
    for root_str, _dirs, files in os.walk(_SRC_ROOT):
        root = Path(root_str)
        for f in files:
            if not f.endswith(".py"):
                continue
            if f in _APPROVED_WIRING_NAMES or any(
                f.startswith(p) for p in _APPROVED_WIRING_PREFIXES
            ):
                filepath = root / f
                try:
                    result[filepath] = filepath.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
    return result


def _handler_has_wiring_path(
    handler_name: str,
    handler_module: str,
    wiring_contents: dict[Path, str],
) -> bool:
    """Check if a handler has a structural wiring path.

    A handler passes if ANY of the following holds:

    1. **Explicit wiring**: Handler class name appears (constructor call or
       register_instance) in an approved wiring file.
    2. **Module-level proof**: The handler's declared module file exists on disk
       AND defines the handler class. This covers handlers loaded dynamically
       by the contract-driven ``HandlerPluginLoader``.

    Both checks are structural — they prove the handler exists and is reachable,
    not that runtime dispatch is correct.
    """
    # Tier 1: Explicit wiring in approved files (constructor call or register)
    call_pattern = re.compile(
        rf"(?:{re.escape(handler_name)}\s*\(|interface\s*=\s*{re.escape(handler_name)}\b)"
    )
    if any(call_pattern.search(content) for content in wiring_contents.values()):
        return True

    # Tier 2: Module exists AND defines the handler class.
    # This covers handlers loaded by HandlerPluginLoader via contract YAML.
    if handler_module:
        module_path = (
            _SRC_ROOT.parent.parent / "src" / handler_module.replace(".", os.sep)
        )
        module_file = Path(str(module_path) + ".py")
        if module_file.exists():
            try:
                content = module_file.read_text(encoding="utf-8")
                class_pattern = re.compile(
                    rf"^class\s+{re.escape(handler_name)}\b", re.MULTILINE
                )
                if class_pattern.search(content):
                    return True
            except (OSError, UnicodeDecodeError):
                pass

    return False


# ---------------------------------------------------------------------------
# INV-4 Tests
# ---------------------------------------------------------------------------


class TestContractHandlerWiringInvariant:
    """INV-4: Contract-declared handlers must have explicit wiring paths.

    Every handler declared in a contract's handler_routing section must be
    instantiated or registered via register_instance() in an approved wiring
    location. This catches the class of bug where a handler is declared in a
    contract but never wired at runtime (OMN-5345).
    """

    def test_all_contract_declared_handlers_are_wired(self) -> None:
        """Every required handler from contract YAML must have a wiring path."""
        contracts = _discover_contracts_with_handler_routing()
        assert len(contracts) >= 30, (
            f"Expected 30+ contracts with handler_routing, found {len(contracts)}. "
            "Contract discovery may be broken."
        )

        wiring_contents = _collect_approved_wiring_file_contents()
        repo_root = _SRC_ROOT.parent.parent
        all_errors: list[str] = []
        unwired: list[str] = []

        for contract_path in contracts:
            handlers, errors = _extract_declared_handlers(contract_path)
            all_errors.extend(errors)

            rel_contract = str(contract_path.relative_to(repo_root))
            for handler_name, handler_module in handlers:
                # Check exemptions
                if (rel_contract, handler_name) in _INV4_WIRING_EXEMPTIONS:
                    continue

                if not _handler_has_wiring_path(
                    handler_name, handler_module, wiring_contents
                ):
                    unwired.append(
                        f"  - {handler_name}\n"
                        f"    contract: {rel_contract}\n"
                        f"    module: {handler_module}"
                    )

        # Fail on malformed contracts first — these are contract bugs
        assert not all_errors, (
            f"INV-4: {len(all_errors)} malformed contract(s):\n"
            + "\n".join(f"  - {e}" for e in all_errors)
        )

        assert not unwired, (
            f"INV-4 VIOLATION: {len(unwired)} contract-declared handler(s) "
            "have no wiring path:\n"
            + "\n".join(unwired)
            + "\n\nEvery handler declared in a contract's handler_routing must be "
            "instantiated or\nregistered via register_instance() in an approved "
            "wiring location.\n\n"
            "To fix:\n"
            "  1. Add handler instantiation in the appropriate wiring file, OR\n"
            "  2. Add to _INV4_WIRING_EXEMPTIONS with justification if intentional."
        )

    def test_contracts_discovered(self) -> None:
        """Smoke test: contract discovery finds expected contracts."""
        contracts = _discover_contracts_with_handler_routing()
        assert len(contracts) >= 30, (
            f"Expected 30+ contracts with handler_routing, found {len(contracts)}."
        )


class TestWiringExemptionsValid:
    """Meta-test: verify INV-4 exemptions are structurally valid and still needed."""

    def test_exemptions_reference_real_contracts_and_handlers(self) -> None:
        """Every exemption must reference an existing contract that declares that handler."""
        repo_root = _SRC_ROOT.parent.parent
        invalid: list[str] = []

        for contract_rel, handler_name in sorted(_INV4_WIRING_EXEMPTIONS):
            contract_path = repo_root / contract_rel
            if not contract_path.exists():
                invalid.append(
                    f"  - ({contract_rel!r}, {handler_name!r}): "
                    "contract file does not exist"
                )
                continue

            handlers, errors = _extract_declared_handlers(contract_path)
            if errors:
                invalid.append(
                    f"  - ({contract_rel!r}, {handler_name!r}): "
                    f"contract parse errors: {errors}"
                )
                continue

            declared_names = {h[0] for h in handlers}
            if handler_name not in declared_names:
                invalid.append(
                    f"  - ({contract_rel!r}, {handler_name!r}): "
                    f"handler not declared in contract (found: {sorted(declared_names)})"
                )

        assert not invalid, (
            "INV-4 exemption entries reference invalid contracts or handlers:\n"
            + "\n".join(invalid)
            + "\n\nFix or remove invalid entries from _INV4_WIRING_EXEMPTIONS."
        )

    def test_stale_exemptions_are_removed(self) -> None:
        """Exemptions for handlers that are now wired must be removed."""
        wiring_contents = _collect_approved_wiring_file_contents()
        repo_root = _SRC_ROOT.parent.parent
        stale: list[str] = []

        for contract_rel, handler_name in sorted(_INV4_WIRING_EXEMPTIONS):
            contract_path = repo_root / contract_rel
            if not contract_path.exists():
                continue
            handlers, _errors = _extract_declared_handlers(contract_path)
            handler_module = ""
            for h_name, h_module in handlers:
                if h_name == handler_name:
                    handler_module = h_module
                    break
            if _handler_has_wiring_path(handler_name, handler_module, wiring_contents):
                stale.append(
                    f"  - Exemption for {handler_name} is stale — handler is now "
                    "wired. Remove from _INV4_WIRING_EXEMPTIONS."
                )

        assert not stale, (
            "INV-4 stale exemptions detected:\n"
            + "\n".join(stale)
            + "\n\nThese handlers now have wiring paths. Remove the exemptions."
        )
