# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""AST-based I/O Audit Test for ONEX Node Purity.

This module provides static analysis validation to detect I/O violations in ONEX
nodes that should be pure (REDUCER and COMPUTE archetypes).

ONEX Node Purity Rules:
    - REDUCER_GENERIC and COMPUTE_GENERIC nodes MUST NOT perform direct I/O
    - EFFECT_GENERIC nodes are allowed to perform I/O (they are exempt)
    - ORCHESTRATOR_GENERIC nodes coordinate but should not directly import I/O libs

Forbidden Patterns:
    1. Imports of I/O libraries (confluent_kafka, qdrant_client, neo4j, asyncpg, httpx)
    2. Access to os.environ or os.getenv
    3. File I/O operations (open(), Path methods for reading/writing)

Exemption Mechanisms:
    1. EFFECT_GENERIC nodes are fully exempt (I/O is their purpose)
    2. Files in handlers/, adapters/, services/, runtime/, tests/ directories
    3. Files matching adapter_*.py, handler_*.py, wiring.py, plugin.py patterns
    4. ``ONEX_EXCLUDE: io_audit`` comment on the same or preceding line

Usage:
    pytest tests/audit/test_io_violations.py -v

CI Integration:
    This test should run on every PR to enforce node purity.
    Violations produce clear error messages with file:line references.

Limitations:
    - The ``_is_path_related()`` function uses naming heuristics to detect Path objects.
      Variables with "path" in the name (e.g., ``file_path``, ``config_path``) are detected,
      but variables with different names (e.g., ``file``, ``source``, ``target``) holding
      Path objects may not be detected.
    - For such cases, users can add ``ONEX_EXCLUDE: io_audit`` comment to exempt the line.
"""

from __future__ import annotations

import ast
import logging
import tempfile
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from textwrap import dedent
from typing import NamedTuple

import pytest
import yaml

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Forbidden import modules for pure nodes.
#
# NOTE: `pathlib` is intentionally NOT in FORBIDDEN_IMPORTS.
# Path construction (e.g., `Path("config.yaml")`, `path / "subdir"`) is allowed
# in pure nodes because it is a pure operation that does not perform I/O.
# Only Path I/O METHODS (read_text, write_bytes, open, etc.) are forbidden.
# This distinction allows reducers to construct paths and pass them as data
# to EFFECT nodes for actual I/O operations.
#
# See PATH_IO_METHODS below for the specific methods that ARE forbidden.
FORBIDDEN_IMPORTS: frozenset[str] = frozenset(
    {
        "confluent_kafka",
        "qdrant_client",
        "neo4j",
        "asyncpg",
        "httpx",
        "aiohttp",
        "psycopg",
        "psycopg2",
        "redis",
        "aioredis",
        "aiokafka",
        "motor",  # MongoDB async driver
        "pymongo",
        "requests",  # Synchronous HTTP client
        "elasticsearch",  # Elasticsearch client
        "boto3",  # AWS SDK
        "grpc",  # gRPC library
    }
)

# os module forbidden patterns
OS_ENVIRON_PATTERNS: frozenset[str] = frozenset(
    {
        "environ",
        "getenv",
    }
)

# File I/O forbidden function calls
FILE_IO_FUNCTIONS: frozenset[str] = frozenset(
    {
        "open",
    }
)

# Path method patterns that indicate file I/O
PATH_IO_METHODS: frozenset[str] = frozenset(
    {
        "read_text",
        "read_bytes",
        "write_text",
        "write_bytes",
        "open",
    }
)

# Node types that are pure (should have no I/O)
PURE_NODE_TYPES: frozenset[str] = frozenset(
    {
        "REDUCER_GENERIC",
        "COMPUTE_GENERIC",
    }
)

# Node types that may have I/O (exempt from audit)
IO_ALLOWED_NODE_TYPES: frozenset[str] = frozenset(
    {
        "EFFECT_GENERIC",
    }
)

# Directories to skip (these are expected to have I/O)
SKIP_DIRECTORIES: frozenset[str] = frozenset(
    {
        "handlers",
        "adapters",
        "services",
        "runtime",
        "tests",
        "projectors",
        "dispatchers",
        "transport",
        "validators",  # Architecture validators do file I/O to analyze code
        # NOTE: "reducers" skips `nodes/reducers/` which contains shared utilities and
        # helper modules. This does NOT skip actual REDUCER nodes like `nodes/node_foo_reducer/`
        # which have their own contract.yaml and ARE audited for purity.
        "reducers",
        "__pycache__",
    }
)

# File patterns to skip
SKIP_FILE_PATTERNS: frozenset[str] = frozenset(
    {
        "wiring.py",
        "plugin.py",
        "conftest.py",
        "__init__.py",
    }
)

# File prefixes to skip
SKIP_FILE_PREFIXES: tuple[str, ...] = (
    "adapter_",
    "handler_",
    "service_",
    "dispatcher_",
    "transport_",
    "test_",
)

# Comment pattern for exemption
_ONEX_EXCLUDE_PATTERN = "ONEX_EXCLUDE:"
_ONEX_EXCLUDE_IO_AUDIT = "io_audit"

# Maximum file size to process (1MB)
_MAX_FILE_SIZE_BYTES: int = 1_000_000

# Number of lines covered by ONEX_EXCLUDE: io_audit exemption.
# This covers the comment line itself plus the next 5 lines (total of 6 lines).
#
# Rationale: Typical import statements span 1-3 lines (single import, from-import,
# or multi-line from-import with parentheses). The 6-line range provides headroom
# for grouped imports and multiline statements while preventing overly broad exemptions
# that could accidentally exempt unrelated code further down in the file.
#
# Example usage:
#   # ONEX_EXCLUDE: io_audit - Required for type hints
#   from httpx import (    # Line 2 - exempted
#       AsyncClient,       # Line 3 - exempted
#       Client,            # Line 4 - exempted
#   )                      # Line 5 - exempted
#   # Line 6 - still exempted (last line of range)
_ONEX_EXCLUDE_RANGE: int = 6


# =============================================================================
# Enums and Models
# =============================================================================


class EnumIOViolationType(StrEnum):
    """Types of I/O violations detected by the auditor."""

    FORBIDDEN_IMPORT = "forbidden_import"
    OS_ENVIRON = "os_environ"
    FILE_IO = "file_io"
    PATH_IO_METHOD = "path_io_method"
    SYNTAX_ERROR = "syntax_error"

    @property
    def severity(self) -> str:
        """Return severity level for this violation type."""
        if self == EnumIOViolationType.SYNTAX_ERROR:
            return "warning"
        return "error"

    @property
    def suggestion(self) -> str:
        """Return a suggestion for fixing this violation type."""
        suggestions = {
            EnumIOViolationType.FORBIDDEN_IMPORT: (
                "Move I/O operations to an EFFECT node. "
                "REDUCER and COMPUTE nodes must be pure functions."
            ),
            EnumIOViolationType.OS_ENVIRON: (
                "Inject configuration via ModelONEXContainer or function parameters. "
                "Do not read environment variables directly in pure nodes."
            ),
            EnumIOViolationType.FILE_IO: (
                "Move file I/O to an EFFECT node or inject file contents via parameters. "
                "Pure nodes should not perform file system operations."
            ),
            EnumIOViolationType.PATH_IO_METHOD: (
                "Move Path read/write operations to an EFFECT node. "
                "Pure nodes must receive data via function parameters."
            ),
            EnumIOViolationType.SYNTAX_ERROR: (
                "Fix the syntax error in the file before validation can proceed."
            ),
        }
        return suggestions.get(self, "Review the violation and apply ONEX principles.")


class IOAuditViolation(NamedTuple):
    """A single I/O violation detected during audit.

    Attributes:
        file_path: Path to the source file containing the violation.
        line_number: Line number where the violation was detected (1-indexed).
        column: Column offset where the violation appears (0-indexed).
        violation_type: The type of I/O violation detected.
        detail: Specific detail about the violation (e.g., module name, function).
        context: Additional context (e.g., node name, function name).
    """

    file_path: Path
    line_number: int
    column: int
    violation_type: EnumIOViolationType
    detail: str
    context: str = ""

    def format_for_ci(self) -> str:
        """Format violation for CI output (GitHub Actions compatible).

        Returns:
            Formatted string in GitHub Actions annotation format.
        """
        annotation_type = (
            "error" if self.violation_type.severity == "error" else "warning"
        )
        return (
            f"::{annotation_type} file={self.file_path},line={self.line_number},"
            f"col={self.column}::{self.violation_type.value}: {self.detail}"
        )

    def format_human_readable(self) -> str:
        """Format violation for human-readable console output.

        Returns:
            Formatted string with file location and suggestion.
        """
        lines = [
            f"{self.file_path}:{self.line_number}:{self.column} - {self.violation_type.value}",
            f"  Detail: {self.detail}",
            f"  Suggestion: {self.violation_type.suggestion}",
        ]
        if self.context:
            lines.insert(1, f"  Context: {self.context}")
        return "\n".join(lines)


# =============================================================================
# AST Visitor
# =============================================================================


class IOPurityAuditor(ast.NodeVisitor):
    """AST visitor that detects I/O violations in pure nodes.

    This visitor walks the AST and identifies patterns that violate
    ONEX node purity rules for REDUCER and COMPUTE archetypes.

    Attributes:
        filepath: Path to the file being analyzed.
        source_lines: List of source code lines for context extraction.
        violations: List of detected violations.
        allowed_lines: Set of line numbers exempted via ONEX_EXCLUDE comment.
        current_function: Name of the current function being visited.
        current_class: Name of the current class being visited.
        in_type_checking_block: Whether currently inside a TYPE_CHECKING block.
    """

    def __init__(self, filepath: str, source_lines: list[str]) -> None:
        """Initialize the auditor.

        Args:
            filepath: Path to the file being analyzed.
            source_lines: List of source code lines for context extraction.
        """
        self.filepath = filepath
        self.source_lines = source_lines
        self.violations: list[IOAuditViolation] = []
        self.allowed_lines: set[int] = set()
        self.current_function: str = ""
        self.current_class: str = ""
        self.in_type_checking_block: bool = False

    def _get_context(self) -> str:
        """Get current context string."""
        parts = []
        if self.current_class:
            parts.append(self.current_class)
        if self.current_function:
            parts.append(self.current_function)
        return ".".join(parts) if parts else ""

    def _add_violation(
        self,
        line: int,
        col: int,
        violation_type: EnumIOViolationType,
        detail: str,
    ) -> None:
        """Add a violation if the line is not exempted.

        Args:
            line: Line number of the violation.
            col: Column offset of the violation.
            violation_type: Type of violation detected.
            detail: Detail about the violation.
        """
        if line not in self.allowed_lines:
            self.violations.append(
                IOAuditViolation(
                    file_path=Path(self.filepath),
                    line_number=line,
                    column=col,
                    violation_type=violation_type,
                    detail=detail,
                    context=self._get_context(),
                )
            )

    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statement to detect forbidden imports."""
        # Skip imports inside TYPE_CHECKING blocks (type hints only)
        if self.in_type_checking_block:
            self.generic_visit(node)
            return

        for alias in node.names:
            module_name = alias.name.split(".")[0]
            if module_name in FORBIDDEN_IMPORTS:
                self._add_violation(
                    node.lineno,
                    node.col_offset,
                    EnumIOViolationType.FORBIDDEN_IMPORT,
                    f"Import of '{alias.name}' - I/O library not allowed in pure nodes",
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from-import statement to detect forbidden imports."""
        # Skip imports inside TYPE_CHECKING blocks (type hints only)
        if self.in_type_checking_block:
            self.generic_visit(node)
            return

        if node.module:
            root_module = node.module.split(".")[0]
            if root_module in FORBIDDEN_IMPORTS:
                # Get imported names for detail
                names = ", ".join(a.name for a in node.names)
                self._add_violation(
                    node.lineno,
                    node.col_offset,
                    EnumIOViolationType.FORBIDDEN_IMPORT,
                    f"Import from '{node.module}' ({names}) - I/O library not allowed",
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Visit attribute access to detect os.environ patterns."""
        # Check for os.environ, os.getenv patterns
        if isinstance(node.value, ast.Name) and node.value.id == "os":
            if node.attr in OS_ENVIRON_PATTERNS:
                self._add_violation(
                    node.lineno,
                    node.col_offset,
                    EnumIOViolationType.OS_ENVIRON,
                    f"Access to 'os.{node.attr}' - environment access not allowed",
                )
        # Check for os.environ.get pattern
        if (
            isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "os"
            and node.value.attr == "environ"
        ):
            self._add_violation(
                node.lineno,
                node.col_offset,
                EnumIOViolationType.OS_ENVIRON,
                f"Access to 'os.environ.{node.attr}' - environment access not allowed",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function calls to detect file I/O patterns."""
        # Check for open() builtin
        if isinstance(node.func, ast.Name) and node.func.id in FILE_IO_FUNCTIONS:
            self._add_violation(
                node.lineno,
                node.col_offset,
                EnumIOViolationType.FILE_IO,
                f"Call to '{node.func.id}()' - file I/O not allowed in pure nodes",
            )

        # Check for Path.read_text(), Path().write_bytes(), etc.
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in PATH_IO_METHODS:
                # Check if it's on a Path object or Path call
                is_path_call = self._is_path_related(node.func.value)
                if is_path_call:
                    self._add_violation(
                        node.lineno,
                        node.col_offset,
                        EnumIOViolationType.PATH_IO_METHOD,
                        f"Call to 'Path.{node.func.attr}()' - Path I/O not allowed",
                    )

        # Check for os.getenv() call pattern
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "os"
            and node.func.attr == "getenv"
        ):
            self._add_violation(
                node.lineno,
                node.col_offset,
                EnumIOViolationType.OS_ENVIRON,
                "Call to 'os.getenv()' - environment access not allowed",
            )

        self.generic_visit(node)

    def _is_path_related(self, node: ast.expr) -> bool:
        """Check if a node is related to pathlib.Path.

        NOTE: This uses a heuristic based on variable naming patterns. Variables
        named with "path" (e.g., file_path, config_path) are detected, but variables
        with different names (e.g., `file`, `source`) holding Path objects may not be
        detected. This is a known limitation - users can use ONEX_EXCLUDE for such cases.

        Args:
            node: AST expression to check.

        Returns:
            True if the node appears to be Path-related.
        """
        # Direct Path() call
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "Path":
                return True
            if isinstance(node.func, ast.Attribute) and node.func.attr == "Path":
                return True
        # Attribute access on Path-related expression
        if isinstance(node, ast.Attribute):
            return self._is_path_related(node.value)
        # Variable that might be a Path (heuristic: lowercase ending in path)
        if isinstance(node, ast.Name):
            name_lower = node.id.lower()
            return "path" in name_lower or name_lower.endswith("_path")
        return False

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition to track context."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition to track context."""
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition to track context."""
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_If(self, node: ast.If) -> None:
        """Visit if statement to detect TYPE_CHECKING blocks.

        Imports inside TYPE_CHECKING blocks are exempt from audit since
        they are only used for type hints and not executed at runtime.

        Note: The `else` branch of `if TYPE_CHECKING:` runs at runtime,
        so it is NOT exempted. Only the `body` (true branch) is exempted.
        """
        # Check if condition is TYPE_CHECKING
        is_type_checking = (
            isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"
        ) or (
            isinstance(node.test, ast.Attribute)
            and isinstance(node.test.value, ast.Name)
            and node.test.value.id == "typing"
            and node.test.attr == "TYPE_CHECKING"
        )

        if is_type_checking:
            # Visit the body (if-branch) with TYPE_CHECKING exemption
            old_in_type_checking = self.in_type_checking_block
            self.in_type_checking_block = True
            for child in node.body:
                self.visit(child)
            self.in_type_checking_block = old_in_type_checking

            # Visit the else-branch WITHOUT exemption (runs at runtime)
            for child in node.orelse:
                self.visit(child)
        else:
            self.generic_visit(node)


# =============================================================================
# Utility Functions
# =============================================================================


def _find_onex_exclude_lines(content: str) -> set[int]:
    """Find lines exempted via ONEX_EXCLUDE: io_audit comments.

    The exemption applies to the comment line and the next (_ONEX_EXCLUDE_RANGE - 1) lines.
    Total coverage: _ONEX_EXCLUDE_RANGE lines (default 6).

    Args:
        content: Source file content.

    Returns:
        Set of line numbers that are exempted.
    """
    excluded_lines: set[int] = set()
    lines = content.split("\n")

    for i, line in enumerate(lines, start=1):
        if _ONEX_EXCLUDE_PATTERN in line and _ONEX_EXCLUDE_IO_AUDIT in line:
            # Exclude this line and the following lines up to _ONEX_EXCLUDE_RANGE total
            for offset in range(_ONEX_EXCLUDE_RANGE):
                excluded_lines.add(i + offset)

    return excluded_lines


def get_node_type_from_contract(node_dir: Path) -> str | None:
    """Parse contract.yaml to determine node type.

    Args:
        node_dir: Directory containing the node (should have contract.yaml).

    Returns:
        The node_type value from contract.yaml, or None if not found.
    """
    contract_path = node_dir / "contract.yaml"
    if not contract_path.exists():
        return None

    try:
        with contract_path.open("r", encoding="utf-8") as f:
            contract: dict[str, object] = yaml.safe_load(f)
        if not isinstance(contract, dict):
            return None
        node_type = contract.get("node_type")
        return str(node_type) if node_type is not None else None
    except (yaml.YAMLError, OSError):
        logger.exception(
            "Failed to parse contract.yaml",
            extra={"path": str(contract_path)},
        )
        return None


def should_audit_file(file_path: Path, nodes_dir: Path) -> bool:
    """Determine if a file should be audited based on whitelist rules.

    Args:
        file_path: Path to the Python file.
        nodes_dir: Root nodes directory.

    Returns:
        True if the file should be audited for I/O violations.
    """
    # Skip files matching skip patterns
    if file_path.name in SKIP_FILE_PATTERNS:
        return False

    # Skip files with skip prefixes
    if file_path.name.startswith(SKIP_FILE_PREFIXES):
        return False

    # Skip files starting with underscore (except __init__.py already handled)
    if file_path.name.startswith("_"):
        return False

    # Check if any parent directory is in skip list
    for part in file_path.parts:
        if part in SKIP_DIRECTORIES:
            return False

    # Check if this is within a node directory with a contract
    # Walk up to find the node directory (should be direct parent or grandparent)
    relative = (
        file_path.relative_to(nodes_dir) if nodes_dir in file_path.parents else None
    )
    if relative:
        # Find the immediate node directory
        parts = relative.parts
        if len(parts) >= 1:
            node_name = parts[0]
            node_dir = nodes_dir / node_name

            # Get node type from contract
            node_type = get_node_type_from_contract(node_dir)

            # Skip EFFECT nodes entirely
            if node_type in IO_ALLOWED_NODE_TYPES:
                return False

            # Only audit if it's a pure node type
            if node_type in PURE_NODE_TYPES:
                return True

    # Default: audit the file (conservative approach)
    return True


def scan_file_for_io_violations(file_path: Path) -> list[IOAuditViolation]:
    """Scan a single Python file for I/O violations.

    Args:
        file_path: Path to the Python file to scan.

    Returns:
        List of detected violations. Empty if no violations found.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning(
            "Failed to read file",
            extra={"file": str(file_path), "error": str(e)},
        )
        return []

    # Find exempted lines
    excluded_lines = _find_onex_exclude_lines(content)

    try:
        tree = ast.parse(content, filename=str(file_path))
    except SyntaxError as e:
        logger.warning(
            "Syntax error in file",
            extra={"file": str(file_path), "error": str(e)},
        )
        return [
            IOAuditViolation(
                file_path=file_path,
                line_number=e.lineno or 1,
                column=0 if e.offset is None else e.offset,
                violation_type=EnumIOViolationType.SYNTAX_ERROR,
                detail=f"Syntax error: {e.msg}",
            )
        ]

    source_lines = content.split("\n")
    auditor = IOPurityAuditor(str(file_path.resolve()), source_lines)
    auditor.allowed_lines.update(excluded_lines)
    auditor.visit(tree)

    return auditor.violations


def audit_all_nodes(nodes_dir: Path) -> list[IOAuditViolation]:
    """Scan all nodes for I/O violations.

    Args:
        nodes_dir: Directory containing ONEX nodes.

    Returns:
        List of all violations found across all pure nodes.
    """
    violations: list[IOAuditViolation] = []

    for file_path in nodes_dir.rglob("*.py"):
        if not file_path.is_file():
            continue

        # Skip large files
        try:
            file_size = file_path.stat().st_size
            if file_size > _MAX_FILE_SIZE_BYTES:
                logger.warning(
                    "Skipping file exceeding size limit",
                    extra={
                        "file": str(file_path),
                        "size_bytes": file_size,
                        "limit_bytes": _MAX_FILE_SIZE_BYTES,
                    },
                )
                continue
        except OSError:
            continue

        # Check if file should be audited
        if not should_audit_file(file_path, nodes_dir):
            continue

        try:
            file_violations = scan_file_for_io_violations(file_path)
            violations.extend(file_violations)
        except Exception as e:  # catch-all-ok: validation continues on file errors
            logger.warning(
                "Failed to audit file",
                extra={
                    "file": str(file_path),
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
                exc_info=True,  # Include full traceback for debugging
            )

    return violations


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


def _create_test_file(temp_dir: Path, content: str, filename: str = "node.py") -> Path:
    """Create a test Python file with given content.

    Args:
        temp_dir: Directory to create file in.
        content: Python source code content.
        filename: Name of the file to create.

    Returns:
        Path to created file.
    """
    filepath = temp_dir / filename
    filepath.write_text(dedent(content))
    return filepath


def _create_contract(temp_dir: Path, node_type: str) -> Path:
    """Create a contract.yaml file with the given node type.

    Args:
        temp_dir: Directory to create contract in.
        node_type: Node type value (e.g., "REDUCER_GENERIC").

    Returns:
        Path to created contract.yaml.
    """
    contract_path = temp_dir / "contract.yaml"
    contract_content = f"""
contract_version:
  major: 1
  minor: 0
  patch: 0
name: "test_node"
node_type: "{node_type}"
description: "Test node"
"""
    contract_path.write_text(dedent(contract_content))
    return contract_path


# =============================================================================
# Detection Tests: Forbidden Imports
# =============================================================================


class TestDetectionForbiddenImports:
    """Test detection of forbidden I/O library imports."""

    def test_import_confluent_kafka(self, temp_dir: Path) -> None:
        """Detect import of confluent_kafka."""
        code = """
        import confluent_kafka

        def process():
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumIOViolationType.FORBIDDEN_IMPORT
        assert "confluent_kafka" in violations[0].detail

    def test_from_import_qdrant(self, temp_dir: Path) -> None:
        """Detect from-import of qdrant_client."""
        code = """
        from qdrant_client import QdrantClient

        def process():
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumIOViolationType.FORBIDDEN_IMPORT
        assert "qdrant_client" in violations[0].detail

    def test_import_neo4j(self, temp_dir: Path) -> None:
        """Detect import of neo4j."""
        code = """
        import neo4j

        def process():
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumIOViolationType.FORBIDDEN_IMPORT

    def test_import_asyncpg(self, temp_dir: Path) -> None:
        """Detect import of asyncpg."""
        code = """
        from asyncpg import connect

        def process():
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumIOViolationType.FORBIDDEN_IMPORT

    def test_import_httpx(self, temp_dir: Path) -> None:
        """Detect import of httpx."""
        code = """
        import httpx

        def process():
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumIOViolationType.FORBIDDEN_IMPORT

    def test_import_aiohttp(self, temp_dir: Path) -> None:
        """Detect import of aiohttp."""
        code = """
        from aiohttp import ClientSession

        def process():
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumIOViolationType.FORBIDDEN_IMPORT

    def test_multiple_forbidden_imports(self, temp_dir: Path) -> None:
        """Detect multiple forbidden imports."""
        code = """
        import httpx
        from asyncpg import connect
        from confluent_kafka import Producer

        def process():
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 3
        assert all(
            v.violation_type == EnumIOViolationType.FORBIDDEN_IMPORT for v in violations
        )

    def test_allowed_imports_not_flagged(self, temp_dir: Path) -> None:
        """Standard library and ONEX imports should not be flagged."""
        code = """
        from typing import Any
        from pathlib import Path
        from omnibase_core.nodes import NodeReducer
        import json
        import logging

        def process():
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 0


# =============================================================================
# Detection Tests: os.environ
# =============================================================================


class TestDetectionOsEnviron:
    """Test detection of os.environ and os.getenv usage."""

    def test_os_environ_access(self, temp_dir: Path) -> None:
        """Detect os.environ access."""
        code = """
        import os

        def process():
            value = os.environ["KEY"]
            return value
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumIOViolationType.OS_ENVIRON
        assert "os.environ" in violations[0].detail

    def test_os_environ_get(self, temp_dir: Path) -> None:
        """Detect os.environ.get() call."""
        code = """
        import os

        def process():
            value = os.environ.get("KEY", "default")
            return value
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) >= 1
        assert any(
            v.violation_type == EnumIOViolationType.OS_ENVIRON for v in violations
        )

    def test_os_getenv_call(self, temp_dir: Path) -> None:
        """Detect os.getenv() call."""
        code = """
        import os

        def process():
            value = os.getenv("KEY")
            return value
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) >= 1
        assert any(
            v.violation_type == EnumIOViolationType.OS_ENVIRON for v in violations
        )


# =============================================================================
# Detection Tests: File I/O
# =============================================================================


class TestDetectionFileIO:
    """Test detection of file I/O operations."""

    def test_open_builtin(self, temp_dir: Path) -> None:
        """Detect open() builtin usage."""
        code = """
        def process():
            with open("file.txt") as f:
                return f.read()
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumIOViolationType.FILE_IO
        assert "open()" in violations[0].detail

    def test_path_read_text(self, temp_dir: Path) -> None:
        """Detect Path.read_text() usage."""
        code = """
        from pathlib import Path

        def process():
            content = Path("file.txt").read_text()
            return content
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumIOViolationType.PATH_IO_METHOD
        assert "read_text" in violations[0].detail

    def test_path_write_bytes(self, temp_dir: Path) -> None:
        """Detect Path.write_bytes() usage."""
        code = """
        from pathlib import Path

        def process(data: bytes):
            Path("output.bin").write_bytes(data)
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumIOViolationType.PATH_IO_METHOD
        assert "write_bytes" in violations[0].detail

    def test_path_variable_read_text(self, temp_dir: Path) -> None:
        """Detect Path read_text() on path variable."""
        code = """
        from pathlib import Path

        def process(file_path: Path):
            content = file_path.read_text()
            return content
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumIOViolationType.PATH_IO_METHOD


# =============================================================================
# Exemption Tests
# =============================================================================


class TestExemptionMechanisms:
    """Test exemption mechanisms for I/O audit."""

    def test_onex_exclude_same_line(self, temp_dir: Path) -> None:
        """ONEX_EXCLUDE comment on same line exempts violation."""
        code = """
        import httpx  # ONEX_EXCLUDE: io_audit

        def process():
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 0

    def test_onex_exclude_preceding_line(self, temp_dir: Path) -> None:
        """ONEX_EXCLUDE comment on preceding line exempts code."""
        code = """
        # ONEX_EXCLUDE: io_audit
        import httpx

        def process():
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 0

    def test_onex_exclude_with_reason(self, temp_dir: Path) -> None:
        """ONEX_EXCLUDE with additional reason context works."""
        code = """
        # ONEX_EXCLUDE: io_audit - Required for testing
        import httpx

        def process():
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 0

    def test_unexempted_code_still_detected(self, temp_dir: Path) -> None:
        """Code without exemption still gets violations detected."""
        # ONEX_EXCLUDE applies to the comment line and 5 lines after (total 6 lines)
        # We need the second import to be more than 6 lines from the ONEX_EXCLUDE
        code = """
        # ONEX_EXCLUDE: io_audit
        import httpx  # Line 2 - exempted

        # Line 4
        # Line 5
        # Line 6
        # Line 7 - exemption range ends here
        import asyncpg  # Line 8 - NOT exempted (beyond 6-line range)

        def process():
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        # asyncpg import on line 8 should be detected (beyond 6-line range)
        assert len(violations) == 1
        assert "asyncpg" in violations[0].detail


# =============================================================================
# Node Type Tests
# =============================================================================


class TestNodeTypeFiltering:
    """Test node type-based filtering of audit."""

    def test_reducer_node_audited(self, temp_dir: Path) -> None:
        """REDUCER_GENERIC nodes should be audited."""
        # Create a node structure
        node_dir = temp_dir / "test_reducer"
        node_dir.mkdir()
        _create_contract(node_dir, "REDUCER_GENERIC")

        # File should be audited
        result = should_audit_file(node_dir / "node.py", temp_dir)
        assert result is True

    def test_compute_node_audited(self, temp_dir: Path) -> None:
        """COMPUTE_GENERIC nodes should be audited."""
        node_dir = temp_dir / "test_compute"
        node_dir.mkdir()
        _create_contract(node_dir, "COMPUTE_GENERIC")

        result = should_audit_file(node_dir / "node.py", temp_dir)
        assert result is True

    def test_effect_node_not_audited(self, temp_dir: Path) -> None:
        """EFFECT_GENERIC nodes should NOT be audited (I/O allowed)."""
        node_dir = temp_dir / "test_effect"
        node_dir.mkdir()
        _create_contract(node_dir, "EFFECT_GENERIC")

        result = should_audit_file(node_dir / "node.py", temp_dir)
        assert result is False


# =============================================================================
# Whitelist Tests
# =============================================================================


class TestWhitelistPatterns:
    """Test directory and file whitelist patterns."""

    def test_handlers_directory_skipped(self, temp_dir: Path) -> None:
        """Files in handlers/ directories are skipped."""
        handlers_dir = temp_dir / "handlers"
        handlers_dir.mkdir()
        filepath = handlers_dir / "handler_example.py"

        result = should_audit_file(filepath, temp_dir)
        assert result is False

    def test_adapters_directory_skipped(self, temp_dir: Path) -> None:
        """Files in adapters/ directories are skipped."""
        adapters_dir = temp_dir / "adapters"
        adapters_dir.mkdir()
        filepath = adapters_dir / "adapter_kafka.py"

        result = should_audit_file(filepath, temp_dir)
        assert result is False

    def test_handler_prefix_skipped(self, temp_dir: Path) -> None:
        """Files with handler_ prefix are skipped."""
        filepath = temp_dir / "handler_registration.py"

        result = should_audit_file(filepath, temp_dir)
        assert result is False

    def test_adapter_prefix_skipped(self, temp_dir: Path) -> None:
        """Files with adapter_ prefix are skipped."""
        filepath = temp_dir / "adapter_database.py"

        result = should_audit_file(filepath, temp_dir)
        assert result is False

    def test_wiring_file_skipped(self, temp_dir: Path) -> None:
        """wiring.py files are skipped."""
        filepath = temp_dir / "wiring.py"

        result = should_audit_file(filepath, temp_dir)
        assert result is False

    def test_tests_directory_skipped(self, temp_dir: Path) -> None:
        """Files in tests/ directories are skipped."""
        tests_dir = temp_dir / "tests"
        tests_dir.mkdir()
        filepath = tests_dir / "test_node.py"

        result = should_audit_file(filepath, temp_dir)
        assert result is False


# =============================================================================
# Violation Formatting Tests
# =============================================================================


class TestViolationFormatting:
    """Test violation formatting methods."""

    def test_format_for_ci(self) -> None:
        """format_for_ci produces GitHub Actions annotation."""
        violation = IOAuditViolation(
            file_path=Path("/test/node.py"),
            line_number=10,
            column=5,
            violation_type=EnumIOViolationType.FORBIDDEN_IMPORT,
            detail="Import of 'httpx' - I/O library not allowed",
            context="MyReducer.process",
        )
        output = violation.format_for_ci()

        assert "::error" in output
        assert "file=/test/node.py" in output
        assert "line=10" in output
        assert "col=5" in output
        assert "forbidden_import" in output

    def test_format_human_readable(self) -> None:
        """format_human_readable produces readable output."""
        violation = IOAuditViolation(
            file_path=Path("/test/node.py"),
            line_number=10,
            column=5,
            violation_type=EnumIOViolationType.FORBIDDEN_IMPORT,
            detail="Import of 'httpx' - I/O library not allowed",
            context="MyReducer.process",
        )
        output = violation.format_human_readable()

        assert "/test/node.py:10:5" in output
        assert "forbidden_import" in output
        assert "httpx" in output
        assert "Context: MyReducer.process" in output
        assert "Suggestion:" in output


# =============================================================================
# TYPE_CHECKING Block Tests
# =============================================================================


class TestTypeCheckingBlockHandling:
    """Test handling of TYPE_CHECKING blocks.

    Imports inside TYPE_CHECKING blocks should be excluded from I/O violation
    detection since they are only used for type hints at static analysis time,
    not at runtime. This is a common pattern in Python for avoiding circular
    imports and reducing runtime import overhead.

    Reference: PEP 484, typing.TYPE_CHECKING
    """

    def test_type_checking_import_not_flagged(self, temp_dir: Path) -> None:
        """Direct import inside TYPE_CHECKING block is not flagged."""
        code = """
        from __future__ import annotations

        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            import httpx

        def process() -> None:
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 0, (
            f"TYPE_CHECKING imports should be exempt, found: {violations}"
        )

    def test_type_checking_from_import_not_flagged(self, temp_dir: Path) -> None:
        """From-import inside TYPE_CHECKING block is not flagged."""
        code = """
        from __future__ import annotations

        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            from httpx import AsyncClient, Client

        def process() -> None:
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 0, (
            f"TYPE_CHECKING from-imports should be exempt, found: {violations}"
        )

    def test_typing_module_type_checking_pattern(self, temp_dir: Path) -> None:
        """typing.TYPE_CHECKING pattern (qualified access) is handled."""
        code = """
        from __future__ import annotations

        import typing

        if typing.TYPE_CHECKING:
            import asyncpg
            from qdrant_client import QdrantClient

        def process() -> None:
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 0, (
            f"typing.TYPE_CHECKING imports should be exempt, found: {violations}"
        )

    def test_same_import_outside_type_checking_flagged(self, temp_dir: Path) -> None:
        """Same imports OUTSIDE TYPE_CHECKING block ARE flagged."""
        code = """
        from __future__ import annotations

        import httpx  # NOT inside TYPE_CHECKING - should be flagged

        def process() -> None:
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 1, "Import outside TYPE_CHECKING should be flagged"
        assert violations[0].violation_type == EnumIOViolationType.FORBIDDEN_IMPORT
        assert "httpx" in violations[0].detail

    def test_mixed_imports_inside_and_outside_type_checking(
        self, temp_dir: Path
    ) -> None:
        """Only imports outside TYPE_CHECKING are flagged."""
        code = """
        from __future__ import annotations

        from typing import TYPE_CHECKING
        import asyncpg  # Outside - should be flagged

        if TYPE_CHECKING:
            import httpx  # Inside - should NOT be flagged
            from qdrant_client import QdrantClient  # Inside - should NOT be flagged

        def process() -> None:
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        # Only asyncpg (outside TYPE_CHECKING) should be flagged
        assert len(violations) == 1, (
            f"Expected 1 violation (asyncpg), found: {violations}"
        )
        assert "asyncpg" in violations[0].detail
        assert "httpx" not in violations[0].detail
        assert "qdrant" not in violations[0].detail.lower()

    def test_multiple_type_checking_blocks(self, temp_dir: Path) -> None:
        """Multiple TYPE_CHECKING blocks are all handled correctly."""
        code = """
        from __future__ import annotations

        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            import httpx

        # ... other code ...

        if TYPE_CHECKING:
            from asyncpg import Connection

        def process() -> None:
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 0, (
            f"Multiple TYPE_CHECKING blocks should all be exempt, found: {violations}"
        )

    def test_multiple_forbidden_imports_in_type_checking(self, temp_dir: Path) -> None:
        """Multiple forbidden imports in single TYPE_CHECKING block all exempt."""
        code = """
        from __future__ import annotations

        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            import httpx
            import asyncpg
            from qdrant_client import QdrantClient
            from confluent_kafka import Producer, Consumer
            import neo4j

        def process() -> None:
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 0, (
            f"All TYPE_CHECKING imports should be exempt, found: {violations}"
        )

    def test_type_checking_else_branch_flagged(self, temp_dir: Path) -> None:
        """Imports in else branch of TYPE_CHECKING ARE flagged.

        The else branch of TYPE_CHECKING executes at runtime, so I/O
        imports there should be flagged.
        """
        code = """
        from __future__ import annotations

        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            import httpx  # Type hint only - NOT flagged
        else:
            import asyncpg  # Runtime import - SHOULD be flagged

        def process() -> None:
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        # asyncpg in else branch should be flagged (it runs at runtime)
        assert len(violations) == 1, (
            f"Expected 1 violation (asyncpg in else), found: {violations}"
        )
        assert "asyncpg" in violations[0].detail

    def test_type_checking_with_other_code_in_block(self, temp_dir: Path) -> None:
        """TYPE_CHECKING block with non-import code still handles imports."""
        code = """
        from __future__ import annotations

        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            import httpx

            # Type aliases defined in TYPE_CHECKING block
            HttpClientType = httpx.AsyncClient

        def process() -> None:
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 0, (
            f"TYPE_CHECKING imports with other code should be exempt, found: {violations}"
        )

    def test_negated_type_checking_not_exempt(self, temp_dir: Path) -> None:
        """Negated TYPE_CHECKING (if not TYPE_CHECKING) IS flagged.

        `if not TYPE_CHECKING:` means the code runs at runtime, so
        I/O imports there should be flagged.
        """
        code = """
        from __future__ import annotations

        from typing import TYPE_CHECKING

        if not TYPE_CHECKING:
            import httpx  # Runtime import - SHOULD be flagged

        def process() -> None:
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        # `if not TYPE_CHECKING:` runs at runtime, so should be flagged
        assert len(violations) == 1, (
            f"Negated TYPE_CHECKING imports should be flagged, found: {violations}"
        )
        assert "httpx" in violations[0].detail

    def test_real_world_reducer_pattern(self, temp_dir: Path) -> None:
        """Test realistic ONEX reducer pattern with TYPE_CHECKING imports."""
        code = '''
        """Pure reducer node following ONEX patterns."""
        from __future__ import annotations

        from typing import TYPE_CHECKING

        from omnibase_core.nodes.node_reducer import NodeReducer

        if TYPE_CHECKING:
            from omnibase_core.models.container import ModelONEXContainer
            # These would be violations if outside TYPE_CHECKING
            from httpx import AsyncClient
            from asyncpg import Connection

        class RegistrationReducer(NodeReducer):
            """Reducer that processes registration events.

            Note: Uses TYPE_CHECKING imports for type hints only.
            Actual I/O is delegated to Effect nodes.
            """

            def __init__(self, container: ModelONEXContainer) -> None:
                super().__init__(container)

            def reduce(self, state: dict, event: dict) -> dict:
                """Pure reduction logic - no I/O allowed."""
                return {**state, "processed": True}
        '''
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 0, (
            f"Real-world reducer pattern should have no violations, found: {violations}"
        )

    def test_nested_type_checking_blocks(self, temp_dir: Path) -> None:
        """Nested conditionals inside TYPE_CHECKING are handled correctly.

        This tests that the TYPE_CHECKING exemption properly propagates to nested
        if/elif/else blocks within the TYPE_CHECKING guard. All imports in nested
        conditions inside TYPE_CHECKING should be exempt since the entire outer
        block only executes during static type analysis.
        """
        code = """
        from __future__ import annotations

        import sys
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            # Nested conditional based on Python version
            if sys.version_info >= (3, 11):
                import httpx
                from asyncpg import Connection
            else:
                import aiohttp
                from neo4j import Driver

            # Another nested conditional
            if True:
                from qdrant_client import QdrantClient
                import confluent_kafka

        def process() -> None:
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 0, (
            f"Nested TYPE_CHECKING imports should be exempt: {violations}"
        )

    def test_deeply_nested_type_checking_blocks(self, temp_dir: Path) -> None:
        """Deeply nested conditionals inside TYPE_CHECKING are handled correctly.

        Tests multiple levels of nesting (3+ levels deep) to ensure the exemption
        flag propagates through all nested scopes.
        """
        code = """
        from __future__ import annotations

        import sys
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            if sys.version_info >= (3, 10):
                if sys.platform == "linux":
                    import httpx  # 3 levels deep
                    if True:
                        from asyncpg import Connection  # 4 levels deep

        def process() -> None:
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 0, (
            f"Deeply nested TYPE_CHECKING imports should be exempt: {violations}"
        )

    def test_nested_type_checking_with_else_branches(self, temp_dir: Path) -> None:
        """Nested else branches within TYPE_CHECKING are still exempt.

        Unlike the outer TYPE_CHECKING else branch (which runs at runtime),
        nested else branches INSIDE the TYPE_CHECKING block are still only
        executed during type checking, so they should be exempt.
        """
        code = """
        from __future__ import annotations

        import sys
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            if sys.version_info >= (3, 11):
                import httpx
            else:
                # This else is INSIDE TYPE_CHECKING, so still exempt
                import aiohttp

        def process() -> None:
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 0, (
            f"Nested else within TYPE_CHECKING should be exempt: {violations}"
        )


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_file(self, temp_dir: Path) -> None:
        """Empty file returns no violations."""
        filepath = _create_test_file(temp_dir, "")
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 0

    def test_syntax_error_returns_syntax_error_violation(self, temp_dir: Path) -> None:
        """File with syntax error returns SYNTAX_ERROR violation."""
        code = """
        def broken(
            return None
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 1
        assert violations[0].violation_type == EnumIOViolationType.SYNTAX_ERROR

    def test_clean_reducer_no_violations(self, temp_dir: Path) -> None:
        """Clean reducer with no I/O has no violations."""
        code = '''
        """Pure reducer node."""
        from __future__ import annotations

        from typing import TYPE_CHECKING

        from omnibase_core.nodes.node_reducer import NodeReducer

        if TYPE_CHECKING:
            from omnibase_core.models.container import ModelONEXContainer

        class MyReducer(NodeReducer):
            """Pure reducer with no I/O."""

            def __init__(self, container: ModelONEXContainer) -> None:
                super().__init__(container)

            def reduce(self, state, event):
                """Pure reduction logic."""
                return state
        '''
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        assert len(violations) == 0

    def test_large_file_near_limit_is_scanned(self, temp_dir: Path) -> None:
        """Files just under the 1MB limit are still scanned.

        Validates that files near but under _MAX_FILE_SIZE_BYTES (1MB)
        are processed correctly by audit_all_nodes.
        """
        # Create a file just under the 1MB limit with a violation
        # _MAX_FILE_SIZE_BYTES = 1_000_000 (1MB)
        base_code = '''
"""Large file test."""
import httpx  # This should be detected

def process() -> None:
    pass
'''
        # Pad with comments to reach ~999KB (just under 1MB)
        padding_size = 999_000 - len(base_code)
        padding = "\n# padding" * (padding_size // 10)
        large_code = base_code + padding

        # Verify we're under the limit
        assert len(large_code.encode("utf-8")) < _MAX_FILE_SIZE_BYTES

        filepath = _create_test_file(temp_dir, large_code)
        violations = scan_file_for_io_violations(filepath)

        # The httpx import should still be detected
        assert len(violations) == 1, (
            f"Large file under limit should be scanned: {violations}"
        )
        assert violations[0].violation_type == EnumIOViolationType.FORBIDDEN_IMPORT
        assert "httpx" in violations[0].detail

    def test_large_file_over_limit_is_skipped(self, temp_dir: Path) -> None:
        """Files exceeding the 1MB limit are skipped by audit_all_nodes.

        The audit_all_nodes function should skip files larger than
        _MAX_FILE_SIZE_BYTES to prevent memory issues.
        """
        # Create a file over the 1MB limit
        base_code = '''
"""Over-limit file test."""
import httpx  # Would be a violation if scanned

def process() -> None:
    pass
'''
        # Pad to exceed 1MB
        padding_size = 1_001_000 - len(base_code)
        padding = "\n# padding" * (padding_size // 10)
        large_code = base_code + padding

        # Verify we're over the limit
        assert len(large_code.encode("utf-8")) > _MAX_FILE_SIZE_BYTES

        # Create node directory structure
        node_dir = temp_dir / "test_node"
        node_dir.mkdir()
        _create_contract(node_dir, "REDUCER_GENERIC")
        filepath = node_dir / "node.py"
        filepath.write_text(large_code)

        # audit_all_nodes should skip this file
        violations = audit_all_nodes(temp_dir)

        # No violations should be returned because file was skipped
        assert len(violations) == 0, (
            f"Files over 1MB limit should be skipped, found: {violations}"
        )

    def test_syntax_error_inside_type_checking_block(self, temp_dir: Path) -> None:
        """Syntax error inside TYPE_CHECKING block returns SYNTAX_ERROR violation.

        Even when a syntax error occurs inside a TYPE_CHECKING block,
        the file should be reported as having a syntax error. The AST
        parser fails before the auditor can determine TYPE_CHECKING context.
        """
        code = """
        from __future__ import annotations

        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            import httpx
            # Syntax error: missing closing parenthesis
            def broken_type_hint(x: list[
                pass

        def process() -> None:
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        # Should return a syntax error violation
        assert len(violations) == 1, f"Expected syntax error violation: {violations}"
        assert violations[0].violation_type == EnumIOViolationType.SYNTAX_ERROR
        assert "Syntax error" in violations[0].detail

    def test_syntax_error_only_in_type_checking_content(self, temp_dir: Path) -> None:
        """Syntax error in TYPE_CHECKING block still prevents full audit.

        When the file has a syntax error, Python's ast.parse() fails entirely,
        so no I/O violations (even those outside TYPE_CHECKING) can be detected.
        The auditor correctly reports the syntax error as the primary issue.
        """
        code = """
        from __future__ import annotations

        import httpx  # Would be a violation if file could be parsed

        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            # Malformed code: invalid syntax
            from asyncpg import [Connection

        def process() -> None:
            pass
        """
        filepath = _create_test_file(temp_dir, code)
        violations = scan_file_for_io_violations(filepath)

        # Should return a syntax error, not the httpx import violation
        assert len(violations) == 1
        assert violations[0].violation_type == EnumIOViolationType.SYNTAX_ERROR


# =============================================================================
# CI Gate Test - Main Enforcement Point
# =============================================================================


class TestCIGateIOPurity:
    """CI gate test that enforces I/O purity rules across the codebase.

    This test class contains the main enforcement point that runs in CI on every PR.
    Failures here block merges until violations are addressed.
    """

    def test_codebase_has_no_io_violations_in_pure_nodes(self) -> None:
        """Validate that no I/O violations exist in REDUCER and COMPUTE nodes.

        This is the main CI gate test. It scans all nodes in the codebase
        and fails if any pure nodes have I/O violations.
        """
        # Find the nodes directory
        nodes_dir = (
            Path(__file__).parent.parent.parent / "src" / "omnibase_infra" / "nodes"
        )

        if not nodes_dir.exists():
            pytest.skip(f"Nodes directory not found: {nodes_dir}")

        violations = audit_all_nodes(nodes_dir)

        if violations:
            # Format violations for output
            print("\n" + "=" * 70)
            print("I/O AUDIT VIOLATIONS DETECTED IN PURE NODES")
            print("=" * 70)

            for violation in violations:
                print()
                print(violation.format_human_readable())

            print()
            print("=" * 70)
            print(f"Total violations: {len(violations)}")
            print()
            print("How to fix:")
            print("  1. Move I/O operations to EFFECT nodes")
            print("  2. Inject data via ModelONEXContainer or parameters")
            print("  3. Use ONEX_EXCLUDE: io_audit comment for legitimate exceptions")
            print("=" * 70)

            # Fail the test with a clear message
            pytest.fail(
                f"Found {len(violations)} I/O violation(s) in pure nodes. "
                "See output above for details."
            )


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    "EnumIOViolationType",
    "IOAuditViolation",
    "IOPurityAuditor",
    "audit_all_nodes",
    "get_node_type_from_contract",
    "scan_file_for_io_violations",
    "should_audit_file",
]
