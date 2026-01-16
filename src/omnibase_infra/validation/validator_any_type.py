# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""AST-based Any Type Validator for ONEX Strong Typing Policy.

This module provides static analysis validation to detect improper Any type usage
following the ONEX ADR policy:

- Any is BLOCKED in function signatures (parameters, return types)
- Any is ALLOWED only in Pydantic Field() definitions WITH required NOTE comment
- All other Any usage is BLOCKED

The validator uses Python AST to detect forbidden patterns without runtime execution.

Exemption Mechanisms:
    1. ``@allow_any`` decorator on function/class
    2. ``ONEX_EXCLUDE: any_type`` comment (applies to that line and the next 5 lines)
    3. ``NOTE:`` comment within 5 lines before Any usage (for Field() only)

Limitations:
    This validator uses AST-based static analysis with inherent limitations:

    **Detection Capabilities**:
    - Direct Any type annotations in function parameters and return types
    - Any in Pydantic Field() definitions with or without NOTE comments
    - Any in variable annotations and type aliases
    - Any as generic type argument (e.g., list[Any], dict[str, Any])

    **Known Limitations**:
    - Cannot detect dynamically constructed types or factory patterns
    - Cannot follow imports to resolve types from other modules
    - Cannot detect Any usage via type aliases from external modules
    - String annotations (e.g., ``"Any"``) are not detected

Usage:
    >>> from omnibase_infra.validation.validator_any_type import (
    ...     validate_any_types,
    ...     validate_any_types_ci,
    ... )
    >>> violations = validate_any_types(Path("src/handlers"))
    >>> result = validate_any_types_ci(Path("src/handlers"))
    >>> if not result.passed:
    ...     print(f"Found {result.blocking_count} blocking violations")
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from omnibase_infra.enums import EnumAnyTypeViolation, EnumValidationSeverity
from omnibase_infra.models.validation.model_any_type_validation_result import (
    ModelAnyTypeValidationResult,
)
from omnibase_infra.models.validation.model_any_type_violation import (
    ModelAnyTypeViolation,
)
from omnibase_infra.types import ASTFunctionDef
from omnibase_infra.validation.mixin_any_type_classification import (
    MixinAnyTypeClassification,
)
from omnibase_infra.validation.mixin_any_type_exemption import MixinAnyTypeExemption
from omnibase_infra.validation.mixin_any_type_reporting import MixinAnyTypeReporting

logger = logging.getLogger(__name__)

# Comment patterns for exemptions
_ONEX_EXCLUDE_PATTERN = "ONEX_EXCLUDE:"
_ONEX_EXCLUDE_ANY_TYPE = "any_type"

# Maximum file size to process (in bytes).
# Files larger than this are skipped to prevent hangs on auto-generated
# or minified code. 1MB is sufficient for any reasonable hand-written Python.
_MAX_FILE_SIZE_BYTES: int = 1_000_000  # 1MB


class AnyTypeDetector(
    MixinAnyTypeClassification,
    MixinAnyTypeExemption,
    MixinAnyTypeReporting,
    ast.NodeVisitor,
):
    """AST visitor to detect Any type usage violations.

    This visitor walks the AST and identifies Any type annotations that
    violate the ONEX strong typing policy. It supports exemption via
    decorators and comments.

    The class is composed of mixins to keep method count manageable:
    - MixinAnyTypeClassification: Type classification helpers
    - MixinAnyTypeExemption: Exemption management (decorators, NOTE comments)
    - MixinAnyTypeReporting: Violation reporting

    Attributes:
        filepath: Path to the file being analyzed.
        source_lines: List of source code lines for extracting snippets.
        violations: List of detected violations.
        allowed_lines: Set of line numbers exempted via decorator or comment.
        in_field_context: Flag indicating we're inside a Field() call.
        current_function: Name of the current function being visited.
        current_class: Name of the current class being visited.
        has_file_level_note: Whether file has NOTE comment near Any import.
    """

    def __init__(self, filepath: str, source_lines: list[str]) -> None:
        """Initialize the detector.

        Args:
            filepath: Path to the file being analyzed.
            source_lines: List of source code lines for snippet extraction.
        """
        self.filepath = filepath
        self.source_lines = source_lines
        self.violations: list[ModelAnyTypeViolation] = []
        self.allowed_lines: set[int] = set()
        self.in_field_context: bool = False
        self.current_function: str = ""
        self.current_class: str = ""
        self.has_file_level_note: bool = self._check_file_level_note()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class definition.

        Checks for @allow_any decorator and processes class body.
        """
        if self._has_allow_decorator(node.decorator_list):
            self._allow_all_lines_in_node(node)

        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a function definition."""
        self._check_function_def(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit an async function definition."""
        self._check_function_def(node)

    def _check_function_def(self, node: ASTFunctionDef) -> None:
        """Check a function definition for Any type violations.

        Checks:
        - @allow_any decorator
        - Parameter type annotations
        - Return type annotation

        Args:
            node: The function definition AST node.
        """
        if self._has_allow_decorator(node.decorator_list):
            self._allow_all_lines_in_node(node)
            self.generic_visit(node)
            return

        old_function = self.current_function
        self.current_function = node.name

        # Check parameter annotations
        for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            if arg.annotation is not None:
                self._check_annotation_for_any(
                    arg.annotation,
                    EnumAnyTypeViolation.FUNCTION_PARAMETER,
                    context_name=f"{self.current_function}({arg.arg})",
                )

        # Check *args and **kwargs
        if node.args.vararg and node.args.vararg.annotation:
            self._check_annotation_for_any(
                node.args.vararg.annotation,
                EnumAnyTypeViolation.FUNCTION_PARAMETER,
                context_name=f"{self.current_function}(*{node.args.vararg.arg})",
            )
        if node.args.kwarg and node.args.kwarg.annotation:
            self._check_annotation_for_any(
                node.args.kwarg.annotation,
                EnumAnyTypeViolation.FUNCTION_PARAMETER,
                context_name=f"{self.current_function}(**{node.args.kwarg.arg})",
            )

        # Check return type annotation
        if node.returns is not None:
            self._check_annotation_for_any(
                node.returns,
                EnumAnyTypeViolation.RETURN_TYPE,
                context_name=self.current_function,
            )

        self.generic_visit(node)
        self.current_function = old_function

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Visit an annotated assignment (type annotation).

        Handles:
        - Variable annotations: ``x: Any = value``
        - Class attributes: ``attr: Any``
        - Type aliases: ``TypeAlias = Any`` (via TypeAlias annotation)
        - Pydantic Field() with Any (requires NOTE comment)
        """
        if node.lineno in self.allowed_lines:
            self.generic_visit(node)
            return

        # Determine context name
        context_name = ""
        if isinstance(node.target, ast.Name):
            context_name = node.target.id

        # Check if this is a Pydantic Field() with potential NOTE comment
        is_field_with_note = self._is_field_with_note(node)

        if is_field_with_note:
            # Field() with NOTE comment is allowed - skip
            self.generic_visit(node)
            return

        # Check if this is a Field() call without proper NOTE
        # This is a special case - report FIELD_MISSING_NOTE instead of other types
        is_field_without_note = (
            node.value is not None
            and self._is_field_call(node.value)
            and self._contains_any(node.annotation)
        )

        if is_field_without_note:
            # Find accurate column for Any token, fall back to node start
            any_col = self._find_any_col(node.annotation)
            self._add_violation(
                node.lineno,
                any_col if any_col is not None else node.col_offset,
                EnumAnyTypeViolation.FIELD_MISSING_NOTE,
                context_name=context_name,
            )
            self.generic_visit(node)
            return

        # Check if this is a type alias
        is_type_alias = self._is_type_alias_annotation(node)

        # For type aliases, check the VALUE for Any (e.g., ConfigType: TypeAlias = dict[str, Any])
        if is_type_alias and node.value is not None:
            if self._contains_any(node.value):
                # Find accurate column for Any token, fall back to node start
                any_col = self._find_any_col(node.value)
                self._add_violation(
                    node.lineno,
                    any_col if any_col is not None else node.col_offset,
                    EnumAnyTypeViolation.TYPE_ALIAS,
                    context_name=context_name,
                )
            self.generic_visit(node)
            return

        # Determine violation type for non-type-alias cases
        if self.current_class and not self.current_function:
            # Class attribute (not in a method)
            violation_type = EnumAnyTypeViolation.CLASS_ATTRIBUTE
        else:
            violation_type = EnumAnyTypeViolation.VARIABLE_ANNOTATION

        # Check the annotation
        self._check_annotation_for_any(
            node.annotation,
            violation_type,
            context_name=context_name,
        )

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit a regular assignment to detect type alias patterns.

        Handles patterns like: ``JsonType = dict[str, Any]``
        """
        if node.lineno in self.allowed_lines:
            self.generic_visit(node)
            return

        # Check if this looks like a type alias (PascalCase name or ends with Type)
        for target in node.targets:
            if isinstance(target, ast.Name):
                name = target.id
                # Type aliases typically use PascalCase or end with "Type"
                if self._is_likely_type_alias_name(name):
                    # Check if the value contains Any
                    if self._contains_any(node.value):
                        # Find accurate column for Any token, fall back to node start
                        any_col = self._find_any_col(node.value)
                        self._add_violation(
                            node.lineno,
                            any_col if any_col is not None else node.col_offset,
                            EnumAnyTypeViolation.TYPE_ALIAS,
                            context_name=name,
                        )

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Visit a subscript expression to detect Any in generic arguments.

        Handles patterns like: list[Any], dict[str, Any], Optional[Any]
        """
        # This is handled within _check_annotation_for_any via recursion
        self.generic_visit(node)

    def _check_annotation_for_any(
        self,
        annotation: ast.expr,
        violation_type: EnumAnyTypeViolation,
        context_name: str = "",
    ) -> None:
        """Check an annotation expression for Any usage.

        Recursively checks the annotation for Any type references,
        including in generic type arguments.

        Args:
            annotation: The AST expression representing the annotation.
            violation_type: The type of violation to report if Any is found.
            context_name: Context identifier (function/variable name).
        """
        if annotation.lineno in self.allowed_lines:
            return

        # Direct Any reference
        if isinstance(annotation, ast.Name) and annotation.id == "Any":
            self._add_violation(
                annotation.lineno,
                annotation.col_offset,
                violation_type,
                context_name=context_name,
            )
            return

        # Attribute access (typing.Any)
        if isinstance(annotation, ast.Attribute) and annotation.attr == "Any":
            self._add_violation(
                annotation.lineno,
                annotation.col_offset,
                violation_type,
                context_name=context_name,
            )
            return

        # Generic subscript (list[Any], dict[str, Any], Optional[Any], Union[..., Any])
        if isinstance(annotation, ast.Subscript):
            self._check_subscript_for_any(annotation, context_name)
            return

        # BinOp for union syntax (X | Any)
        if isinstance(annotation, ast.BinOp):
            self._check_annotation_for_any(
                annotation.left,
                EnumAnyTypeViolation.GENERIC_ARGUMENT,
                context_name=context_name,
            )
            self._check_annotation_for_any(
                annotation.right,
                EnumAnyTypeViolation.GENERIC_ARGUMENT,
                context_name=context_name,
            )

    def _check_subscript_for_any(
        self,
        node: ast.Subscript,
        context_name: str,
    ) -> None:
        """Check a subscript expression for Any in type arguments.

        Args:
            node: The subscript AST node.
            context_name: Context identifier (function/variable name).
        """
        # Check the slice (type arguments)
        slice_node = node.slice

        if isinstance(slice_node, ast.Tuple):
            # Multiple type arguments: dict[str, Any]
            for elt in slice_node.elts:
                self._check_annotation_for_any(
                    elt,
                    EnumAnyTypeViolation.GENERIC_ARGUMENT,
                    context_name=context_name,
                )
        else:
            # Single type argument: list[Any], Optional[Any]
            self._check_annotation_for_any(
                slice_node,
                EnumAnyTypeViolation.GENERIC_ARGUMENT,
                context_name=context_name,
            )

    def _contains_any(self, node: ast.expr) -> bool:
        """Check if an annotation contains Any type.

        Args:
            node: The AST expression to check.

        Returns:
            True if Any is found anywhere in the annotation.
        """
        if isinstance(node, ast.Name) and node.id == "Any":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "Any":
            return True
        if isinstance(node, ast.Subscript):
            slice_node = node.slice
            if isinstance(slice_node, ast.Tuple):
                return any(self._contains_any(elt) for elt in slice_node.elts)
            return self._contains_any(slice_node)
        if isinstance(node, ast.BinOp):
            return self._contains_any(node.left) or self._contains_any(node.right)
        return False

    def _find_any_col(self, node: ast.expr) -> int | None:
        """Find the column offset of the first Any token in an expression.

        Walks the annotation/value AST recursively to find an ast.Name node
        with id == "Any" and returns its col_offset. This provides accurate
        column reporting for CI annotations.

        Args:
            node: The AST expression to search.

        Returns:
            The column offset of the Any token, or None if not found.
        """
        if isinstance(node, ast.Name) and node.id == "Any":
            return node.col_offset
        if isinstance(node, ast.Attribute) and node.attr == "Any":
            return node.col_offset
        if isinstance(node, ast.Subscript):
            slice_node = node.slice
            if isinstance(slice_node, ast.Tuple):
                for elt in slice_node.elts:
                    result = self._find_any_col(elt)
                    if result is not None:
                        return result
            else:
                return self._find_any_col(slice_node)
        if isinstance(node, ast.BinOp):
            left_result = self._find_any_col(node.left)
            if left_result is not None:
                return left_result
            return self._find_any_col(node.right)
        return None


def _find_onex_exclude_lines(content: str) -> set[int]:
    """Find lines exempted via ONEX_EXCLUDE: any_type comments.

    The exemption applies to the comment line and the next 5 lines (to cover
    multi-line signatures such as function definitions that span multiple lines).

    Args:
        content: Source file content.

    Returns:
        Set of line numbers that are exempted.
    """
    excluded_lines: set[int] = set()
    lines = content.split("\n")

    for i, line in enumerate(lines, start=1):
        if _ONEX_EXCLUDE_PATTERN in line and _ONEX_EXCLUDE_ANY_TYPE in line:
            # Exclude this line and the next 5 lines
            for offset in range(6):
                excluded_lines.add(i + offset)

    return excluded_lines


def validate_any_types_in_file(filepath: Path) -> list[ModelAnyTypeViolation]:
    """Validate a single Python file for Any type violations.

    Args:
        filepath: Path to the Python file to validate.

    Returns:
        List of detected violations. Empty list if no violations found.
        For syntax errors, returns a single SYNTAX_ERROR violation.
    """
    try:
        content = filepath.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning(
            "Failed to read file",
            extra={"file": str(filepath), "error": str(e)},
        )
        return []

    # Find lines excluded via ONEX_EXCLUDE comments
    excluded_lines = _find_onex_exclude_lines(content)

    try:
        tree = ast.parse(content, filename=str(filepath))
    except SyntaxError as e:
        logger.warning(
            "Syntax error in file",
            extra={"file": str(filepath), "error": str(e)},
        )
        return [
            ModelAnyTypeViolation(
                file_path=filepath,
                line_number=e.lineno or 1,
                column=0 if e.offset is None else e.offset,
                violation_type=EnumAnyTypeViolation.SYNTAX_ERROR,
                code_snippet=f"Syntax error: {e.msg}",
                suggestion=EnumAnyTypeViolation.SYNTAX_ERROR.suggestion,
                severity=EnumValidationSeverity.ERROR,
                context_name="",
            )
        ]

    source_lines = content.split("\n")
    detector = AnyTypeDetector(str(filepath.resolve()), source_lines)
    detector.allowed_lines.update(excluded_lines)
    detector.visit(tree)

    return detector.violations


# Directories to skip for Any type validation (exact name matching).
# Uses frozenset for O(1) lookup performance.
# NOTE: "tests" is included because test files are allowed to use Any types.
# This differs from the general skip directories in infra_validators.py.
_ANY_TYPE_SKIP_DIRECTORIES: frozenset[str] = frozenset(
    {
        "tests",  # Test files are allowed to use Any types
        "archive",  # Historical code not subject to validation
        "archived",  # Alternative naming for archived code
        "__pycache__",  # Python bytecode cache
    }
)


def _should_skip_file(filepath: Path) -> bool:
    """Check if a file should be skipped based on exact directory name matching.

    Uses exact parent directory matching (not substring) to prevent false positives.
    For example, '/my_tests/foo.py' is NOT skipped because 'my_tests' != 'tests'.

    Matching behavior:
    - Only parent directories are checked (filenames are NOT checked against skip list)
    - Matching is case-sensitive (Linux standard)
    - A path is skipped if ANY parent directory matches exactly

    Args:
        filepath: Path to check.

    Returns:
        True if the file should be skipped.

    Examples:
        Paths that ARE skipped (exact directory match):
        >>> _should_skip_file(Path("src/tests/foo.py"))
        True
        >>> _should_skip_file(Path("src/archive/foo.py"))
        True

        Paths that are NOT skipped (no false positives):
        >>> _should_skip_file(Path("src/my_tests/foo.py"))
        False
        >>> _should_skip_file(Path("src/testing_utils/foo.py"))
        False
    """
    parts = filepath.parts

    # Check parent directories for exact matches (exclude filename at parts[-1])
    for part in parts[:-1]:
        if part in _ANY_TYPE_SKIP_DIRECTORIES:
            return True

    # Special case: skip files in scripts/validation/ nested path.
    # This specifically targets validation test scripts, not the validation module itself.
    # Checks for exact sequence: "scripts" followed immediately by "validation".
    for i, part in enumerate(parts[:-1]):
        if (
            part == "scripts"
            and i + 1 < len(parts) - 1
            and parts[i + 1] == "validation"
        ):
            return True

    # Skip files starting with underscore (test fixtures, private modules)
    if filepath.name.startswith("_"):
        return True

    return False


def validate_any_types(
    directory: Path, recursive: bool = True
) -> list[ModelAnyTypeViolation]:
    """Validate all Python files in a directory for Any type violations.

    This is the main entry point for batch validation.

    Args:
        directory: Path to the directory to validate.
        recursive: If True, recursively validate subdirectories.

    Returns:
        List of all detected violations across all files.

    Example:
        >>> violations = validate_any_types(Path("src/handlers"))
        >>> for v in violations:
        ...     print(v.format_human_readable())
    """
    violations: list[ModelAnyTypeViolation] = []
    pattern = "**/*.py" if recursive else "*.py"

    for filepath in directory.glob(pattern):
        if filepath.is_file() and not _should_skip_file(filepath):
            # Skip very large files to prevent hangs on auto-generated code
            try:
                file_size = filepath.stat().st_size
                if file_size > _MAX_FILE_SIZE_BYTES:
                    logger.debug(
                        "Skipping large file",
                        extra={"file": str(filepath), "size_bytes": file_size},
                    )
                    continue
            except OSError as e:
                logger.warning(
                    "Failed to stat file",
                    extra={"file": str(filepath), "error": str(e)},
                )
                continue

            try:
                file_violations = validate_any_types_in_file(filepath)
                violations.extend(file_violations)
            except Exception as e:  # catch-all-ok: validation continues on file errors
                logger.warning(
                    "Failed to validate file",
                    extra={
                        "file": str(filepath),
                        "error_type": type(e).__name__,
                        "error": str(e),
                    },
                )
                continue

    return violations


def validate_any_types_ci(
    directory: Path,
    recursive: bool = True,
) -> ModelAnyTypeValidationResult:
    """CI gate for Any type validation.

    This function is designed for CI pipeline integration. It returns a
    structured result model containing the pass/fail status and all violations
    for reporting.

    Args:
        directory: Path to the directory to validate.
        recursive: If True, recursively validate subdirectories.

    Returns:
        ModelAnyTypeValidationResult containing pass/fail status and violations.

    Example:
        >>> result = validate_any_types_ci(Path("src/handlers"))
        >>> if not result.passed:
        ...     for line in result.format_for_ci():
        ...         print(line)
        ...     sys.exit(1)
    """
    # Count files (excluding skipped patterns and large files)
    pattern = "**/*.py" if recursive else "*.py"
    files_checked = sum(
        1
        for f in directory.glob(pattern)
        if f.is_file()
        and not _should_skip_file(f)
        and f.stat().st_size <= _MAX_FILE_SIZE_BYTES
    )

    violations = validate_any_types(directory, recursive=recursive)
    return ModelAnyTypeValidationResult.from_violations(violations, files_checked)


def _print_fix_instructions() -> None:
    """Print instructions for fixing Any type violations."""
    print("\nHow to fix Any type violations:")
    print("   1. Replace Any with specific types (object, Union, Protocol)")
    print("   2. For Pydantic models with JSON data, add NOTE comment:")
    print("      field: Any = Field(...)  # NOTE: Using Any for JSON payload")
    print("   3. Use @allow_any decorator if exemption is truly needed")
    print("   4. Add ONEX_EXCLUDE: any_type comment for legacy code")
    print("\n   Example fixes:")
    print("   BAD:  def process(data: Any) -> Any:")
    print("   GOOD: def process(data: object) -> ModelResult:")
    print("   GOOD: def process(data: JsonType) -> ModelResult:")


# Module-level singleton validator is not needed for this validator
# since validation is stateless (each file is parsed independently)

__all__ = [
    "AnyTypeDetector",
    "ModelAnyTypeValidationResult",
    "ModelAnyTypeViolation",
    "validate_any_types",
    "validate_any_types_ci",
    "validate_any_types_in_file",
]
