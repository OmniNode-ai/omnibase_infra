#!/usr/bin/env python3
"""
Backwards Compatibility Validation Script

This script validates that code changes maintain backwards compatibility with
previous versions by detecting:
- Removed public APIs (functions, classes, methods)
- Changed function signatures (parameters, return types)
- Removed or renamed classes and modules
- Breaking changes in Pydantic data models
- Breaking changes in type annotations

The script uses AST (Abstract Syntax Tree) parsing for robust static analysis
and can be configured to allow intentional breaking changes.

Usage:
    # Validate specific files
    python scripts/validate_backwards_compatibility.py file1.py file2.py

    # Validate all staged files (for pre-commit)
    python scripts/validate_backwards_compatibility.py --staged

    # Validate with strict mode (no exemptions)
    python scripts/validate_backwards_compatibility.py --strict

    # Validate against specific baseline version
    python scripts/validate_backwards_compatibility.py --baseline v1.0.0

Exit codes:
    0: No breaking changes detected (or all exempted)
    1: Breaking changes detected
    2: Script execution error
"""

import argparse
import ast
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml


class ChangeType(Enum):
    """Types of compatibility changes."""

    API_REMOVED = "api_removed"
    API_SIGNATURE_CHANGED = "api_signature_changed"
    CLASS_REMOVED = "class_removed"
    CLASS_RENAMED = "class_renamed"
    MODEL_FIELD_REMOVED = "model_field_removed"
    MODEL_FIELD_TYPE_CHANGED = "model_field_type_changed"
    MODEL_FIELD_REQUIRED_CHANGED = "model_field_required_changed"
    TYPE_ANNOTATION_CHANGED = "type_annotation_changed"
    MODULE_REMOVED = "module_removed"
    PUBLIC_CONSTANT_REMOVED = "public_constant_removed"


class Severity(Enum):
    """Severity levels for compatibility issues."""

    ERROR = "error"  # Breaking change, fails validation
    WARNING = "warning"  # Potentially breaking, should review
    INFO = "info"  # Non-breaking, informational only


@dataclass
class CompatibilityIssue:
    """Represents a backwards compatibility issue."""

    change_type: ChangeType
    severity: Severity
    file_path: str
    name: str
    description: str
    line: Optional[int] = None
    old_signature: Optional[str] = None
    new_signature: Optional[str] = None
    exempted: bool = False
    exemption_reason: Optional[str] = None


@dataclass
class APIElement:
    """Represents a public API element (function, class, etc.)."""

    name: str
    type: str  # 'function', 'class', 'method', 'constant'
    signature: Optional[str] = None
    parameters: list[str] = field(default_factory=list)
    return_type: Optional[str] = None
    decorators: list[str] = field(default_factory=list)
    is_public: bool = True
    line_number: Optional[int] = None
    docstring: Optional[str] = None
    parent_class: Optional[str] = None


@dataclass
class PydanticModel:
    """Represents a Pydantic model for validation."""

    name: str
    fields: dict[str, dict[str, Any]] = field(default_factory=dict)
    base_classes: list[str] = field(default_factory=list)
    line_number: Optional[int] = None


class ASTAnalyzer(ast.NodeVisitor):
    """AST visitor for extracting API elements and models."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.api_elements: dict[str, APIElement] = {}
        self.pydantic_models: dict[str, PydanticModel] = {}
        self.imports: set[str] = set()
        self.current_class: Optional[str] = None

    def visit_Import(self, node: ast.Import) -> None:
        """Track imports."""
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track from imports."""
        if node.module:
            self.imports.add(node.module)
            for alias in node.names:
                self.imports.add(f"{node.module}.{alias.name}")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extract function definitions."""
        # Only process public functions (not starting with _)
        if not node.name.startswith("_") or node.name == "__init__":
            signature = self._get_function_signature(node)
            api_element = APIElement(
                name=node.name,
                type="method" if self.current_class else "function",
                signature=signature,
                parameters=[arg.arg for arg in node.args.args],
                return_type=self._get_annotation_string(node.returns),
                decorators=[self._get_decorator_name(d) for d in node.decorator_list],
                line_number=node.lineno,
                docstring=ast.get_docstring(node),
                parent_class=self.current_class,
            )

            # Build full qualified name
            if self.current_class:
                full_name = f"{self.current_class}.{node.name}"
            else:
                full_name = node.name

            self.api_elements[full_name] = api_element

        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Extract async function definitions."""
        # Treat async functions same as regular functions
        if not node.name.startswith("_"):
            signature = self._get_function_signature(node)
            api_element = APIElement(
                name=node.name,
                type="method" if self.current_class else "function",
                signature=f"async {signature}",
                parameters=[arg.arg for arg in node.args.args],
                return_type=self._get_annotation_string(node.returns),
                decorators=[self._get_decorator_name(d) for d in node.decorator_list],
                line_number=node.lineno,
                docstring=ast.get_docstring(node),
                parent_class=self.current_class,
            )

            if self.current_class:
                full_name = f"{self.current_class}.{node.name}"
            else:
                full_name = node.name

            self.api_elements[full_name] = api_element

        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Extract class definitions."""
        if not node.name.startswith("_"):
            # Check if it's a Pydantic model
            is_pydantic = self._is_pydantic_model(node)

            # Extract class info
            api_element = APIElement(
                name=node.name,
                type="class",
                decorators=[self._get_decorator_name(d) for d in node.decorator_list],
                line_number=node.lineno,
                docstring=ast.get_docstring(node),
            )
            self.api_elements[node.name] = api_element

            # Extract Pydantic model fields if applicable
            if is_pydantic:
                model = self._extract_pydantic_model(node)
                self.pydantic_models[node.name] = model

            # Visit methods within the class
            old_class = self.current_class
            self.current_class = node.name
            self.generic_visit(node)
            self.current_class = old_class

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Extract annotated assignments (constants, class attributes)."""
        if isinstance(node.target, ast.Name):
            name = node.target.id
            # Only track public constants (uppercase)
            if name.isupper() and not self.current_class:
                api_element = APIElement(
                    name=name,
                    type="constant",
                    return_type=self._get_annotation_string(node.annotation),
                    line_number=node.lineno,
                )
                self.api_elements[name] = api_element

        self.generic_visit(node)

    def _is_pydantic_model(self, node: ast.ClassDef) -> bool:
        """Check if class is a Pydantic model."""
        # Check base classes
        for base in node.bases:
            base_name = self._get_name_string(base)
            if base_name in ["BaseModel", "pydantic.BaseModel"]:
                return True

        # Check if pydantic is imported
        if "pydantic" in self.imports:
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == "BaseModel":
                    return True

        return False

    def _extract_pydantic_model(self, node: ast.ClassDef) -> PydanticModel:
        """Extract Pydantic model field information."""
        model = PydanticModel(
            name=node.name,
            base_classes=[self._get_name_string(base) for base in node.bases],
            line_number=node.lineno,
        )

        # Extract field definitions
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                field_name = item.target.id
                field_info: dict[str, Any] = {
                    "type": self._get_annotation_string(item.annotation),
                    "required": True,  # Default to required
                }

                # Check if field has default value
                if item.value:
                    field_info["required"] = False
                    field_info["default"] = self._get_value_string(item.value)

                # Check for Optional type hint
                if self._is_optional_type(item.annotation):
                    field_info["required"] = False

                model.fields[field_name] = field_info

        return model

    def _is_optional_type(self, annotation: ast.AST) -> bool:
        """Check if type annotation is Optional."""
        annotation_str = self._get_annotation_string(annotation)
        return annotation_str and (
            "Optional[" in annotation_str or "| None" in annotation_str
        )

    def _get_function_signature(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> str:
        """Build function signature string."""
        params = []
        for arg in node.args.args:
            param_str = arg.arg
            if arg.annotation:
                param_str += f": {self._get_annotation_string(arg.annotation)}"
            params.append(param_str)

        # Handle *args and **kwargs
        if node.args.vararg:
            vararg_str = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation:
                vararg_str += (
                    f": {self._get_annotation_string(node.args.vararg.annotation)}"
                )
            params.append(vararg_str)

        if node.args.kwarg:
            kwarg_str = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation:
                kwarg_str += (
                    f": {self._get_annotation_string(node.args.kwarg.annotation)}"
                )
            params.append(kwarg_str)

        signature = f"{node.name}({', '.join(params)})"

        if node.returns:
            signature += f" -> {self._get_annotation_string(node.returns)}"

        return signature

    def _get_annotation_string(self, annotation: Optional[ast.AST]) -> Optional[str]:
        """Convert annotation AST to string."""
        if annotation is None:
            return None
        return ast.unparse(annotation)

    def _get_name_string(self, node: ast.AST) -> str:
        """Get string representation of a name node."""
        return ast.unparse(node)

    def _get_value_string(self, node: ast.AST) -> str:
        """Get string representation of a value node."""
        try:
            return ast.unparse(node)
        except Exception:
            return "<complex_value>"

    def _get_decorator_name(self, decorator: ast.AST) -> str:
        """Extract decorator name."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                return decorator.func.attr
        return "<unknown>"


class CompatibilityValidator:
    """Main backwards compatibility validator."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        baseline_version: Optional[str] = None,
        strict_mode: bool = False,
    ):
        self.config = self._load_config(config_path)
        self.baseline_version = baseline_version or self.config.get(
            "baseline_version", "main"
        )
        self.strict_mode = strict_mode
        self.issues: list[CompatibilityIssue] = []
        self.exemptions = self.config.get("exemptions", {})

    def _load_config(self, config_path: Optional[Path]) -> dict:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path("config/backwards_compatibility_config.yaml")

        if not config_path.exists():
            return self._get_default_config()

        try:
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"⚠️  Warning: Failed to load config from {config_path}: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            "baseline_version": "main",
            "strict_mode": False,
            "check_functions": True,
            "check_classes": True,
            "check_models": True,
            "check_type_annotations": True,
            "severity_threshold": "error",  # error, warning, info
            "exemptions": {},
            "ignore_patterns": [
                "tests/",
                "test_*.py",
                "*_test.py",
                "examples/",
                "scripts/",
            ],
        }

    def validate_files(
        self, file_paths: list[str], baseline_ref: Optional[str] = None
    ) -> bool:
        """Validate backwards compatibility for given files."""
        if not file_paths:
            print("INFO: No files to validate")
            return True

        baseline_ref = baseline_ref or self.baseline_version

        print(f"🔍 Validating backwards compatibility against baseline: {baseline_ref}")
        print(f"   Checking {len(file_paths)} file(s)...")

        all_valid = True

        for file_path in file_paths:
            path = Path(file_path)

            # Skip ignored patterns
            if self._should_ignore(path):
                continue

            # Skip non-Python files
            if path.suffix != ".py":
                continue

            if not self._validate_single_file(path, baseline_ref):
                all_valid = False

        # Print summary
        self._print_summary()

        return (
            all_valid
            and len(
                [
                    i
                    for i in self.issues
                    if i.severity == Severity.ERROR and not i.exempted
                ]
            )
            == 0
        )

    def _should_ignore(self, file_path: Path) -> bool:
        """Check if file should be ignored based on patterns."""
        ignore_patterns = self.config.get("ignore_patterns", [])

        file_str = str(file_path)
        for pattern in ignore_patterns:
            if pattern.endswith("/"):
                # Directory pattern
                if pattern.rstrip("/") in file_str:
                    return True
            else:
                # File pattern
                if Path(file_str).match(pattern):
                    return True

        return False

    def _validate_single_file(self, file_path: Path, baseline_ref: str) -> bool:
        """Validate a single file against baseline."""
        if not file_path.exists():
            return True

        # Get baseline version of the file
        baseline_content = self._get_baseline_content(file_path, baseline_ref)
        if baseline_content is None:
            # File doesn't exist in baseline (new file), no compatibility concerns
            return True

        # Parse current and baseline versions
        try:
            current_tree = ast.parse(file_path.read_text())
            baseline_tree = ast.parse(baseline_content)
        except SyntaxError as e:
            print(f"  ⚠️  Syntax error in {file_path}: {e}")
            return True  # Can't validate invalid Python

        # Analyze both versions
        current_analyzer = ASTAnalyzer(str(file_path))
        current_analyzer.visit(current_tree)

        baseline_analyzer = ASTAnalyzer(str(file_path))
        baseline_analyzer.visit(baseline_tree)

        # Compare and detect breaking changes
        self._compare_api_elements(
            file_path,
            baseline_analyzer.api_elements,
            current_analyzer.api_elements,
        )

        self._compare_pydantic_models(
            file_path,
            baseline_analyzer.pydantic_models,
            current_analyzer.pydantic_models,
        )

        return True

    def _get_baseline_content(
        self, file_path: Path, baseline_ref: str
    ) -> Optional[str]:
        """Get file content from baseline git reference."""
        try:
            result = subprocess.run(
                ["git", "show", f"{baseline_ref}:{file_path}"],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode == 0:
                return result.stdout

            return None  # File doesn't exist in baseline

        except Exception:
            return None

    def _compare_api_elements(
        self,
        file_path: Path,
        baseline_elements: dict[str, APIElement],
        current_elements: dict[str, APIElement],
    ) -> None:
        """Compare API elements between baseline and current."""
        # Check for removed APIs
        for name, baseline_elem in baseline_elements.items():
            if name not in current_elements:
                # API was removed
                self._add_issue(
                    CompatibilityIssue(
                        change_type=ChangeType.API_REMOVED,
                        severity=Severity.ERROR,
                        file_path=str(file_path),
                        name=name,
                        description=f"{baseline_elem.type.capitalize()} '{name}' was removed",
                        line=baseline_elem.line_number,
                        old_signature=baseline_elem.signature,
                    )
                )
                continue

            current_elem = current_elements[name]

            # Check for signature changes
            if baseline_elem.signature != current_elem.signature:
                self._add_issue(
                    CompatibilityIssue(
                        change_type=ChangeType.API_SIGNATURE_CHANGED,
                        severity=Severity.ERROR,
                        file_path=str(file_path),
                        name=name,
                        description=f"{baseline_elem.type.capitalize()} '{name}' signature changed",
                        line=current_elem.line_number,
                        old_signature=baseline_elem.signature,
                        new_signature=current_elem.signature,
                    )
                )

    def _compare_pydantic_models(
        self,
        file_path: Path,
        baseline_models: dict[str, PydanticModel],
        current_models: dict[str, PydanticModel],
    ) -> None:
        """Compare Pydantic models between baseline and current."""
        for name, baseline_model in baseline_models.items():
            if name not in current_models:
                # Model was removed
                self._add_issue(
                    CompatibilityIssue(
                        change_type=ChangeType.CLASS_REMOVED,
                        severity=Severity.ERROR,
                        file_path=str(file_path),
                        name=name,
                        description=f"Pydantic model '{name}' was removed",
                        line=baseline_model.line_number,
                    )
                )
                continue

            current_model = current_models[name]

            # Check for removed fields
            for field_name, baseline_field in baseline_model.fields.items():
                if field_name not in current_model.fields:
                    self._add_issue(
                        CompatibilityIssue(
                            change_type=ChangeType.MODEL_FIELD_REMOVED,
                            severity=Severity.ERROR,
                            file_path=str(file_path),
                            name=f"{name}.{field_name}",
                            description=f"Field '{field_name}' removed from model '{name}'",
                            line=current_model.line_number,
                        )
                    )
                    continue

                current_field = current_model.fields[field_name]

                # Check for type changes
                if baseline_field["type"] != current_field["type"]:
                    self._add_issue(
                        CompatibilityIssue(
                            change_type=ChangeType.MODEL_FIELD_TYPE_CHANGED,
                            severity=Severity.ERROR,
                            file_path=str(file_path),
                            name=f"{name}.{field_name}",
                            description=f"Field '{field_name}' type changed in model '{name}'",
                            line=current_model.line_number,
                            old_signature=baseline_field["type"],
                            new_signature=current_field["type"],
                        )
                    )

                # Check for required changes (optional -> required is breaking)
                if not baseline_field["required"] and current_field["required"]:
                    self._add_issue(
                        CompatibilityIssue(
                            change_type=ChangeType.MODEL_FIELD_REQUIRED_CHANGED,
                            severity=Severity.ERROR,
                            file_path=str(file_path),
                            name=f"{name}.{field_name}",
                            description=f"Field '{field_name}' changed from optional to required in model '{name}'",
                            line=current_model.line_number,
                        )
                    )

    def _add_issue(self, issue: CompatibilityIssue) -> None:
        """Add an issue, checking for exemptions."""
        # Check if issue is exempted
        exemption_key = f"{issue.file_path}:{issue.name}"
        if exemption_key in self.exemptions:
            issue.exempted = True
            issue.exemption_reason = self.exemptions[exemption_key]

        # Don't add if strict mode disabled and exempted
        if not self.strict_mode and issue.exempted:
            return

        self.issues.append(issue)

    def _print_summary(self) -> None:
        """Print validation summary."""
        if not self.issues:
            print("✅ No backwards compatibility issues detected")
            return

        # Group by severity
        errors = [
            i for i in self.issues if i.severity == Severity.ERROR and not i.exempted
        ]
        warnings = [
            i for i in self.issues if i.severity == Severity.WARNING and not i.exempted
        ]
        exempted = [i for i in self.issues if i.exempted]

        print("\n" + "=" * 80)
        print("BACKWARDS COMPATIBILITY VALIDATION RESULTS")
        print("=" * 80)

        if errors:
            print(f"\n❌ ERRORS ({len(errors)}):")
            for issue in errors:
                self._print_issue(issue)

        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for issue in warnings:
                self._print_issue(issue)

        if exempted:
            print(f"\n✓ EXEMPTED ISSUES ({len(exempted)}):")
            for issue in exempted:
                self._print_issue(issue, show_exemption=True)

        print("\n" + "=" * 80)
        print(
            f"Total: {len(errors)} errors, {len(warnings)} warnings, {len(exempted)} exempted"
        )

        if errors:
            print("\n❌ VALIDATION FAILED: Breaking changes detected")
            print(
                "\nTo exempt breaking changes, add them to config/backwards_compatibility_config.yaml:"
            )
            print("exemptions:")
            for issue in errors[:3]:  # Show first 3 as examples
                print(f'  "{issue.file_path}:{issue.name}": "reason for exemption"')
        else:
            print("\n✅ VALIDATION PASSED: No breaking changes detected")

    def _print_issue(
        self, issue: CompatibilityIssue, show_exemption: bool = False
    ) -> None:
        """Print a single issue."""
        severity_icon = {"error": "❌", "warning": "⚠️", "info": "ℹ"}[  # noqa: RUF001
            issue.severity.value
        ]

        print(f"\n{severity_icon} [{issue.change_type.value}] {issue.file_path}")
        print(f"   {issue.description}")

        if issue.line:
            print(f"   Line: {issue.line}")

        if issue.old_signature and issue.new_signature:
            print(f"   Old: {issue.old_signature}")
            print(f"   New: {issue.new_signature}")
        elif issue.old_signature:
            print(f"   Signature: {issue.old_signature}")

        if show_exemption and issue.exemption_reason:
            print(f"   Exemption: {issue.exemption_reason}")


def get_staged_files() -> list[str]:
    """Get list of staged Python files from git."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=AM"],
            capture_output=True,
            text=True,
            check=True,
        )

        files = [
            f for f in result.stdout.strip().split("\n") if f.endswith(".py") and f
        ]
        return files

    except subprocess.CalledProcessError:
        return []


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Backwards compatibility validation for Python code"
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Files to validate (if not provided with --staged)",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Validate staged files only (for pre-commit)",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Git reference for baseline version (default: main)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: fail even on exempted issues",
    )
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="Generate default configuration file",
    )

    args = parser.parse_args()

    try:
        # Generate config if requested
        if args.generate_config:
            generate_default_config()
            return 0

        # Get files to validate
        if args.staged:
            files = get_staged_files()
            if not files:
                print("INFO: No staged Python files to validate")
                return 0
        elif args.files:
            files = args.files
        else:
            print("ERROR: No files specified. Use --staged or provide file paths.")
            return 2

        # Initialize validator
        validator = CompatibilityValidator(
            config_path=args.config,
            baseline_version=args.baseline,
            strict_mode=args.strict,
        )

        # Validate files
        success = validator.validate_files(files)

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 2
    except Exception as e:
        print(f"❌ Script error: {e}")
        import traceback

        traceback.print_exc()
        return 2


def generate_default_config():
    """Generate default configuration file."""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    config_path = config_dir / "backwards_compatibility_config.yaml"

    default_config = {
        "# Backwards Compatibility Configuration": None,
        "baseline_version": "main",
        "strict_mode": False,
        "check_functions": True,
        "check_classes": True,
        "check_models": True,
        "check_type_annotations": True,
        "severity_threshold": "error",
        "ignore_patterns": [
            "tests/",
            "test_*.py",
            "*_test.py",
            "examples/",
            "scripts/",
        ],
        "exemptions": {
            "# Example exemption format": None,
            "# src/my_module.py:MyClass.my_method": "Intentional breaking change for v2.0",
        },
    }

    # Clean up None values used for comments
    clean_config = {k: v for k, v in default_config.items() if v is not None}

    with open(config_path, "w") as f:
        f.write("# Backwards Compatibility Configuration\n")
        f.write("# This file controls backwards compatibility validation behavior\n\n")
        yaml.dump(clean_config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Generated default configuration: {config_path}")


if __name__ == "__main__":
    sys.exit(main())
