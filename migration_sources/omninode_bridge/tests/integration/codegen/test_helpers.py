#!/usr/bin/env python3
"""
Test Helper Utilities for Mixin Generation Tests.

Provides common assertion helpers, code analysis utilities, and test data generators
for comprehensive mixin-enhanced code generation testing.

Author: Test Generator
"""

import ast
import re
from dataclasses import dataclass
from typing import Any, Optional

import yaml

# ============================================================================
# Data Classes for Test Results
# ============================================================================


@dataclass
class CodeAnalysisResult:
    """Result of code analysis."""

    is_valid_python: bool
    has_class_definition: bool
    class_name: Optional[str]
    base_classes: list[str]
    methods: list[str]
    imports: list[str]
    mixins: list[str]
    errors: list[str]
    warnings: list[str]


@dataclass
class MixinAnalysisResult:
    """Result of mixin-specific analysis."""

    declared_mixins: list[str]
    inherited_mixins: list[str]
    imported_mixins: list[str]
    missing_imports: list[str]
    missing_inheritance: list[str]
    extra_mixins: list[str]


# ============================================================================
# Code Analysis Utilities
# ============================================================================


class CodeAnalyzer:
    """Analyze generated Python code for testing."""

    @staticmethod
    def analyze_code(code: str) -> CodeAnalysisResult:
        """
        Comprehensive code analysis.

        Args:
            code: Python source code to analyze

        Returns:
            CodeAnalysisResult with all analysis data
        """
        errors = []
        warnings = []
        is_valid = True
        has_class = False
        class_name = None
        base_classes = []
        methods = []
        imports = []
        mixins = []

        # Parse syntax
        try:
            tree = ast.parse(code)
            is_valid = True
        except SyntaxError as e:
            is_valid = False
            errors.append(f"Syntax error: {e}")
            return CodeAnalysisResult(
                is_valid_python=False,
                has_class_definition=False,
                class_name=None,
                base_classes=[],
                methods=[],
                imports=[],
                mixins=[],
                errors=errors,
                warnings=[],
            )

        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")

        # Extract class information
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                has_class = True
                class_name = node.name

                # Extract base classes
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_classes.append(base.id)
                        if "Mixin" in base.id:
                            mixins.append(base.id)

                # Extract methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) or isinstance(
                        item, ast.AsyncFunctionDef
                    ):
                        methods.append(item.name)

        return CodeAnalysisResult(
            is_valid_python=is_valid,
            has_class_definition=has_class,
            class_name=class_name,
            base_classes=base_classes,
            methods=methods,
            imports=imports,
            mixins=mixins,
            errors=errors,
            warnings=warnings,
        )

    @staticmethod
    def extract_class_name(code: str) -> Optional[str]:
        """Extract class name from code."""
        match = re.search(r"class\s+(\w+)", code)
        return match.group(1) if match else None

    @staticmethod
    def extract_base_classes(code: str) -> list[str]:
        """Extract base classes from class definition."""
        match = re.search(r"class\s+\w+\(([^)]+)\):", code)
        if not match:
            return []

        bases_str = match.group(1)
        # Split on commas and clean whitespace
        return [base.strip() for base in bases_str.split(",")]

    @staticmethod
    def extract_methods(code: str) -> list[str]:
        """Extract method names from code."""
        pattern = r"(?:async\s+)?def\s+(\w+)\s*\("
        return re.findall(pattern, code)

    @staticmethod
    def extract_imports(code: str) -> list[str]:
        """Extract all import statements."""
        pattern = r"(?:from\s+[\w.]+\s+)?import\s+(.+)"
        matches = re.findall(pattern, code)
        imports = []
        for match in matches:
            # Handle multiple imports on one line
            for item in match.split(","):
                imports.append(item.strip().split(" as ")[0])
        return imports

    @staticmethod
    def has_super_init_call(code: str) -> bool:
        """Check if code has super().__init__() call."""
        return bool(re.search(r"super\(\).__init__\(", code))

    @staticmethod
    def count_lines_of_code(code: str) -> int:
        """Count non-empty, non-comment lines."""
        lines = code.split("\n")
        loc = 0
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                loc += 1
        return loc


# ============================================================================
# Mixin Analysis Utilities
# ============================================================================


class MixinAnalyzer:
    """Analyze mixin usage in generated code."""

    @staticmethod
    def analyze_mixins(code: str, contract: dict[str, Any]) -> MixinAnalysisResult:
        """
        Analyze mixin declaration vs usage.

        Args:
            code: Generated Python code
            contract: Original contract dict

        Returns:
            MixinAnalysisResult with discrepancies
        """
        # Extract declared mixins from contract
        declared_mixins = []
        mixin_config = contract.get("mixin_configuration", {})
        for mixin in mixin_config.get("mixins", []):
            declared_mixins.append(mixin["mixin_name"])

        # Extract inherited mixins from code
        inherited_mixins = []
        match = re.search(r"class\s+\w+\(([^)]+)\):", code)
        if match:
            bases = match.group(1).split(",")
            for base in bases:
                base = base.strip()
                if "Mixin" in base:
                    inherited_mixins.append(base)

        # Extract imported mixins
        imported_mixins = []
        import_pattern = r"from\s+[\w.]+\s+import\s+.*?(Mixin\w+)"
        for match in re.finditer(import_pattern, code):
            imported_mixins.append(match.group(1))

        # Find discrepancies
        declared_set = set(declared_mixins)
        inherited_set = set(inherited_mixins)
        imported_set = set(imported_mixins)

        missing_imports = list(declared_set - imported_set)
        missing_inheritance = list(declared_set - inherited_set)
        extra_mixins = list(inherited_set - declared_set)

        return MixinAnalysisResult(
            declared_mixins=declared_mixins,
            inherited_mixins=inherited_mixins,
            imported_mixins=imported_mixins,
            missing_imports=missing_imports,
            missing_inheritance=missing_inheritance,
            extra_mixins=extra_mixins,
        )

    @staticmethod
    def has_mixin_in_inheritance(code: str, mixin_name: str) -> bool:
        """Check if mixin is in class inheritance."""
        pattern = rf"class\s+\w+\([^)]*{mixin_name}[^)]*\):"
        return bool(re.search(pattern, code))

    @staticmethod
    def has_mixin_import(code: str, mixin_name: str) -> bool:
        """Check if mixin is imported."""
        pattern = rf"from\s+[\w.]+\s+import\s+.*{mixin_name}"
        return bool(re.search(pattern, code))

    @staticmethod
    def count_mixins(code: str) -> int:
        """Count number of mixins in class inheritance."""
        match = re.search(r"class\s+\w+\(([^)]+)\):", code)
        if not match:
            return 0
        inheritance = match.group(1)
        return inheritance.count("Mixin")


# ============================================================================
# Contract Analysis Utilities
# ============================================================================


class ContractAnalyzer:
    """Analyze YAML contracts."""

    @staticmethod
    def load_contract_from_file(contract_path: str) -> dict[str, Any]:
        """Load and parse YAML contract."""
        with open(contract_path) as f:
            return yaml.safe_load(f)

    @staticmethod
    def get_declared_mixins(contract: dict[str, Any]) -> list[str]:
        """Extract mixin names from contract."""
        mixins = []
        mixin_config = contract.get("mixin_configuration", {})
        for mixin in mixin_config.get("mixins", []):
            mixins.append(mixin["mixin_name"])
        return mixins

    @staticmethod
    def has_event_patterns(contract: dict[str, Any]) -> bool:
        """Check if contract defines event patterns."""
        return "event_patterns" in contract

    @staticmethod
    def get_node_type(contract: dict[str, Any]) -> Optional[str]:
        """Extract node type from contract."""
        return contract.get("node_type")

    @staticmethod
    def validate_required_fields(contract: dict[str, Any]) -> list[str]:
        """Check for required fields, return list of missing fields."""
        required = ["node_id", "node_type", "version"]
        missing = []
        for field in required:
            if field not in contract:
                missing.append(field)
        return missing


# ============================================================================
# Assertion Helpers
# ============================================================================


class AssertionHelpers:
    """Custom assertion helpers for tests."""

    @staticmethod
    def assert_valid_python(code: str):
        """Assert code is syntactically valid Python."""
        try:
            ast.parse(code)
        except SyntaxError as e:
            raise AssertionError(f"Invalid Python syntax: {e}")

    @staticmethod
    def assert_has_class(code: str, class_name: Optional[str] = None):
        """Assert code has class definition."""
        if class_name:
            pattern = rf"class\s+{class_name}\s*\("
            if not re.search(pattern, code):
                raise AssertionError(f"Class {class_name} not found in code")
        else:
            if not re.search(r"class\s+\w+\s*\(", code):
                raise AssertionError("No class definition found in code")

    @staticmethod
    def assert_has_method(code: str, method_name: str):
        """Assert code has method definition."""
        pattern = rf"(?:async\s+)?def\s+{method_name}\s*\("
        if not re.search(pattern, code):
            raise AssertionError(f"Method {method_name} not found in code")

    @staticmethod
    def assert_has_mixin(code: str, mixin_name: str):
        """Assert mixin is in class inheritance."""
        if not MixinAnalyzer.has_mixin_in_inheritance(code, mixin_name):
            raise AssertionError(f"Mixin {mixin_name} not in class inheritance")

    @staticmethod
    def assert_imports_mixin(code: str, mixin_name: str):
        """Assert mixin is imported."""
        if not MixinAnalyzer.has_mixin_import(code, mixin_name):
            raise AssertionError(f"Mixin {mixin_name} not imported")

    @staticmethod
    def assert_has_super_init(code: str):
        """Assert code calls super().__init__()."""
        if not CodeAnalyzer.has_super_init_call(code):
            raise AssertionError("No super().__init__() call found")

    @staticmethod
    def assert_mixin_count(code: str, expected_count: int):
        """Assert number of mixins in inheritance."""
        actual_count = MixinAnalyzer.count_mixins(code)
        if actual_count != expected_count:
            raise AssertionError(
                f"Expected {expected_count} mixins, found {actual_count}"
            )

    @staticmethod
    def assert_no_syntax_errors(code: str):
        """Assert code has no syntax errors (stricter than assert_valid_python)."""
        AssertionHelpers.assert_valid_python(code)
        # Additional checks
        if "SyntaxError" in code:
            raise AssertionError("Code contains 'SyntaxError' string")


# ============================================================================
# Test Data Generators
# ============================================================================


class TestDataGenerator:
    """Generate test data for various scenarios."""

    @staticmethod
    def create_minimal_contract(
        node_id: str = "test_node",
        node_type: str = "effect",
        version: str = "v1_0_0",
    ) -> dict[str, Any]:
        """Create minimal valid contract."""
        return {
            "node_id": node_id,
            "node_type": node_type,
            "version": version,
            "metadata": {
                "name": f"Test {node_type.capitalize()} Node",
                "description": f"Test {node_type} node",
            },
        }

    @staticmethod
    def create_contract_with_mixins(
        node_id: str, node_type: str, mixins: list[str]
    ) -> dict[str, Any]:
        """Create contract with specified mixins."""
        contract = TestDataGenerator.create_minimal_contract(node_id, node_type)
        contract["mixin_configuration"] = {
            "mixins": [{"mixin_name": mixin, "config": {}} for mixin in mixins]
        }
        return contract

    @staticmethod
    def create_contract_with_events(
        node_id: str,
        node_type: str,
        subscribes: list[str],
        publishes: list[str],
    ) -> dict[str, Any]:
        """Create contract with event patterns."""
        contract = TestDataGenerator.create_minimal_contract(node_id, node_type)
        contract["event_patterns"] = {
            "subscribes": [
                {"topic": topic, "event_type": f"{topic}Event"} for topic in subscribes
            ],
            "publishes": [
                {"topic": topic, "event_type": f"{topic}Event"} for topic in publishes
            ],
        }
        return contract


# ============================================================================
# Performance Helpers
# ============================================================================


class PerformanceHelpers:
    """Helpers for performance testing."""

    @staticmethod
    def measure_generation_time(func, *args, **kwargs):
        """Measure function execution time."""
        import time

        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        return result, duration

    @staticmethod
    def measure_lines_of_code(code: str) -> dict[str, int]:
        """Measure LOC metrics."""
        lines = code.split("\n")
        total = len(lines)
        blank = sum(1 for line in lines if not line.strip())
        comments = sum(1 for line in lines if line.strip().startswith("#"))
        code_lines = total - blank - comments

        return {
            "total": total,
            "blank": blank,
            "comments": comments,
            "code": code_lines,
        }
