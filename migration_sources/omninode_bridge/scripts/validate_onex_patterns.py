#!/usr/bin/env python3
"""
ONEX v2.0 Pattern Validation Script.

Validates ONEX architectural compliance:
1. SPI Purity - No core imports in SPI layer
2. Deprecated Patterns - No use of deprecated code
3. Node Type Compliance - Proper EnumNodeType usage
4. Protocol Compliance - ModelContract and naming conventions
"""

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


@dataclass
class Violation:
    """Represents a validation violation."""

    category: str
    file: str
    line: int
    description: str
    severity: str = "error"  # error, warning


class ONEXValidator:
    """Validates ONEX architectural patterns."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_root = project_root / "src" / "omninode_bridge"
        self.violations: list[Violation] = []

    def validate_all(self) -> bool:
        """Run all validation checks."""
        print(f"\n{BLUE}{'=' * 70}{RESET}")
        print(f"{BLUE}ONEX v2.0 Pattern Validation{RESET}")
        print(f"{BLUE}{'=' * 70}{RESET}\n")

        # Run validation categories
        self.validate_spi_purity()
        self.validate_deprecated_patterns()
        self.validate_node_types()
        self.validate_protocol_compliance()

        # Report results
        return self.report_results()

    def validate_spi_purity(self) -> None:
        """Validate SPI layer doesn't import from core layer."""
        print(f"{BLUE}Checking SPI Purity...{RESET}")

        # SPI layer should not import from omnibase_core
        # SPI files are typically in nodes/*/v*/node.py or models/
        spi_patterns = [
            self.src_root / "nodes" / "**" / "node.py",
            self.src_root / "nodes" / "**" / "models" / "*.py",
        ]

        for pattern_base in [
            self.src_root / "nodes",
        ]:
            if not pattern_base.exists():
                continue

            for py_file in pattern_base.rglob("*.py"):
                if self._is_test_file(py_file):
                    continue

                # Check for omnibase_core imports
                violations = self._check_core_imports(py_file)
                self.violations.extend(violations)

        if not any(v.category == "SPI Purity" for v in self.violations):
            print(f"{GREEN}✓ No SPI purity violations{RESET}")

    def validate_deprecated_patterns(self) -> None:
        """Check for deprecated pattern usage."""
        print(f"\n{BLUE}Checking Deprecated Patterns...{RESET}")

        deprecated_patterns = [
            # Old imports that should be updated
            (r"from omnibase_legacy", "Use omnibase_core instead"),
            (
                r"NodeBase\b",
                "Use proper ONEX node types (NodeEffect, NodeCompute, etc.)",
            ),
            # Add more deprecated patterns as needed
        ]

        for py_file in self.src_root.rglob("*.py"):
            if self._is_test_file(py_file):
                continue

            content = py_file.read_text()
            lines = content.split("\n")

            for i, line in enumerate(lines, 1):
                for pattern, message in deprecated_patterns:
                    if re.search(pattern, line):
                        self.violations.append(
                            Violation(
                                category="Deprecated Pattern",
                                file=str(py_file.relative_to(self.project_root)),
                                line=i,
                                description=f"{message} (found: {line.strip()[:60]}...)",
                                severity="warning",
                            )
                        )

        if not any(v.category == "Deprecated Pattern" for v in self.violations):
            print(f"{GREEN}✓ No deprecated patterns found{RESET}")

    def validate_node_types(self) -> None:
        """Validate proper node type declarations."""
        print(f"\n{BLUE}Checking Node Type Compliance...{RESET}")

        node_files = list(self.src_root.glob("nodes/*/v*/node.py"))

        for node_file in node_files:
            # Parse the Python file
            try:
                content = node_file.read_text()
                tree = ast.parse(content)

                # Check for proper class inheritance
                class_defs = [
                    node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
                ]

                for cls in class_defs:
                    if cls.name.startswith("Node"):
                        # Check if it inherits from proper ONEX base
                        base_names = [self._get_name(base) for base in cls.bases]

                        # Valid ONEX bases
                        valid_bases = [
                            "NodeEffect",
                            "NodeCompute",
                            "NodeReducer",
                            "NodeOrchestrator",
                        ]

                        has_valid_base = any(base in valid_bases for base in base_names)

                        if not has_valid_base and not any(
                            "Mixin" in base for base in base_names
                        ):
                            self.violations.append(
                                Violation(
                                    category="Node Type",
                                    file=str(node_file.relative_to(self.project_root)),
                                    line=cls.lineno,
                                    description=f"Class {cls.name} should inherit from a valid ONEX node type ({', '.join(valid_bases)}). Found: {base_names}",
                                )
                            )

            except Exception as e:
                print(
                    f"{YELLOW}⚠ Could not parse {node_file.relative_to(self.project_root)}: {e}{RESET}"
                )

        if not any(v.category == "Node Type" for v in self.violations):
            print(f"{GREEN}✓ All node types are compliant{RESET}")

    def validate_protocol_compliance(self) -> None:
        """Validate ONEX protocol compliance (naming, contracts)."""
        print(f"\n{BLUE}Checking Protocol Compliance...{RESET}")

        # Check naming conventions
        model_files = list(self.src_root.glob("nodes/*/v*/models/*.py"))

        for model_file in model_files:
            filename = model_file.name

            # Validate file naming patterns
            if filename.startswith("model_") or filename.startswith("enum_"):
                # Check if class names match conventions
                try:
                    content = model_file.read_text()
                    tree = ast.parse(content)

                    class_defs = [
                        node
                        for node in ast.walk(tree)
                        if isinstance(node, ast.ClassDef)
                    ]

                    for cls in class_defs:
                        # Model files should have Model* classes
                        if filename.startswith("model_") and not cls.name.startswith(
                            "Model"
                        ):
                            self.violations.append(
                                Violation(
                                    category="Protocol Compliance",
                                    file=str(model_file.relative_to(self.project_root)),
                                    line=cls.lineno,
                                    description=f"Class {cls.name} in {filename} should start with 'Model' prefix",
                                    severity="warning",
                                )
                            )

                        # Enum files should have Enum* classes
                        if filename.startswith("enum_") and not cls.name.startswith(
                            "Enum"
                        ):
                            self.violations.append(
                                Violation(
                                    category="Protocol Compliance",
                                    file=str(model_file.relative_to(self.project_root)),
                                    line=cls.lineno,
                                    description=f"Class {cls.name} in {filename} should start with 'Enum' prefix",
                                    severity="warning",
                                )
                            )

                except Exception as e:
                    pass  # Skip parse errors

        if not any(v.category == "Protocol Compliance" for v in self.violations):
            print(f"{GREEN}✓ All protocol naming compliant{RESET}")

    def _check_core_imports(self, py_file: Path) -> list[Violation]:
        """Check a file for forbidden core imports."""
        violations = []

        try:
            content = py_file.read_text()
            lines = content.split("\n")

            for i, line in enumerate(lines, 1):
                # Check for omnibase_core imports
                if "from omnibase_core" in line or "import omnibase_core" in line:
                    # Exceptions: tests, main files, and certain allowed files
                    if self._is_allowed_core_import(py_file):
                        continue

                    violations.append(
                        Violation(
                            category="SPI Purity",
                            file=str(py_file.relative_to(self.project_root)),
                            line=i,
                            description=f"SPI layer should not import from omnibase_core (found: {line.strip()})",
                        )
                    )

        except Exception as e:
            pass  # Skip read errors

        return violations

    def _is_allowed_core_import(self, py_file: Path) -> bool:
        """Check if a file is allowed to import from core."""
        # Allow core imports in:
        # - Main entry points
        # - Service layers (non-SPI)
        # - Infrastructure
        # - Tests

        allowed_patterns = [
            "main.py",
            "main_standalone.py",
            "services/",
            "infrastructure/",
            "persistence/",
            "test_",
            "conftest.py",
        ]

        path_str = str(py_file)
        return any(pattern in path_str for pattern in allowed_patterns)

    def _is_test_file(self, py_file: Path) -> bool:
        """Check if file is a test file."""
        return "test_" in py_file.name or "tests/" in str(py_file)

    def _get_name(self, node: Any) -> str:
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def report_results(self) -> bool:
        """Report validation results."""
        print(f"\n{BLUE}{'=' * 70}{RESET}")
        print(f"{BLUE}Validation Results{RESET}")
        print(f"{BLUE}{'=' * 70}{RESET}\n")

        # Group by category
        categories = {}
        for v in self.violations:
            if v.category not in categories:
                categories[v.category] = []
            categories[v.category].append(v)

        # Print by category
        for category, violations in sorted(categories.items()):
            print(f"\n{RED if violations else GREEN}{category}:{RESET}")

            for v in violations:
                severity_color = RED if v.severity == "error" else YELLOW
                print(f"  {severity_color}✗ {v.file}:{v.line}{RESET} - {v.description}")

        # Summary
        print(f"\n{BLUE}Summary:{RESET}")
        error_count = sum(1 for v in self.violations if v.severity == "error")
        warning_count = sum(1 for v in self.violations if v.severity == "warning")

        print(f"  Errors: {RED}{error_count}{RESET}")
        print(f"  Warnings: {YELLOW}{warning_count}{RESET}")

        if error_count == 0 and warning_count == 0:
            print(f"\n{GREEN}✓ All ONEX pattern validations passed!{RESET}\n")
            return True
        elif error_count == 0:
            print(
                f"\n{YELLOW}⚠ Validation passed with {warning_count} warnings{RESET}\n"
            )
            return True
        else:
            print(f"\n{RED}✗ Validation failed with {error_count} errors{RESET}\n")
            return False


def main() -> int:
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    validator = ONEXValidator(project_root)

    success = validator.validate_all()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
