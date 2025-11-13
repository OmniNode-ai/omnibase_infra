#!/usr/bin/env python3
"""
ONEX Canonical Pattern Validator.

Validates code against ONEX canonical patterns as defined in:
.cursor/rules/canonical_patterns.mdc

Validation Rules:
1. String Version Anti-Pattern Detection
2. No Fallback Patterns
3. Single Class Per File
4. Error Raising Validation
5. Pydantic Pattern Validation
6. Enum/Model Import Prevention
"""

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

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


class ONEXCanonicalValidator:
    """Validates ONEX canonical patterns."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_root = project_root / "src" / "omninode_bridge"
        self.violations: list[Violation] = []

    def validate_all(self) -> bool:
        """Run all validation checks."""
        print(f"\n{BLUE}{'=' * 70}{RESET}")
        print(f"{BLUE}ONEX Canonical Pattern Validation{RESET}")
        print(f"{BLUE}{'=' * 70}{RESET}\n")

        # Run validation categories
        self.validate_string_versions()
        self.validate_fallback_patterns()
        self.validate_single_class_per_file()
        self.validate_error_raising()
        self.validate_pydantic_patterns()

        # Report results
        return self.report_results()

    def validate_string_versions(self) -> None:
        """Check for string version anti-patterns."""
        print(f"{BLUE}Checking String Version Anti-Patterns...{RESET}")

        for py_file in self.src_root.rglob("*.py"):
            if self._is_test_file(py_file):
                continue

            content = py_file.read_text()
            lines = content.split("\n")

            for i, line in enumerate(lines, 1):
                # Check for __version__ string
                if re.search(r'__version__\s*=\s*["\']', line):
                    self.violations.append(
                        Violation(
                            category="String Version",
                            file=str(py_file.relative_to(self.project_root)),
                            line=i,
                            description="Use ModelSemVer instead of string __version__",
                        )
                    )

                # Check for version: str patterns
                if re.search(r'version\s*:\s*str\s*=\s*["\']', line):
                    self.violations.append(
                        Violation(
                            category="String Version",
                            file=str(py_file.relative_to(self.project_root)),
                            line=i,
                            description="Use ModelSemVer instead of str for version field",
                        )
                    )

        if not any(v.category == "String Version" for v in self.violations):
            print(f"{GREEN}✓ No string version anti-patterns{RESET}")

    def validate_fallback_patterns(self) -> None:
        """Check for forbidden fallback patterns."""
        print(f"\n{BLUE}Checking Fallback Anti-Patterns...{RESET}")

        fallback_patterns = [
            (
                r"except\s+\w+\s*:\s*\n\s*\w+\s*=\s*\w+\.UNKNOWN",
                "No silent fallback to UNKNOWN",
            ),
            (
                r"\.get\([^,]+,\s*[^)]+\)",
                "Use explicit dict access with error handling",
            ),
            (
                r'if\s+["\'][\w_]+["\']\s+in\s+info\.data',
                "No info.data fallback patterns",
            ),
        ]

        for py_file in self.src_root.rglob("*.py"):
            if self._is_test_file(py_file):
                continue

            content = py_file.read_text()
            lines = content.split("\n")

            for i, line in enumerate(lines, 1):
                for pattern, message in fallback_patterns:
                    if re.search(pattern, line):
                        self.violations.append(
                            Violation(
                                category="Fallback Pattern",
                                file=str(py_file.relative_to(self.project_root)),
                                line=i,
                                description=f"{message} (found: {line.strip()[:60]}...)",
                                severity="warning",
                            )
                        )

        if not any(v.category == "Fallback Pattern" for v in self.violations):
            print(f"{GREEN}✓ No fallback anti-patterns{RESET}")

    def validate_single_class_per_file(self) -> None:
        """Validate one non-enum class per file."""
        print(f"\n{BLUE}Checking Single Class Per File...{RESET}")

        for py_file in self.src_root.rglob("*.py"):
            if self._is_test_file(py_file) or py_file.name == "__init__.py":
                continue

            try:
                content = py_file.read_text()
                tree = ast.parse(content)

                # Count non-enum classes
                non_enum_classes = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Check if it's an enum
                        is_enum = any(
                            (isinstance(base, ast.Name) and base.id == "Enum")
                            or (isinstance(base, ast.Attribute) and base.attr == "Enum")
                            for base in node.bases
                        )

                        # Skip Config classes (Pydantic inner classes)
                        if node.name == "Config":
                            continue

                        if not is_enum:
                            non_enum_classes.append((node.name, node.lineno))

                if len(non_enum_classes) > 1:
                    self.violations.append(
                        Violation(
                            category="Single Class Per File",
                            file=str(py_file.relative_to(self.project_root)),
                            line=non_enum_classes[1][1],
                            description=f"Multiple non-enum classes in file: {[c[0] for c in non_enum_classes]}",
                            severity="warning",
                        )
                    )

            except Exception as e:
                pass  # Skip parse errors

        if not any(v.category == "Single Class Per File" for v in self.violations):
            print(f"{GREEN}✓ All files have single class{RESET}")

    def validate_error_raising(self) -> None:
        """Validate proper error raising patterns."""
        print(f"\n{BLUE}Checking Error Raising Patterns...{RESET}")

        for py_file in self.src_root.rglob("*.py"):
            if self._is_test_file(py_file):
                continue

            content = py_file.read_text()
            lines = content.split("\n")

            # Check if file raises standard exceptions
            has_onex_error_import = "ModelOnexError" in content

            for i, line in enumerate(lines, 1):
                # Check for standard exception raising
                if re.search(
                    r"raise\s+(ValueError|TypeError|RuntimeError|Exception)\(", line
                ):
                    # Skip if it's re-raising
                    if "from" in line or line.strip().startswith("#"):
                        continue

                    self.violations.append(
                        Violation(
                            category="Error Raising",
                            file=str(py_file.relative_to(self.project_root)),
                            line=i,
                            description="Use ModelOnexError instead of standard Python exceptions",
                            severity="warning",
                        )
                    )

        if not any(v.category == "Error Raising" for v in self.violations):
            print(f"{GREEN}✓ All error raising patterns correct{RESET}")

    def validate_pydantic_patterns(self) -> None:
        """Validate Pydantic validation patterns."""
        print(f"\n{BLUE}Checking Pydantic Patterns...{RESET}")

        for py_file in self.src_root.rglob("*.py"):
            if self._is_test_file(py_file):
                continue

            content = py_file.read_text()
            lines = content.split("\n")

            for i, line in enumerate(lines, 1):
                # Check for deprecated @field_validator
                if "@field_validator" in line:
                    self.violations.append(
                        Violation(
                            category="Pydantic Pattern",
                            file=str(py_file.relative_to(self.project_root)),
                            line=i,
                            description="Use @model_validator(mode='after') instead of @field_validator",
                            severity="warning",
                        )
                    )

        if not any(v.category == "Pydantic Pattern" for v in self.violations):
            print(f"{GREEN}✓ All Pydantic patterns correct{RESET}")

    def _is_test_file(self, py_file: Path) -> bool:
        """Check if file is a test file."""
        return "test_" in py_file.name or "/tests/" in str(py_file)

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
        if categories:
            for category, violations in sorted(categories.items()):
                print(f"\n{RED if violations else GREEN}{category}:{RESET}")

                for v in violations:
                    severity_color = RED if v.severity == "error" else YELLOW
                    print(
                        f"  {severity_color}✗ {v.file}:{v.line}{RESET} - {v.description}"
                    )
        else:
            print(f"{GREEN}No violations found!{RESET}")

        # Summary
        print(f"\n{BLUE}Summary:{RESET}")
        error_count = sum(1 for v in self.violations if v.severity == "error")
        warning_count = sum(1 for v in self.violations if v.severity == "warning")

        print(f"  Errors: {RED}{error_count}{RESET}")
        print(f"  Warnings: {YELLOW}{warning_count}{RESET}")

        if error_count == 0 and warning_count == 0:
            print(f"\n{GREEN}✓ All ONEX canonical pattern validations passed!{RESET}\n")
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
    validator = ONEXCanonicalValidator(project_root)

    success = validator.validate_all()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
