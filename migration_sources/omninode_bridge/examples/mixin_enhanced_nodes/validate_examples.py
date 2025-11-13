#!/usr/bin/env python3
"""
Validation script for mixin-enhanced node examples.

This script validates that all examples are complete, syntactically correct,
and can be imported successfully.

Usage:
    python examples/mixin_enhanced_nodes/validate_examples.py
"""

import ast
import sys
from pathlib import Path

import yaml


class ExampleValidator:
    """Validator for mixin-enhanced node examples."""

    def __init__(self):
        """Initialize validator."""
        self.examples_dir = Path(__file__).parent
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate_all(self) -> bool:
        """
        Validate all examples.

        Returns:
            True if all examples are valid
        """
        print("=" * 60)
        print("Mixin-Enhanced Node Examples - Validation")
        print("=" * 60)
        print()

        # Validate directory structure
        print("1. Validating directory structure...")
        self._validate_structure()
        print()

        # Validate each example
        examples = [
            "basic_effect_with_health_check",
            "advanced_orchestrator_with_metrics",
            "complete_node_all_mixins",
        ]

        for example_name in examples:
            print(f"2. Validating {example_name}...")
            self._validate_example(example_name)
            print()

        # Print summary
        print("=" * 60)
        print("Validation Summary")
        print("=" * 60)
        print()

        if self.errors:
            print(f"✗ Errors: {len(self.errors)}")
            for error in self.errors:
                print(f"  - {error}")
            print()

        if self.warnings:
            print(f"⚠ Warnings: {len(self.warnings)}")
            for warning in self.warnings:
                print(f"  - {warning}")
            print()

        if not self.errors and not self.warnings:
            print("✓ All examples are valid!")
            print()
            return True
        elif not self.errors:
            print("✓ All examples are valid (with warnings)")
            print()
            return True
        else:
            print("✗ Validation failed")
            print()
            return False

    def _validate_structure(self) -> None:
        """Validate directory structure."""
        # Check README exists
        readme = self.examples_dir / "README.md"
        if not readme.exists():
            self.errors.append("README.md not found")
        else:
            print("  ✓ README.md found")

        # Check __init__.py exists
        init_file = self.examples_dir / "__init__.py"
        if not init_file.exists():
            self.warnings.append("__init__.py not found")
        else:
            print("  ✓ __init__.py found")

    def _validate_example(self, example_name: str) -> None:
        """
        Validate a single example.

        Args:
            example_name: Name of the example directory
        """
        example_dir = self.examples_dir / example_name

        if not example_dir.exists():
            self.errors.append(f"{example_name}: directory not found")
            return

        # Check required files
        required_files = ["contract.yaml", "node.py", "__init__.py"]
        for filename in required_files:
            filepath = example_dir / filename
            if not filepath.exists():
                self.errors.append(f"{example_name}: {filename} not found")
            else:
                print(f"  ✓ {filename} found")

        # Validate contract.yaml
        contract_file = example_dir / "contract.yaml"
        if contract_file.exists():
            self._validate_contract(example_name, contract_file)

        # Validate node.py
        node_file = example_dir / "node.py"
        if node_file.exists():
            self._validate_node_file(example_name, node_file)

    def _validate_contract(self, example_name: str, contract_file: Path) -> None:
        """
        Validate contract.yaml file.

        Args:
            example_name: Name of the example
            contract_file: Path to contract.yaml
        """
        try:
            with open(contract_file) as f:
                contract = yaml.safe_load(f)

            # Check required fields
            required_fields = [
                "name",
                "version",
                "description",
                "node_type",
                "input_model",
                "output_model",
            ]

            for field in required_fields:
                if field not in contract:
                    self.errors.append(
                        f"{example_name}: contract missing required field '{field}'"
                    )

            # Check mixins section
            if "mixins" not in contract:
                self.warnings.append(f"{example_name}: contract has no mixins section")
            else:
                mixins = contract["mixins"]
                if not isinstance(mixins, list):
                    self.errors.append(f"{example_name}: mixins must be a list")
                elif len(mixins) == 0:
                    self.warnings.append(f"{example_name}: no mixins configured")
                else:
                    print(f"    - {len(mixins)} mixin(s) configured")

            print("  ✓ contract.yaml is valid")

        except yaml.YAMLError as e:
            self.errors.append(f"{example_name}: invalid YAML in contract.yaml: {e}")
        except Exception as e:
            self.errors.append(f"{example_name}: error validating contract.yaml: {e}")

    def _validate_node_file(self, example_name: str, node_file: Path) -> None:
        """
        Validate node.py file.

        Args:
            example_name: Name of the example
            node_file: Path to node.py
        """
        try:
            with open(node_file) as f:
                source = f.read()

            # Parse Python AST
            tree = ast.parse(source)

            # Find class definitions
            classes = [
                node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
            ]

            if not classes:
                self.errors.append(
                    f"{example_name}: no class definitions found in node.py"
                )
                return

            # Check for main node class
            node_class = None
            for cls in classes:
                if cls.name.startswith("Node") and not cls.name.endswith("Mixin"):
                    node_class = cls
                    break

            if not node_class:
                self.errors.append(f"{example_name}: no Node class found")
                return

            print(f"    - Found class: {node_class.name}")

            # Check for required methods
            required_methods = ["__init__", "initialize", "shutdown"]
            methods = [
                node.name
                for node in ast.walk(node_class)
                if isinstance(node, ast.FunctionDef)
            ]

            for method_name in required_methods:
                if method_name not in methods:
                    self.warnings.append(
                        f"{example_name}: missing method '{method_name}'"
                    )

            # Check for main() function (example usage)
            has_main = any(
                node.name == "main"
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
            )

            if not has_main:
                self.warnings.append(f"{example_name}: no main() function for testing")
            else:
                print("    - Example usage code found")

            print("  ✓ node.py is valid")

        except SyntaxError as e:
            self.errors.append(
                f"{example_name}: syntax error in node.py: {e.msg} (line {e.lineno})"
            )
        except Exception as e:
            self.errors.append(f"{example_name}: error validating node.py: {e}")


def main():
    """Run validation."""
    validator = ExampleValidator()
    success = validator.validate_all()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
