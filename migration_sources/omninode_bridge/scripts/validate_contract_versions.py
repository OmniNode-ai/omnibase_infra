#!/usr/bin/env python3
"""
ONEX v2.0 Contract Version Format Validator

Validates that all contract.yaml files use the structured version format:
  version:
    major: 1
    minor: 0
    patch: 0

Instead of string format:
  version: "1.0.0"
"""

import sys
from pathlib import Path

import yaml

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def validate_contract_version(contract_path: Path) -> tuple[bool, str]:
    """
    Validate a single contract file.

    Returns:
        (is_valid, message)
    """
    try:
        with open(contract_path) as f:
            contract = yaml.safe_load(f)

        # Check if version field exists
        if "version" not in contract:
            return False, "Missing 'version' field"

        version = contract["version"]

        # Check if version is a dict (structured format)
        if not isinstance(version, dict):
            return (
                False,
                f"Version is {type(version).__name__}, should be dict with major/minor/patch",
            )

        # Check required fields
        required_fields = ["major", "minor", "patch"]
        missing_fields = [f for f in required_fields if f not in version]

        if missing_fields:
            return False, f"Version missing fields: {', '.join(missing_fields)}"

        # Check types
        for field in required_fields:
            if not isinstance(version[field], int):
                return (
                    False,
                    f"Version.{field} should be int, got {type(version[field]).__name__}",
                )

        # Check required top-level fields
        required_top_level = ["name", "node_type", "description"]
        missing_top = [f for f in required_top_level if f not in contract]

        if missing_top:
            return False, f"Missing required fields: {', '.join(missing_top)}"

        return True, "✓ Valid structured version format"

    except yaml.YAMLError as e:
        return False, f"YAML parse error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Main validation entry point."""
    project_root = Path(__file__).parent.parent
    nodes_dir = project_root / "src" / "omninode_bridge" / "nodes"

    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}ONEX v2.0 Contract Version Format Validation{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}\n")

    # Find all contract.yaml files
    contract_files = list(nodes_dir.glob("*/v*/contract.yaml"))

    if not contract_files:
        print(f"{RED}No contract.yaml files found in {nodes_dir}{RESET}")
        return 1

    print(f"Found {len(contract_files)} contract files\n")

    # Validate each contract
    results = []
    for contract_path in sorted(contract_files):
        relative_path = contract_path.relative_to(project_root)
        is_valid, message = validate_contract_version(contract_path)
        results.append((relative_path, is_valid, message))

        # Print result
        color = GREEN if is_valid else RED
        symbol = "✓" if is_valid else "✗"
        print(f"{color}{symbol} {relative_path}{RESET}")
        if not is_valid:
            print(f"  {RED}{message}{RESET}")
        else:
            # Read version for display
            with open(contract_path) as f:
                contract = yaml.safe_load(f)
                version = contract["version"]
                print(
                    f"  {YELLOW}  Version: {version['major']}.{version['minor']}.{version['patch']}{RESET}"
                )

    # Summary
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}Summary{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}\n")

    total = len(results)
    passed = sum(1 for _, is_valid, _ in results if is_valid)
    failed = total - passed

    print(f"Total contracts: {total}")
    print(f"{GREEN}✓ Passed: {passed}{RESET}")

    if failed > 0:
        print(f"{RED}✗ Failed: {failed}{RESET}")
        print(
            f"\n{RED}Validation failed - {failed} contract(s) have invalid version format{RESET}\n"
        )
        return 1
    else:
        print(
            f"\n{GREEN}✓ All contracts have valid ONEX v2.0 structured version format!{RESET}\n"
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
