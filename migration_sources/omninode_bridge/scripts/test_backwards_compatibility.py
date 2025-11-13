#!/usr/bin/env python3
"""
Test script for backwards compatibility validator.

This script creates test cases and validates that the backwards compatibility
validator correctly detects breaking changes.

Usage:
    python scripts/test_backwards_compatibility.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path


class TestCase:
    """Represents a test case for the validator."""

    def __init__(
        self, name: str, baseline_code: str, current_code: str, should_fail: bool
    ):
        self.name = name
        self.baseline_code = baseline_code
        self.current_code = current_code
        self.should_fail = should_fail


TEST_CASES = [
    TestCase(
        name="Function Removal (Breaking)",
        baseline_code="""
def public_function(param: str) -> str:
    '''A public function.'''
    return param
""",
        current_code="""
# Function removed
""",
        should_fail=True,
    ),
    TestCase(
        name="Function Signature Change (Breaking)",
        baseline_code="""
def create_user(name: str, email: str) -> dict:
    '''Create a user.'''
    return {"name": name, "email": email}
""",
        current_code="""
def create_user(name: str, email: str, tenant_id: str) -> dict:
    '''Create a user.'''
    return {"name": name, "email": email, "tenant_id": tenant_id}
""",
        should_fail=True,
    ),
    TestCase(
        name="Optional Parameter Addition (Safe)",
        baseline_code="""
def create_user(name: str, email: str) -> dict:
    '''Create a user.'''
    return {"name": name, "email": email}
""",
        current_code="""
from typing import Optional

def create_user(name: str, email: str, tenant_id: Optional[str] = None) -> dict:
    '''Create a user.'''
    return {"name": name, "email": email, "tenant_id": tenant_id}
""",
        should_fail=False,
    ),
    TestCase(
        name="Class Removal (Breaking)",
        baseline_code="""
class UserClass:
    '''A user class.'''
    def __init__(self, name: str):
        self.name = name
""",
        current_code="""
# Class removed
""",
        should_fail=True,
    ),
    TestCase(
        name="Pydantic Model Field Removal (Breaking)",
        baseline_code="""
from pydantic import BaseModel

class UserModel(BaseModel):
    name: str
    email: str
    age: int
""",
        current_code="""
from pydantic import BaseModel

class UserModel(BaseModel):
    name: str
    email: str
    # age field removed
""",
        should_fail=True,
    ),
    TestCase(
        name="Pydantic Model Field Type Change (Breaking)",
        baseline_code="""
from pydantic import BaseModel
from typing import Optional

class UserModel(BaseModel):
    name: str
    email: Optional[str] = None
""",
        current_code="""
from pydantic import BaseModel, EmailStr

class UserModel(BaseModel):
    name: str
    email: EmailStr  # Changed from Optional[str]
""",
        should_fail=True,
    ),
    TestCase(
        name="Pydantic Model Field Addition (Safe)",
        baseline_code="""
from pydantic import BaseModel

class UserModel(BaseModel):
    name: str
    email: str
""",
        current_code="""
from pydantic import BaseModel
from typing import Optional

class UserModel(BaseModel):
    name: str
    email: str
    age: Optional[int] = None  # New optional field
""",
        should_fail=False,
    ),
    TestCase(
        name="Private Function Changes (Safe)",
        baseline_code="""
def _private_function(param: str) -> str:
    '''A private function.'''
    return param
""",
        current_code="""
def _private_function(param: int) -> int:
    '''A private function with changed signature.'''
    return param * 2
""",
        should_fail=False,
    ),
    TestCase(
        name="Method Signature Change (Breaking)",
        baseline_code="""
class Service:
    def process(self, data: dict) -> dict:
        '''Process data.'''
        return data
""",
        current_code="""
class Service:
    def process(self, data: list) -> list:
        '''Process data.'''
        return data
""",
        should_fail=True,
    ),
    TestCase(
        name="Return Type Change (Breaking)",
        baseline_code="""
def get_user() -> dict:
    '''Get user data.'''
    return {"name": "John"}
""",
        current_code="""
def get_user() -> list:
    '''Get user data.'''
    return [{"name": "John"}]
""",
        should_fail=True,
    ),
]


def run_test_case(test_case: TestCase, validator_script: Path) -> tuple[bool, str]:
    """Run a single test case and return result."""
    print(f"\n{'=' * 80}")
    print(f"TEST: {test_case.name}")
    print(f"Expected to fail: {test_case.should_fail}")
    print(f"{'=' * 80}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create a git repo
        subprocess.run(["git", "init"], cwd=tmppath, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=tmppath,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=tmppath,
            capture_output=True,
            check=True,
        )

        # Create test file with baseline code
        test_file = tmppath / "test_module.py"
        test_file.write_text(test_case.baseline_code)

        # Commit baseline
        subprocess.run(
            ["git", "add", "."], cwd=tmppath, capture_output=True, check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "baseline"],
            cwd=tmppath,
            capture_output=True,
            check=True,
        )

        # Update file with current code
        test_file.write_text(test_case.current_code)

        # Run validator
        result = subprocess.run(
            [
                sys.executable,
                str(validator_script),
                str(test_file),
                "--baseline",
                "HEAD",
            ],
            cwd=tmppath,
            capture_output=True,
            text=True,
        )

        print("STDOUT:")
        print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        # Check if result matches expectation
        validation_failed = result.returncode != 0

        if validation_failed == test_case.should_fail:
            print("\n✅ TEST PASSED")
            return True, "Test passed"
        else:
            print("\n❌ TEST FAILED")
            if test_case.should_fail:
                return False, "Expected validation to fail, but it passed"
            else:
                return False, "Expected validation to pass, but it failed"


def main():
    """Run all test cases."""
    validator_script = Path(__file__).parent / "validate_backwards_compatibility.py"

    if not validator_script.exists():
        print(f"❌ ERROR: Validator script not found: {validator_script}")
        return 1

    print("=" * 80)
    print("BACKWARDS COMPATIBILITY VALIDATOR TEST SUITE")
    print("=" * 80)
    print(f"Validator: {validator_script}")
    print(f"Test cases: {len(TEST_CASES)}")

    results = []
    for test_case in TEST_CASES:
        try:
            passed, message = run_test_case(test_case, validator_script)
            results.append((test_case.name, passed, message))
        except Exception as e:
            print(f"\n❌ TEST ERROR: {e}")
            results.append((test_case.name, False, f"Exception: {e}"))

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed_count = sum(1 for _, passed, _ in results if passed)
    failed_count = len(results) - passed_count

    for name, passed, message in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if not passed:
            print(f"       {message}")

    print("\n" + "=" * 80)
    print(f"Results: {passed_count}/{len(results)} passed, {failed_count} failed")
    print("=" * 80)

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
