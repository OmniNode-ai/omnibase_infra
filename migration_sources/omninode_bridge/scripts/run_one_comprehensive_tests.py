#!/usr/bin/env python3
"""
Comprehensive O.N.E. v0.1 Protocol Component Test Execution Script.

This script executes all O.N.E. protocol component tests and validates
that registry, security, execution, and schema management components
work correctly both individually and in integration.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ONETestRunner:
    """Comprehensive test runner for O.N.E. protocol components."""

    def __init__(self):
        """Initialize test runner."""
        self.project_root = project_root
        self.test_results: dict[str, dict] = {}
        self.overall_success = True

    def run_pytest_suite(
        self, test_file: str, description: str, markers: Optional[str] = None
    ) -> dict:
        """
        Run a pytest test suite and capture results.

        Args:
            test_file: Path to test file
            description: Human-readable description
            markers: Optional pytest markers to filter tests

        Returns:
            dict: Test results summary
        """
        print(f"\n{'=' * 60}")
        print(f"🧪 Running {description}")
        print(f"📄 File: {test_file}")
        if markers:
            print(f"🏷️  Markers: {markers}")
        print(f"{'=' * 60}")

        # Build pytest command
        cmd = ["python", "-m", "pytest", test_file, "-v", "--tb=short"]

        if markers:
            cmd.extend(["-m", markers])

        # Add coverage if available
        import importlib.util

        if importlib.util.find_spec("pytest_cov") is not None:
            cmd.extend(
                [
                    "--cov=src/omninode_bridge/services/metadata_stamping",
                    "--cov-report=term-missing",
                ]
            )
        else:
            print("📝 Note: pytest-cov not available, running without coverage")

        start_time = time.time()

        try:
            # Run tests
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            execution_time = time.time() - start_time

            # Parse results
            output_lines = result.stdout.split("\n")
            passed = sum(1 for line in output_lines if " PASSED " in line)
            failed = sum(1 for line in output_lines if " FAILED " in line)
            skipped = sum(1 for line in output_lines if " SKIPPED " in line)

            success = result.returncode == 0

            test_result = {
                "description": description,
                "file": test_file,
                "success": success,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "execution_time": execution_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
            }

            # Print summary
            status_emoji = "✅" if success else "❌"
            print(
                f"\n{status_emoji} Results: {passed} passed, {failed} failed, {skipped} skipped"
            )
            print(f"⏱️  Execution time: {execution_time:.2f} seconds")

            if not success:
                print(f"❌ STDERR:\n{result.stderr}")
                self.overall_success = False

            return test_result

        except subprocess.TimeoutExpired:
            print("⏰ Test suite timed out after 5 minutes")
            self.overall_success = False
            return {
                "description": description,
                "file": test_file,
                "success": False,
                "error": "Timeout after 5 minutes",
                "execution_time": execution_time,
            }

        except Exception as e:
            print(f"💥 Test execution failed: {e}")
            self.overall_success = False
            return {
                "description": description,
                "file": test_file,
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time,
            }

    def check_test_file_exists(self, test_file: str) -> bool:
        """Check if test file exists."""
        test_path = self.project_root / test_file
        if not test_path.exists():
            print(f"⚠️  Warning: Test file not found: {test_file}")
            return False
        return True

    def run_comprehensive_test_suite(self):
        """Run the complete O.N.E. protocol test suite."""
        print("🚀 Starting O.N.E. v0.1 Protocol Comprehensive Test Suite")
        print(f"📁 Project root: {self.project_root}")
        print(f"🐍 Python version: {sys.version}")

        # Define test suites
        test_suites = [
            {
                "file": "tests/test_one_registry_comprehensive.py",
                "description": "O.N.E. Registry Component Tests",
                "markers": None,
                "critical": True,
            },
            {
                "file": "tests/test_one_security_comprehensive.py",
                "description": "O.N.E. Security Component Tests",
                "markers": None,
                "critical": True,
            },
            {
                "file": "tests/test_one_execution_comprehensive.py",
                "description": "O.N.E. Execution Component Tests",
                "markers": None,
                "critical": True,
            },
            {
                "file": "tests/test_one_schema_comprehensive.py",
                "description": "O.N.E. Schema Management Tests",
                "markers": None,
                "critical": True,
            },
            {
                "file": "tests/test_one_integration_comprehensive.py",
                "description": "O.N.E. Integration Tests",
                "markers": None,
                "critical": True,
            },
            {
                "file": "tests/compliance/test_one_protocol_v01.py",
                "description": "O.N.E. Protocol Compliance Tests",
                "markers": "compliance",
                "critical": True,
            },
        ]

        # Check all test files exist
        print("\n📋 Checking test file availability...")
        missing_files = []
        for suite in test_suites:
            if not self.check_test_file_exists(suite["file"]):
                missing_files.append(suite["file"])

        if missing_files:
            print(f"❌ Missing test files: {missing_files}")
            if any(
                suite["critical"]
                for suite in test_suites
                if suite["file"] in missing_files
            ):
                print("🛑 Critical test files missing, aborting")
                return False

        # Run test suites
        print("\n🎯 Executing test suites...")

        for suite in test_suites:
            if suite["file"] not in missing_files:
                result = self.run_pytest_suite(
                    suite["file"], suite["description"], suite.get("markers")
                )
                self.test_results[suite["file"]] = result
            else:
                print(f"⏭️  Skipping missing test file: {suite['file']}")

        # Generate comprehensive summary
        self.generate_test_summary()

        return self.overall_success

    def generate_test_summary(self):
        """Generate comprehensive test summary."""
        print(f"\n{'🏁 COMPREHENSIVE TEST SUMMARY':=^80}")

        total_passed = 0
        total_failed = 0
        total_skipped = 0
        total_execution_time = 0.0

        successful_suites = 0
        failed_suites = 0

        for test_file, result in self.test_results.items():
            status = "✅ PASSED" if result["success"] else "❌ FAILED"
            print(f"\n📄 {result['description']}")
            print(f"   {status}")

            if "passed" in result:
                print(
                    f"   📊 {result['passed']} passed, {result['failed']} failed, {result['skipped']} skipped"
                )
                total_passed += result["passed"]
                total_failed += result["failed"]
                total_skipped += result["skipped"]

            if "execution_time" in result:
                print(f"   ⏱️  {result['execution_time']:.2f} seconds")
                total_execution_time += result["execution_time"]

            if result["success"]:
                successful_suites += 1
            else:
                failed_suites += 1
                if "error" in result:
                    print(f"   ❌ Error: {result['error']}")

        # Overall summary
        print(f"\n{'OVERALL RESULTS':=^60}")
        print(f"📊 Test Suites: {successful_suites} passed, {failed_suites} failed")
        print(
            f"🧪 Individual Tests: {total_passed} passed, {total_failed} failed, {total_skipped} skipped"
        )
        print(f"⏱️  Total Execution Time: {total_execution_time:.2f} seconds")

        success_rate = (
            (total_passed / (total_passed + total_failed)) * 100
            if (total_passed + total_failed) > 0
            else 0
        )
        print(f"📈 Success Rate: {success_rate:.1f}%")

        # Component-specific summary
        print(f"\n{'COMPONENT VALIDATION':=^60}")

        component_status = {
            "Registry": (
                "✅"
                if "tests/test_one_registry_comprehensive.py" in self.test_results
                and self.test_results["tests/test_one_registry_comprehensive.py"][
                    "success"
                ]
                else "❌"
            ),
            "Security": (
                "✅"
                if "tests/test_one_security_comprehensive.py" in self.test_results
                and self.test_results["tests/test_one_security_comprehensive.py"][
                    "success"
                ]
                else "❌"
            ),
            "Execution": (
                "✅"
                if "tests/test_one_execution_comprehensive.py" in self.test_results
                and self.test_results["tests/test_one_execution_comprehensive.py"][
                    "success"
                ]
                else "❌"
            ),
            "Schema Management": (
                "✅"
                if "tests/test_one_schema_comprehensive.py" in self.test_results
                and self.test_results["tests/test_one_schema_comprehensive.py"][
                    "success"
                ]
                else "❌"
            ),
            "Integration": (
                "✅"
                if "tests/test_one_integration_comprehensive.py" in self.test_results
                and self.test_results["tests/test_one_integration_comprehensive.py"][
                    "success"
                ]
                else "❌"
            ),
            "Protocol Compliance": (
                "✅"
                if "tests/compliance/test_one_protocol_v01.py" in self.test_results
                and self.test_results["tests/compliance/test_one_protocol_v01.py"][
                    "success"
                ]
                else "❌"
            ),
        }

        for component, status in component_status.items():
            print(f"   {status} {component}")

        # Final verdict
        print(f"\n{'FINAL VERDICT':=^60}")
        if self.overall_success:
            print("🎉 ALL O.N.E. PROTOCOL COMPONENTS PASSED COMPREHENSIVE TESTING!")
            print("✅ The metadata stamping service is O.N.E. v0.1 compliant")
        else:
            print("❌ SOME O.N.E. PROTOCOL COMPONENTS FAILED TESTING")
            print("🔧 Review failed tests and fix issues before deployment")

        return self.overall_success

    def run_quick_validation(self):
        """Run a quick validation of critical O.N.E. components."""
        print("⚡ Running O.N.E. Quick Validation...")

        # Quick smoke tests
        quick_tests = [
            {
                "file": "tests/test_one_registry_comprehensive.py",
                "markers": "not slow",
                "description": "Registry Quick Check",
            },
            {
                "file": "tests/test_one_security_comprehensive.py",
                "markers": "not slow",
                "description": "Security Quick Check",
            },
        ]

        for test in quick_tests:
            if self.check_test_file_exists(test["file"]):
                result = self.run_pytest_suite(
                    test["file"], test["description"], test["markers"]
                )
                if not result["success"]:
                    print(f"❌ Quick validation failed for {test['description']}")
                    return False

        print("✅ Quick validation passed!")
        return True


def main():
    """Main entry point."""
    runner = ONETestRunner()

    # Check if quick mode requested
    if "--quick" in sys.argv:
        success = runner.run_quick_validation()
    else:
        success = runner.run_comprehensive_test_suite()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
