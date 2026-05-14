# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the delegation env read scanner.

Verifies that the scanner:
- finds known os.environ/os.getenv violations in delegation modules
- ignores test fixtures and non-delegation modules
- operates in report mode without failing CI

Ticket: OMN-10917
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

scripts_ci_dir = Path(__file__).parent.parent.parent / "scripts" / "ci"
sys.path.insert(0, str(scripts_ci_dir))

from check_delegation_env_reads import (
    ScanResult,
    _find_env_calls_in_source,
    _is_allowlisted,
    _is_delegation_module,
    scan_delegation_modules,
)


@pytest.mark.unit
class TestIsDelegationModule:
    def test_delegation_orchestrator(self) -> None:
        assert _is_delegation_module(
            "src/omnibase_infra/nodes/node_delegation_orchestrator/handlers/h.py"
        )

    def test_delegation_routing_reducer(self) -> None:
        assert _is_delegation_module(
            "src/omnibase_infra/nodes/node_delegation_routing_reducer/handlers/h.py"
        )

    def test_delegate_skill_orchestrator(self) -> None:
        assert _is_delegation_module(
            "src/omnibase_infra/nodes/node_delegate_skill_orchestrator/node.py"
        )

    def test_llm_inference_effect(self) -> None:
        assert _is_delegation_module(
            "src/omnibase_infra/nodes/node_llm_inference_effect/handlers/bifrost/cfg.py"
        )

    def test_llm_completion_effect(self) -> None:
        assert _is_delegation_module(
            "src/omnibase_infra/nodes/node_llm_completion_effect/handlers/h.py"
        )

    def test_adapters_llm(self) -> None:
        assert _is_delegation_module(
            "src/omnibase_infra/adapters/llm/adapter_llm_provider_openai.py"
        )

    def test_non_delegation_module_excluded(self) -> None:
        assert not _is_delegation_module(
            "src/omnibase_infra/nodes/node_registration_orchestrator/node.py"
        )

    def test_event_bus_excluded(self) -> None:
        assert not _is_delegation_module(
            "src/omnibase_infra/event_bus/kafka_consumer.py"
        )


@pytest.mark.unit
class TestIsAllowlisted:
    def test_test_file_allowlisted(self) -> None:
        assert _is_allowlisted(
            "src/omnibase_infra/nodes/node_delegation_orchestrator/tests/test_foo.py"
        )

    def test_fixtures_allowlisted(self) -> None:
        assert _is_allowlisted("tests/fixtures/delegation_fixture.py")

    def test_conftest_allowlisted(self) -> None:
        assert _is_allowlisted("tests/ci/conftest.py")

    def test_regular_handler_not_allowlisted(self) -> None:
        assert not _is_allowlisted(
            "src/omnibase_infra/adapters/llm/adapter_llm_provider_openai.py"
        )


@pytest.mark.unit
class TestFindEnvCallsInSource:
    def test_detects_os_environ_get(self) -> None:
        source = 'import os\nvalue = os.environ.get("KEY", "")\n'
        violations = _find_env_calls_in_source(source, "test.py")
        assert len(violations) == 1
        assert "os.environ.get" in violations[0]
        assert "test.py:2" in violations[0]

    def test_detects_os_getenv(self) -> None:
        source = 'import os\nvalue = os.getenv("KEY")\n'
        violations = _find_env_calls_in_source(source, "test.py")
        assert len(violations) == 1
        assert "os.getenv" in violations[0]

    def test_detects_os_environ_subscript(self) -> None:
        source = 'import os\nvalue = os.environ["KEY"]\n'
        violations = _find_env_calls_in_source(source, "test.py")
        assert len(violations) == 1
        assert "os.environ" in violations[0]

    def test_skips_onex_flag_exempt(self) -> None:
        source = (
            'import os\nvalue = os.getenv("KEY")  # ONEX_FLAG_EXEMPT: activation gate\n'
        )
        violations = _find_env_calls_in_source(source, "test.py")
        assert len(violations) == 0

    def test_skips_onex_exclude(self) -> None:
        source = (
            'import os\nvalue = os.environ.get("KEY")  # ONEX_EXCLUDE: archive port\n'
        )
        violations = _find_env_calls_in_source(source, "test.py")
        assert len(violations) == 0

    def test_no_false_positive_on_attribute_not_os(self) -> None:
        source = 'value = config.environ.get("KEY")\n'
        violations = _find_env_calls_in_source(source, "test.py")
        assert len(violations) == 0

    def test_multiple_violations_detected(self) -> None:
        source = (
            "import os\n"
            'url = os.environ.get("LLM_URL", "")\n'
            'key = os.getenv("API_KEY")\n'
        )
        violations = _find_env_calls_in_source(source, "test.py")
        assert len(violations) == 2


@pytest.mark.unit
class TestScanDelegationModules:
    def test_scanner_finds_known_violations(self) -> None:
        repo_root = Path(__file__).parent.parent.parent
        result = scan_delegation_modules(repo_root=repo_root, mode="report")
        assert result.scanned_files > 0
        assert result.report_generated

    def test_delegation_modules_have_no_env_reads(self) -> None:
        """OMN-10925 removed all env reads from delegation modules; scanner
        must confirm the clean state stays clean. Any new violation means
        someone reintroduced an env read inside the delegation pipeline."""
        repo_root = Path(__file__).parent.parent.parent
        result = scan_delegation_modules(repo_root=repo_root, mode="report")
        assert result.violations == [], (
            "Delegation modules must read all config from contracts. "
            f"Found env reads: {result.violations}"
        )

    def test_report_mode_does_not_fail(self) -> None:
        repo_root = Path(__file__).parent.parent.parent
        result = scan_delegation_modules(repo_root=repo_root, mode="report")
        # report mode must always set report_generated = True regardless of violations
        assert result.report_generated

    def test_scan_result_is_dataclass(self) -> None:
        result = ScanResult()
        assert result.scanned_files == 0
        assert result.violations == []
        assert result.report_generated is False

    def test_scanner_ignores_non_delegation_files(self, tmp_path: Path) -> None:
        """Files outside delegation module patterns must not appear in violations."""
        # Write a file with env reads in a non-delegation path
        src = (
            tmp_path / "src" / "mypackage" / "nodes" / "node_registration_orchestrator"
        )
        src.mkdir(parents=True)
        (src / "handler.py").write_text('import os\nos.getenv("KEY")\n')
        # Also create a .git dir so repo_root detection works
        (tmp_path / ".git").mkdir()

        result = scan_delegation_modules(repo_root=tmp_path, mode="report")
        assert result.scanned_files == 0
        assert result.violations == []
        assert result.report_generated

    def test_scanner_ignores_test_files_in_delegation_paths(
        self, tmp_path: Path
    ) -> None:
        """Test files inside delegation module directories must be skipped."""
        src = (
            tmp_path
            / "src"
            / "mypackage"
            / "nodes"
            / "node_delegation_orchestrator"
            / "tests"
        )
        src.mkdir(parents=True)
        (src / "test_handler.py").write_text('import os\nos.getenv("KEY")\n')
        (tmp_path / ".git").mkdir()

        result = scan_delegation_modules(repo_root=tmp_path, mode="report")
        assert result.scanned_files == 0
        assert result.violations == []
