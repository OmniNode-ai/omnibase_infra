#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the delegation env read scanner (OMN-10926).

Verifies enforce mode catches new violations and respects ONEX_FLAG_EXEMPT /
ONEX_EXCLUDE annotations, including on the closing ) of formatter-wrapped calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_CI = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPTS_CI))

from check_delegation_env_reads import (
    ScanResult,
    _find_env_calls_in_source,
    _is_allowlisted,
    _is_delegation_module,
    scan_delegation_modules,
)


@pytest.mark.unit
class TestIsDelegationModule:
    def test_adapters_llm(self) -> None:
        assert _is_delegation_module(
            "src/omnibase_infra/adapters/llm/adapter_llm_provider_openai.py"
        )

    def test_node_llm_inference(self) -> None:
        assert _is_delegation_module(
            "src/omnibase_infra/nodes/node_llm_inference_effect/handlers/bifrost/config_loader_bifrost.py"
        )

    def test_delegation_orchestrator(self) -> None:
        assert _is_delegation_module(
            "src/omnibase_infra/nodes/node_delegation_orchestrator/handlers/h.py"
        )

    def test_non_delegation_excluded(self) -> None:
        assert not _is_delegation_module("src/omnibase_infra/runtime/service_kernel.py")


@pytest.mark.unit
class TestIsAllowlisted:
    def test_test_file_allowlisted(self) -> None:
        assert _is_allowlisted("tests/unit/delegation/test_foo.py")

    def test_conftest_allowlisted(self) -> None:
        assert _is_allowlisted("tests/conftest.py")

    def test_handler_not_allowlisted(self) -> None:
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

    def test_skips_onex_flag_exempt_same_line(self) -> None:
        source = (
            'import os\nvalue = os.environ.get("KEY", "")  # ONEX_FLAG_EXEMPT: test\n'
        )
        violations = _find_env_calls_in_source(source, "test.py")
        assert len(violations) == 0

    def test_skips_onex_exclude_on_closing_paren(self) -> None:
        source = (
            "import os\n"
            "value = os.environ.get(\n"
            '    "KEY", ""\n'
            ")  # ONEX_EXCLUDE: migration tracked\n"
        )
        violations = _find_env_calls_in_source(source, "test.py")
        assert len(violations) == 0

    def test_skips_onex_flag_exempt_on_closing_paren(self) -> None:
        source = (
            "import os\n"
            "value = os.environ.get(\n"
            '    "KEY", ""\n'
            ")  # ONEX_FLAG_EXEMPT: Wave 3 migration (OMN-10915)\n"
        )
        violations = _find_env_calls_in_source(source, "test.py")
        assert len(violations) == 0


@pytest.mark.unit
class TestScanDelegationModules:
    def test_scanner_clean_in_enforce_mode(self) -> None:
        """After OMN-10926, enforce mode must return zero violations on current code."""
        repo_root = Path(__file__).parent.parent.parent
        result = scan_delegation_modules(repo_root=repo_root, mode="enforce")
        assert len(result.violations) == 0, (
            "Unexpected env reads in delegation modules — add ONEX_FLAG_EXEMPT or "
            "replace with contract-driven config (OMN-10915):\n"
            + "\n".join(result.violations)
        )

    def test_report_mode_does_not_fail(self) -> None:
        repo_root = Path(__file__).parent.parent.parent
        result = scan_delegation_modules(repo_root=repo_root, mode="report")
        assert result.report_generated

    def test_scan_result_dataclass(self) -> None:
        result = ScanResult()
        assert result.scanned_files == 0
        assert result.violations == []
        assert result.report_generated is False

    def test_enforce_mode_catches_new_violation(self, tmp_path: Path) -> None:
        """Enforce mode exits non-zero when a bare env read appears in a delegation module."""
        src = tmp_path / "src" / "omnibase_infra" / "adapters" / "llm"
        src.mkdir(parents=True)
        (src / "new_adapter.py").write_text(
            'import os\nendpoint = os.environ.get("LLM_URL", "")\n'
        )
        (tmp_path / ".git").mkdir()

        result = scan_delegation_modules(repo_root=tmp_path, mode="enforce")
        assert len(result.violations) == 1
        assert "os.environ.get" in result.violations[0]

    def test_enforce_mode_passes_when_exempt(self, tmp_path: Path) -> None:
        """Enforce mode passes when ONEX_FLAG_EXEMPT is on the violation line."""
        src = tmp_path / "src" / "omnibase_infra" / "adapters" / "llm"
        src.mkdir(parents=True)
        (src / "new_adapter.py").write_text(
            "import os\n"
            'endpoint = os.environ.get("LLM_URL", "")  # ONEX_FLAG_EXEMPT: Wave 3\n'
        )
        (tmp_path / ".git").mkdir()

        result = scan_delegation_modules(repo_root=tmp_path, mode="enforce")
        assert len(result.violations) == 0

    def test_scanner_ignores_non_llm_adapters(self, tmp_path: Path) -> None:
        src = tmp_path / "src" / "omnibase_infra" / "runtime"
        src.mkdir(parents=True)
        (src / "service_kernel.py").write_text('import os\nos.environ.get("KEY")\n')
        (tmp_path / ".git").mkdir()

        result = scan_delegation_modules(repo_root=tmp_path, mode="enforce")
        assert result.scanned_files == 0
        assert result.violations == []
