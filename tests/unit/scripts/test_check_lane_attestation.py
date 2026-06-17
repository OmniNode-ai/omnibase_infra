# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for scripts/check_lane_attestation.py (OMN-13034).

Tests pin the attestation gate behavior: files with lane-boundary language
but NO verified: annotation must fail; files with the annotation must pass.
"""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "check_lane_attestation.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("check_lane_attestation", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


def _write_tmp(content: str, suffix: str = ".md") -> Path:
    with tempfile.NamedTemporaryFile(
        suffix=suffix, mode="w", encoding="utf-8", delete=False
    ) as f:
        f.write(content)
        return Path(f.name)


# ---------------------------------------------------------------------------
# Files WITHOUT lane attestation language — always pass
# ---------------------------------------------------------------------------


def test_no_lane_content_passes() -> None:
    """A file with no lane-boundary language must always pass."""
    path = _write_tmp("# Some random markdown\n\nNo lane content here.\n")
    violations = MOD.check_file(path)
    assert violations == []
    path.unlink(missing_ok=True)


def test_empty_file_passes() -> None:
    path = _write_tmp("")
    violations = MOD.check_file(path)
    assert violations == []
    path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Files WITH lane attestation language + verified annotation — pass
# ---------------------------------------------------------------------------


def test_markdown_lane_table_with_verified_passes() -> None:
    """Markdown lane table with verified: annotation must pass."""
    content = (
        "## .201 Server\n\n"
        "verified: 2026-06-17T10:00Z via ssh jonah@192.168.86.201 docker ps\n\n"
        "| Lane | Compose project | Main port |\n"
        "|------|-----------------|----------|\n"
        "| dev | `omnibase-infra` | `8085` |\n"
    )
    path = _write_tmp(content)
    violations = MOD.check_file(path)
    assert violations == []
    path.unlink(missing_ok=True)


def test_lane_manifest_yaml_with_verified_passes() -> None:
    """lane-manifest.yaml with verified: annotation must pass."""
    content = (
        "# verified: 2026-06-17T10:00Z via bash scripts/lane-census-check.sh --json\n"
        "schema_version: '1.0.0'\n"
        "lanes:\n"
        "  prod:\n"
        "    compose_project: omnibase-infra-prod\n"
    )
    path = _write_tmp(content, suffix=".yaml")
    violations = MOD.check_file(path)
    assert violations == []
    path.unlink(missing_ok=True)


def test_generated_block_with_verified_passes() -> None:
    """GENERATED_LANE_TABLE block with verified: annotation must pass."""
    content = (
        "<!-- GENERATED_LANE_TABLE BEGIN\n"
        "     verified: 2026-06-17T10:00Z via lane-census-check.sh on 192.168.86.201\n"
        "-->\n"
        "| Lane | Compose project |\n"
        "|------|----------------|\n"
        "| dev | `omnibase-infra` |\n"
        "<!-- GENERATED_LANE_TABLE END -->\n"
    )
    path = _write_tmp(content)
    violations = MOD.check_file(path)
    assert violations == []
    path.unlink(missing_ok=True)


def test_verified_date_only_passes() -> None:
    """verified: with date-only (no time) must pass — date is sufficient."""
    content = (
        "verified: 2026-06-17 via ssh jonah@192.168.86.201 docker ps\n\n"
        "| dev | `omnibase-infra` | `8085` |\n"
    )
    path = _write_tmp(content)
    violations = MOD.check_file(path)
    assert violations == []
    path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Files WITH lane attestation language but NO verified annotation — fail
# ---------------------------------------------------------------------------


def test_markdown_lane_table_without_verified_fails() -> None:
    """Markdown lane table without verified: annotation must fail."""
    content = (
        "## .201 Server\n\n"
        "| Lane | Compose project | Main port |\n"
        "|------|-----------------|----------|\n"
        "| dev | `omnibase-infra` | `8085` |\n"
        "| prod | `omnibase-infra-prod` | `28085` |\n"
    )
    path = _write_tmp(content)
    violations = MOD.check_file(path)
    assert len(violations) == 1
    assert "verified:" in violations[0]
    assert "OMN-13034" in violations[0]
    path.unlink(missing_ok=True)


def test_compose_project_in_yaml_without_verified_fails() -> None:
    """lane-manifest.yaml with compose_project but no verified: must fail."""
    content = (
        "schema_version: '1.0.0'\n"
        "lanes:\n"
        "  prod:\n"
        "    compose_project: omnibase-infra-prod\n"
        "    network: omnibase-infra-prod-network\n"
    )
    path = _write_tmp(content, suffix=".yaml")
    violations = MOD.check_file(path)
    assert len(violations) == 1
    assert "verified:" in violations[0]
    path.unlink(missing_ok=True)


def test_generated_block_without_verified_fails() -> None:
    """GENERATED_LANE_TABLE block without verified: annotation must fail."""
    content = (
        "<!-- GENERATED_LANE_TABLE BEGIN\n"
        "     generated: 2026-06-17T10:00:00Z\n"
        "-->\n"
        "| Lane | Compose project |\n"
        "|------|----------------|\n"
        "| dev | `omnibase-infra` |\n"
        "<!-- GENERATED_LANE_TABLE END -->\n"
    )
    path = _write_tmp(content)
    violations = MOD.check_file(path)
    assert len(violations) == 1
    path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Exempt paths
# ---------------------------------------------------------------------------


def test_test_file_is_exempt() -> None:
    """Files under tests/ must be exempt from the attestation gate."""
    content = "# test fixture\n| dev | `omnibase-infra` | `8085` |\n"
    # Simulate a path under tests/ by using the _is_exempt helper
    assert MOD._is_exempt("tests/unit/fixtures/some_lane_fixture.md")


def test_workflow_file_is_exempt() -> None:
    assert MOD._is_exempt(".github/workflows/lane-census-staleness.yml")


def test_attestation_script_itself_is_exempt() -> None:
    assert MOD._is_exempt("scripts/check_lane_attestation.py")


# ---------------------------------------------------------------------------
# verified: pattern edge cases
# ---------------------------------------------------------------------------


def test_verified_with_full_iso_timestamp_passes() -> None:
    """Full ISO-8601 datetime in verified: must match."""
    content = (
        "verified: 2026-06-17T09:15:32Z via curl http://192.168.86.201:8085/health\n"
        "| dev | `omnibase-infra` | `8085` |\n"
    )
    path = _write_tmp(content)
    violations = MOD.check_file(path)
    assert violations == []
    path.unlink(missing_ok=True)


def test_verified_case_insensitive() -> None:
    """VERIFIED: (uppercase) must also pass."""
    content = (
        "VERIFIED: 2026-06-17 via ssh jonah@192.168.86.201 'docker ps'\n"
        "| dev | `omnibase-infra` | `8085` |\n"
    )
    path = _write_tmp(content)
    violations = MOD.check_file(path)
    assert violations == []
    path.unlink(missing_ok=True)


def test_verified_without_via_command_fails() -> None:
    """verified: with no 'via <command>' part must fail — command is required."""
    content = "verified: 2026-06-17\n| dev | `omnibase-infra` | `8085` |\n"
    path = _write_tmp(content)
    violations = MOD.check_file(path)
    assert len(violations) == 1
    path.unlink(missing_ok=True)
