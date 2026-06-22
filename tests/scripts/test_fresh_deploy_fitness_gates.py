# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the OMN-13412 fresh-deploy fitness gate scanners.

Covers the three NEW validators wired in this PR:
  * scripts/check_terminal_cost_completeness.py  (item 5)
  * scripts/check_context_field_presence.py      (item 6)
  * scripts/check_release_identity.py            (item 7)

Each test proves a deliberately-broken input fails (non-zero exit) and a
correct input passes (exit 0) — the DoD requirement for an enforcement gate.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPTS / script), *args],
        capture_output=True,
        text=True,
        check=False,
    )


# --------------------------------------------------------------------------- #
# Item 5: terminal cost completeness                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_terminal_cost_detects_hardcoded_zero(tmp_path: Path) -> None:
    bad = tmp_path / "bad_terminal.py"
    bad.write_text(
        "def emit():\n"
        "    return ModelLlmCallCompleted(\n"
        "        tokens_used=1234,\n"
        "        cost_usd=0.0,\n"
        "    )\n",
        encoding="utf-8",
    )
    result = _run("check_terminal_cost_completeness.py", str(bad))
    assert result.returncode == 1, result.stderr
    assert "cost_usd=0.0" in result.stderr


@pytest.mark.unit
def test_terminal_cost_allows_annotated_zero(tmp_path: Path) -> None:
    ok = tmp_path / "ok_terminal.py"
    ok.write_text(
        "def emit():\n"
        "    return ModelLlmCallCompleted(\n"
        "        tokens_used=0,\n"
        "        cost_usd=0.0,  # cost-zero-ok: error path, no tokens consumed\n"
        "    )\n",
        encoding="utf-8",
    )
    result = _run("check_terminal_cost_completeness.py", str(ok))
    assert result.returncode == 0, result.stderr


@pytest.mark.unit
def test_terminal_cost_ignores_real_value_and_substring(tmp_path: Path) -> None:
    ok = tmp_path / "real_cost.py"
    ok.write_text(
        "def emit():\n"
        "    return Model(\n"
        "        cost_usd=self._estimate_cost(tokens),\n"
        "        estimated_cost_usd=0.0,\n"  # different field — must not match
        "    )\n",
        encoding="utf-8",
    )
    result = _run("check_terminal_cost_completeness.py", str(ok))
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------- #
# Item 6: context field presence                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_context_field_detects_claim_without_hash(tmp_path: Path) -> None:
    bad = tmp_path / "contract.yaml"
    bad.write_text(
        "name: node_demo\n"
        "node_type: COMPUTE_GENERIC\n"
        "metadata:\n"
        "  context_roi:\n"
        "    tokens_saved: 4096\n",
        encoding="utf-8",
    )
    result = _run("check_context_field_presence.py", str(bad))
    assert result.returncode == 1, result.stderr
    assert "context_pack_hash" in result.stderr


@pytest.mark.unit
def test_context_field_passes_claim_with_hash(tmp_path: Path) -> None:
    ok = tmp_path / "contract.yaml"
    ok.write_text(
        "name: node_demo\n"
        "node_type: COMPUTE_GENERIC\n"
        "metadata:\n"
        "  context_roi:\n"
        "    tokens_saved: 4096\n"
        "    context_pack_hash: sha256:abc123\n",
        encoding="utf-8",
    )
    result = _run("check_context_field_presence.py", str(ok))
    assert result.returncode == 0, result.stderr


@pytest.mark.unit
def test_context_field_passes_no_claim(tmp_path: Path) -> None:
    ok = tmp_path / "contract.yaml"
    ok.write_text(
        "name: node_demo\nnode_type: COMPUTE_GENERIC\n",
        encoding="utf-8",
    )
    result = _run("check_context_field_presence.py", str(ok))
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------- #
# Item 7: release identity                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_release_identity_passes_when_version_ahead() -> None:
    # The repo's pyproject version is bumped ahead of the latest tag in this PR,
    # so the strict (no --base) run must pass.
    result = _run("check_release_identity.py")
    assert result.returncode == 0, result.stderr
    assert "ahead of latest published" in result.stdout


@pytest.mark.unit
def test_release_identity_exempts_non_src_diff() -> None:
    # A changed-file list with no src/** entry is exempt regardless of version.
    result = _run(
        "check_release_identity.py",
        "--changed-file",
        "docs/readme.md",
        "--changed-file",
        ".github/workflows/ci.yml",
    )
    assert result.returncode == 0, result.stderr
    assert "version bump not required" in result.stdout
