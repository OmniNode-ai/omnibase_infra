# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the LLM endpoint env contract guard."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "check_llm_endpoint_env_contract.py"


def _run_check(env_file: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python", str(_SCRIPT), "--env-file", str(env_file)],
        cwd=_REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )


@pytest.mark.unit
def test_rejects_disabled_embedding_endpoint(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_CODER_URL=http://192.168.86.201:8000",
                "LLM_CODER_FAST_URL=http://192.168.86.201:8001",
                "LLM_EMBEDDING_URL=http://192.168.86.200:8100",
                "LLM_DEEPSEEK_R1_URL=http://192.168.86.200:8101",
            ]
        )
    )

    result = _run_check(env_file)

    assert result.returncode == 1
    assert "LLM_EMBEDDING_URL=http://192.168.86.200:8100" in result.stderr
    assert "status='disabled'" in result.stderr


@pytest.mark.unit
def test_accepts_running_embedding_endpoint(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_CODER_URL=http://192.168.86.201:8000",
                "LLM_CODER_FAST_URL=http://192.168.86.201:8001",
                "LLM_EMBEDDING_URL=http://192.168.86.201:8100",
                "LLM_DEEPSEEK_R1_URL=http://192.168.86.200:8101",
            ]
        )
    )

    result = _run_check(env_file)

    assert result.returncode == 0
    assert "passed" in result.stdout
