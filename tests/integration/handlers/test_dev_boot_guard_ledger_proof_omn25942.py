# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration: the dev boot guard's post-roll ledger proof runs a real subprocess (OMN-25942).

The declaration is the committed ``work_ledger_query_client.contract.yaml``; the
``scripts/onex`` entry is a stand-in executable in a scratch checkout, so the proof
spawns a real process and reads its real exit code and stdout.
"""

from __future__ import annotations

import subprocess
import types
from pathlib import Path

import pytest

from omnibase_infra.handlers import handler_dev_boot_guard as guard

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]


def _checkout(tmp_path: Path, script_body: str) -> types.SimpleNamespace:
    contract = tmp_path / guard.WORK_LEDGER_CONTRACT
    contract.parent.mkdir(parents=True)
    contract.write_text(
        (REPO_ROOT / guard.WORK_LEDGER_CONTRACT).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    entry = tmp_path / guard.ONEX_ENTRY
    entry.parent.mkdir(parents=True)
    entry.write_text("#!/bin/sh\n" + script_body, encoding="utf-8")
    entry.chmod(0o755)

    def run(
        command: list[str], timeout: int, *, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )

    return types.SimpleNamespace(
        REPO_DIR=str(tmp_path), _run=run, subprocess=subprocess
    )


def test_answering_ledger_passes(tmp_path: Path) -> None:
    execution = _checkout(tmp_path, "echo '{\"rows\": []}'\n")
    guard._prove_work_ledger_answers(execution)


def test_failing_ledger_fails_the_roll(tmp_path: Path) -> None:
    execution = _checkout(tmp_path, "echo boom >&2\nexit 3\n")
    with pytest.raises(guard.DevBootFailedError, match="exited 3"):
        guard._prove_work_ledger_answers(execution)


def test_non_object_answer_fails_the_roll(tmp_path: Path) -> None:
    execution = _checkout(tmp_path, "echo not-json\n")
    with pytest.raises(guard.DevBootFailedError, match="not one JSON object"):
        guard._prove_work_ledger_answers(execution)


def test_failing_ledger_message_names_the_exception_last_on_stderr(
    tmp_path: Path,
) -> None:
    """A real process whose stderr opens with a long warning still has its exception named (OMN-26316)."""
    warning = "w" * 600
    execution = _checkout(
        tmp_path,
        f"echo 'RuntimeWarning: {warning}' >&2\n"
        "echo 'ProtocolConfigurationError: kafka config invalid' >&2\n"
        "exit 1\n",
    )
    with pytest.raises(guard.DevBootFailedError, match="ProtocolConfigurationError"):
        guard._prove_work_ledger_answers(execution)
