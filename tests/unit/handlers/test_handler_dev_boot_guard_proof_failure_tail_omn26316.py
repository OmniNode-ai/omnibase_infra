# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev boot guard's failed ledger proof names the exception, not the warning above it (OMN-26316).

The deploy job records of dev jobs ae9ceb3d, 43b9fb72 and b571579b (2026-11-23 and
2026-11-24, omni-201) each end at the same character: the first 240 characters of the
proof's stderr, which are the ``runpy`` RuntimeWarning the ``onex`` entry always
prints, cut mid-word at ``unpredictabl``. The exception that ended the proof comes
last in a Python stderr, so the job record never carried it and the cause of three
restored rolls stayed unread until a lane reproduced the proof by hand.
"""

from __future__ import annotations

import subprocess
import types
from pathlib import Path

import pytest

from omnibase_infra.handlers import handler_dev_boot_guard as guard

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]

_RUNPY_WARNING = (
    "<frozen runpy>:128: RuntimeWarning: 'omnibase_core.cli.cli_commands' found in "
    "sys.modules after import of package 'omnibase_core.cli', but prior to execution "
    "of 'omnibase_core.cli.cli_commands'; this may result in unpredictable behaviour"
)
_EXCEPTION_LINE = (
    "ProtocolConfigurationError: the lane's kafka client config is invalid"
)


def _traceback(frames: int) -> str:
    stack = "".join(
        f'  File "/srv/agent/omnibase_infra/module_{index}.py", line {index}, in step_{index}\n'
        f"    result = step_{index + 1}(context)\n"
        for index in range(frames)
    )
    return f"{_RUNPY_WARNING}\nTraceback (most recent call last):\n{stack}{_EXCEPTION_LINE}\n"


def _execution(tmp_path: Path, stderr: str) -> types.SimpleNamespace:
    contract = tmp_path / guard.WORK_LEDGER_CONTRACT
    contract.parent.mkdir(parents=True)
    contract.write_text(
        (REPO_ROOT / guard.WORK_LEDGER_CONTRACT).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    def run(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, stdout="", stderr=stderr)

    return types.SimpleNamespace(
        REPO_DIR=str(tmp_path), _run=run, subprocess=subprocess
    )


@pytest.mark.parametrize(
    "frames", [2, 40], ids=["short_stack", "stack_past_2000_chars"]
)
def test_failed_proof_names_the_exception_that_ended_it(
    tmp_path: Path, frames: int
) -> None:
    stderr = _traceback(frames)
    assert stderr.index(_EXCEPTION_LINE) > 240, "the warning must fill the old window"
    execution = _execution(tmp_path, stderr)

    with pytest.raises(guard.DevBootFailedError) as raised:
        guard._prove_work_ledger_answers(execution)

    assert "ProtocolConfigurationError" in str(raised.value)


def test_failed_proof_message_stays_inside_its_bound(tmp_path: Path) -> None:
    execution = _execution(tmp_path, _traceback(40))

    with pytest.raises(guard.DevBootFailedError) as raised:
        guard._prove_work_ledger_answers(execution)

    assert len(str(raised.value)) <= len("work ledger: ledger exited 1: ") + 241


def test_short_stderr_is_reported_whole(tmp_path: Path) -> None:
    execution = _execution(tmp_path, "boom\n")

    with pytest.raises(guard.DevBootFailedError, match=r"exited 1: .*boom$"):
        guard._prove_work_ledger_answers(execution)
