# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""End-to-end: the real ``scripts/ledger_lock.py`` refuses a test-run write (OMN-19513).

Runs the script as a subprocess, as a lane shell would, with the test-runner signal inherited.
The ledger sits beside the child's temp root, so it is canonical to the guard, and every byte
this test could write, even on a regression, stays under ``tmp_path``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from omnibase_infra.handlers import handler_ledger_write_guard as guard

pytestmark = pytest.mark.integration

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ledger_lock.py"


def _row() -> str:
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{stamp} | STATUS | lane=alpha | ticket=OMN-1 | a fixture row"


def test_the_script_refuses_a_canonical_append_from_a_test_subprocess(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    ledger = tmp_path / "ROLLING_WORK_LEDGER.md"
    ledger.write_text("## Work ledger\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
    env["ONEX_TEST_CONTEXT"] = "pytest"
    env["TMPDIR"] = str(scratch)
    env.pop("OMNI_HOME", None)
    before = ledger.read_bytes()
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), str(ledger), "--append", _row()],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == guard.EXIT_TEST_WRITE_REFUSED, proc.stderr
    assert guard.GUARD_NAME in proc.stderr
    assert ledger.read_bytes() == before
