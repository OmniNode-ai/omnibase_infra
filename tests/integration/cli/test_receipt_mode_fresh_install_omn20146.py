# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A fresh install has no emit daemon, and a run must say nothing about it (OMN-20146).

End to end through the real click entry point in a child process: no
``--emit-socket``, a home directory that has never held ``.claude/emit.sock``.
That is the state of a machine that has just installed the packages. The run
must succeed with one receipt, print nothing that names a spool, and leave no
outbox directory under the state root.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit.cli.test_cli_node_receipt import (
    _PROOF_NOOP_CONTRACT,
    _write_fixture_inputs,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]

_ENTRY = (
    "import sys;"
    "from omnibase_infra.cli.cli_node import run_node_by_name;"
    "run_node_by_name(sys.argv[1:], standalone_mode=True)"
)


def test_fresh_install_run_prints_no_spool_warning_and_writes_no_outbox(
    tmp_path: Path,
) -> None:
    contract_path, input_path = _write_fixture_inputs(tmp_path, _PROOF_NOOP_CONTRACT)
    home = tmp_path / "home"
    home.mkdir()
    state_root = tmp_path / "state"

    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("ONEX_", "OMNICLAUDE_"))
    }
    env["HOME"] = str(home)
    env["PYTHONPATH"] = str(_REPO_ROOT)
    env["ONEX_ARTIFACT_STORE_ROOT"] = str(tmp_path / "artifacts")

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _ENTRY,
            "proof_noop",
            "--contract",
            str(contract_path),
            "--input",
            str(input_path),
            "--state-root",
            str(state_root),
            "--output",
            "receipt",
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout)
    assert receipt["status"] == "success"
    assert "spool" not in completed.stdout.lower()
    assert "spool" not in completed.stderr.lower()
    assert not (home / ".claude" / "emit.sock").exists()
    assert not (state_root / "emit_spool").exists()
    for capture_file in (state_root / "captures").glob("*.log"):
        assert "spool" not in capture_file.read_text(encoding="utf-8").lower()
