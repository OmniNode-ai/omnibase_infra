# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The Done-write verifier runs as a real subprocess in an unwritable cwd (OMN-25667).

The guard's own argv is built by ``dod_verify_argv``; only the executable behind
``onex skill`` is replaced by a stand-in that does what the CLI does with its
payload: write it under the ``--state-root`` of that argv, then print a verdict
on stdout. The cwd holds a file named ``.onex_state``, so a verifier that keeps
the cwd-relative default dies exactly as the runtime's did.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from omnibase_infra.handlers import done_write_receipt_guard as guard_module

pytestmark = pytest.mark.integration

_STAND_IN = """\
import json, sys
argv = sys.argv[1:]
root = argv[argv.index("--state-root") + 1] if "--state-root" in argv else ".onex_state"
import os
os.makedirs(root, exist_ok=True)
with open(os.path.join(root, "payload.json"), "w", encoding="utf-8") as fh:
    fh.write("{}")
print(json.dumps({"status": "failed", "error_code": "EVIDENCE_CHECK_FAILED"}))
sys.exit(1)
"""


def test_real_subprocess_reaches_a_verdict_in_an_unwritable_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = tmp_path / "stand_in.py"
    script.write_text(_STAND_IN, encoding="utf-8")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / ".onex_state").write_text("a file, not a directory", encoding="utf-8")

    real_argv = guard_module.dod_verify_argv

    def _argv(ticket_id: str, **kwargs: str) -> list[str]:
        real = real_argv(ticket_id, **kwargs)
        return [sys.executable, str(script), *real[real.index("dod_verify") + 1 :]]

    monkeypatch.setattr(guard_module, "dod_verify_argv", _argv)
    result, exit_code, error = asyncio.run(
        guard_module.run_dod_verify_command("OMN-24221", str(cwd), 60.0)
    )

    assert error == ""
    assert exit_code == 1
    assert result is not None
    assert json.loads(json.dumps(result))["error_code"] == "EVIDENCE_CHECK_FAILED"
