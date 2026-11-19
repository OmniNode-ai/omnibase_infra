# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The Done-write verifier reaches a verdict in the runtime's cwd (OMN-25667).

Recorded failure (dev runtime, omnibase_infra 75969ce9e2ca, 2026-11-18): every
Done write was refused with ``dod_verify reached no output ... exit_code=1`` and
the refusal text ended at ``Traceback (most recent call last):``. The verifier's
own stderr (fixture) shows ``onex skill`` dying on ``PermissionError: Permission
denied: '.onex_state'``: it writes its payload under the cwd-relative default
``--state-root``, and the runtime's cwd (``/app``) is not writable by its user.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_infra.cli import cli_skill
from omnibase_infra.cli.cli_skill import run_skill_by_name
from omnibase_infra.handlers import done_write_receipt_guard as guard_module
from omnibase_infra.handlers.done_write_receipt_guard import (
    MAX_VERIFIER_ERROR_CHARS,
    DoneWriteReceiptGuard,
    bound_verifier_error,
    dod_verify_argv,
)

pytestmark = pytest.mark.unit

_RECORDED_STDERR = (
    (
        Path(__file__).parent
        / "fixtures"
        / "dod_verify_stderr_state_root_permission_denied.txt"
    )
    .read_text(encoding="utf-8")
    .strip()
)


def test_dod_verify_argv_names_the_state_root_it_is_given() -> None:
    argv = dod_verify_argv("OMN-24221", state_root="/scratch/dod")
    assert argv[argv.index("--state-root") + 1] == "/scratch/dod"


def test_dod_verify_argv_keeps_the_cli_default_without_a_state_root() -> None:
    assert "--state-root" not in dod_verify_argv("OMN-24221")


def test_verifier_invocation_survives_a_cwd_it_cannot_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The guard's own invocation, through the real ``onex skill`` command, in a
    cwd where ``.onex_state`` cannot be created (a file holds the name), reaches
    the dispatch. Without a state root in the argv the command dies on the
    payload write, as the runtime's verifier did."""
    reached: list[Path] = []
    argvs: list[list[str]] = []

    def _fake_receipt_mode(**kwargs: object) -> int:
        input_path = kwargs["input_path"]
        assert isinstance(input_path, Path)
        reached.append(input_path)
        return 0

    monkeypatch.setattr(cli_skill, "run_receipt_mode", _fake_receipt_mode)
    monkeypatch.setattr(cli_skill, "check_omnimarket_drift", lambda **_: None)
    monkeypatch.setattr(
        cli_skill, "_resolve_packaged_contract", lambda name: tmp_path / "c.yaml"
    )
    monkeypatch.delenv("OMNI_HOME", raising=False)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / ".onex_state").write_text("not a directory", encoding="utf-8")
    monkeypatch.chdir(cwd)

    real_argv = guard_module.dod_verify_argv

    def _recording_argv(ticket_id: str, **kwargs: str) -> list[str]:
        argv = real_argv(ticket_id, **kwargs)
        argvs.append(argv)
        return [sys.executable, "-c", "import sys; sys.exit(0)"]

    monkeypatch.setattr(guard_module, "dod_verify_argv", _recording_argv)
    asyncio.run(guard_module.run_dod_verify_command("OMN-24221", str(cwd), 30.0))

    assert len(argvs) == 1
    result = CliRunner().invoke(run_skill_by_name, argvs[0][2:])
    assert result.exit_code == 0, repr(result.exception)
    assert len(reached) == 1


def _stderr_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, text: str) -> None:
    """Make the guard's subprocess a script that prints ``text`` to stderr and exits 1."""
    script = tmp_path / "emit_stderr.py"
    script.write_text(
        f"import sys\nsys.stderr.write({text!r})\nsys.exit(1)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        guard_module,
        "dod_verify_argv",
        lambda ticket_id, **_kwargs: [sys.executable, str(script)],
    )


def test_refusal_carries_the_recorded_verifier_stderr_in_full(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stderr_runner(monkeypatch, tmp_path, _RECORDED_STDERR)
    decision = asyncio.run(
        DoneWriteReceiptGuard().decide(ticket_id="OMN-24221", description="")
    )
    assert decision.allowed is False
    assert "(exit_code=1)" in decision.reason
    assert _RECORDED_STDERR in decision.reason
    assert decision.reason.endswith(
        "PermissionError: [Errno 13] Permission denied: '.onex_state'"
    )


def test_over_long_verifier_stderr_keeps_its_tail_and_says_what_it_dropped() -> None:
    text = "x" * 5000 + "PermissionError: the cause"
    bounded = bound_verifier_error(text)
    assert bounded.endswith("PermissionError: the cause")
    omitted = len(text) - MAX_VERIFIER_ERROR_CHARS
    assert bounded.startswith(f"[{omitted} chars omitted]")
    assert bound_verifier_error("short") == "short"


def test_verifier_scratch_state_root_exists_during_the_run_and_is_removed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[str] = []
    script = tmp_path / "probe.py"
    script.write_text(
        "import os, sys\n"
        "sys.stderr.write('state_root_exists=' + str(os.path.isdir(sys.argv[1])))\n"
        "sys.exit(1)\n",
        encoding="utf-8",
    )

    def _argv(ticket_id: str, *, state_root: str = "", **_kwargs: str) -> list[str]:
        seen.append(state_root)
        return [sys.executable, str(script), state_root]

    monkeypatch.setattr(guard_module, "dod_verify_argv", _argv)
    _, exit_code, error = asyncio.run(
        guard_module.run_dod_verify_command("OMN-24221", "", 30.0)
    )
    assert exit_code == 1
    assert error == "state_root_exists=True"
    assert len(seen) == 1
    assert not Path(seen[0]).exists()
