# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Child processes keep the session's LD_LIBRARY_PATH (OMN-23967).

Many tests build a child environment from scratch (``env={"PATH": ...}``)
and so drop ``LD_LIBRARY_PATH``. On runners whose venv python is a
shared-libpython CPython the library is found only through that variable, so
such a child dies with exit 127 (``error while loading shared libraries``).
These tests pin the autouse fixture that copies the session value into a
scrubbed child environment, and that the caller's mapping is never mutated.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_SH = "/bin/sh"
_PROBE = 'printf %s "$LD_LIBRARY_PATH"'
_CONTRACT = Path(__file__).resolve().parents[2] / "contracts" / "OMN-23967.yaml"


def test_scrubbed_env_child_inherits_session_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/session/lib")
    result = subprocess.run(
        [_SH, "-c", _PROBE],
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == "/session/lib"


def test_explicit_value_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/session/lib")
    result = subprocess.run(
        [_SH, "-c", _PROBE],
        env={"PATH": os.environ["PATH"], "LD_LIBRARY_PATH": "/caller/lib"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == "/caller/lib"


def test_explicit_empty_value_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/session/lib")
    result = subprocess.run(
        [_SH, "-c", _PROBE],
        env={"PATH": os.environ["PATH"], "LD_LIBRARY_PATH": ""},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == ""


def test_no_session_value_adds_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    result = subprocess.run(
        [_SH, "-c", _PROBE],
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == ""


def test_callers_mapping_is_not_mutated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/session/lib")
    caller_env = {"PATH": os.environ["PATH"]}
    subprocess.run(
        [_SH, "-c", _PROBE],
        env=caller_env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "LD_LIBRARY_PATH" not in caller_env


def test_env_none_inherits_whole_environment_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/session/lib")
    result = subprocess.run(
        [_SH, "-c", _PROBE],
        env=None,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == "/session/lib"


def test_positional_env_argument_is_covered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/session/lib")
    # Popen(args, bufsize, executable, stdin, stdout, stderr, preexec_fn,
    #       close_fds, shell, cwd, env)
    with subprocess.Popen(
        [_SH, "-c", _PROBE],
        -1,
        None,
        None,
        subprocess.PIPE,
        None,
        None,
        True,
        False,
        None,
        {"PATH": os.environ["PATH"]},
        text=True,
    ) as proc:
        stdout, _ = proc.communicate()
    assert stdout == "/session/lib"


async def test_asyncio_subprocess_scrubbed_env_inherits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/session/lib")
    proc = await asyncio.create_subprocess_exec(
        _SH,
        "-c",
        _PROBE,
        env={"PATH": os.environ["PATH"]},
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    assert proc.returncode == 0
    assert stdout.decode() == "/session/lib"


def test_scrubbed_env_child_interpreter_starts() -> None:
    """The venv interpreter starts under a scrubbed env on this runner."""
    result = subprocess.run(
        [sys.executable, "-c", "print(1)"],
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "1"


def test_repo_evidence_dod_verify_contract_binds_only_hosted_items() -> None:
    """No item of this test-only change is scoped to the local done gate.

    The fix lives in tests/conftest.py, so the hosted repo-evidence / dod-verify
    control overlays it onto the merge base and the bound tests pass there. A
    test-only diff is accepted only when every bound item is judged at the
    head; an item scoped to ``local_done_gate`` is never evaluated by the hosted
    job, so it makes the control refuse the whole ticket (always-pass).
    """
    contract = yaml.safe_load(_CONTRACT.read_text())
    scopes = {
        item["id"]: item.get("execution_scope", "default")
        for item in contract["dod_evidence"]
    }
    assert scopes
    assert "local_done_gate" not in scopes.values(), scopes
