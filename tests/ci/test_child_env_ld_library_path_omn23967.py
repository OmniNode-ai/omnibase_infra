# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Child processes keep the session's LD_LIBRARY_PATH (OMN-23967).

Many tests build a child environment from scratch (``env={"PATH": ...}``)
and so drop ``LD_LIBRARY_PATH``. On runners whose venv python is a
shared-libpython CPython the library is found only through that variable, so
such a child dies with exit 127 (``error while loading shared libraries``).
These tests pin the ``Popen`` wrapper installed by ``tests/conftest.py`` that
copies the session value into a scrubbed child environment, and that the
caller's mapping is never mutated.

``SESSION_LD_LIBRARY_PATH`` is the value the wrapper hands out. It is captured
once at session start, so each test sets it directly (and the process
environment alongside it, as a real session would hold it) instead of relying
on whatever the runner exports.
"""

from __future__ import annotations

import ast
import asyncio
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any

import pytest
import yaml

import tests.conftest as session_conftest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONTRACT = _REPO_ROOT / "contracts" / "OMN-23967.yaml"
_SH = "/bin/sh"
#: Prints "set" only when the variable exists, empty or not, then its value.
_PROBE_SET = 'printf %s "${LD_LIBRARY_PATH+set}"'
_PROBE_VALUE = 'printf %s "$LD_LIBRARY_PATH"'
_SENTINEL = "OMN23967_SENTINEL"
_PROBE_SENTINEL = f'printf %s "$LD_LIBRARY_PATH|${_SENTINEL}"'


def _session_value(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Make ``value`` the session's LD_LIBRARY_PATH, as the wrapper sees it."""
    if value:
        monkeypatch.setenv("LD_LIBRARY_PATH", value)
    else:
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    # raising=False: against a conftest without the wrapper the attribute does
    # not exist, and the tests must then fail on what the child sees.
    monkeypatch.setattr(
        session_conftest, "SESSION_LD_LIBRARY_PATH", value, raising=False
    )


def _run(probe: str, env: Any) -> str:
    return subprocess.run(
        [_SH, "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _popen_positional_env(probe: str, env: Any) -> str:
    # Popen(args, bufsize, executable, stdin, stdout, stderr, preexec_fn,
    #       close_fds, shell, cwd, env); a new session keeps any descendant
    #       out of the test's process group (OMN-16995 spawn audit).
    with subprocess.Popen(
        [_SH, "-c", probe],
        -1,
        None,
        None,
        subprocess.PIPE,
        None,
        None,
        True,
        False,
        None,
        env,
        text=True,
        start_new_session=True,
    ) as proc:
        stdout, _ = proc.communicate()
    return stdout


def test_scrubbed_env_child_inherits_session_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _session_value(monkeypatch, "/session/lib")
    assert _run(_PROBE_VALUE, {"PATH": os.environ["PATH"]}) == "/session/lib"


def test_explicit_value_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    _session_value(monkeypatch, "/session/lib")
    env = {"PATH": os.environ["PATH"], "LD_LIBRARY_PATH": "/caller/lib"}
    assert _run(_PROBE_VALUE, env) == "/caller/lib"


def test_explicit_empty_value_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    _session_value(monkeypatch, "/session/lib")
    env = {"PATH": os.environ["PATH"], "LD_LIBRARY_PATH": ""}
    # The variable must still exist in the child, with an empty value.
    assert _run(_PROBE_SET, env) == "set"
    assert _run(_PROBE_VALUE, env) == ""


def test_no_session_value_adds_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _session_value(monkeypatch, "")
    # "${VAR+set}" is empty only when the variable is unset: a wrapper that
    # injected LD_LIBRARY_PATH="" would print "set" and fail here.
    assert _run(_PROBE_SET, {"PATH": os.environ["PATH"]}) == ""


def test_session_value_is_captured_once_not_read_at_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _session_value(monkeypatch, "/session/lib")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    assert _run(_PROBE_VALUE, {"PATH": os.environ["PATH"]}) == "/session/lib"


def test_callers_mapping_is_not_mutated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _session_value(monkeypatch, "/session/lib")
    caller_env = {"PATH": os.environ["PATH"]}
    assert _run(_PROBE_VALUE, caller_env) == "/session/lib"
    assert "LD_LIBRARY_PATH" not in caller_env


def test_callers_mapping_is_not_mutated_on_the_positional_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _session_value(monkeypatch, "/session/lib")
    caller_env = {"PATH": os.environ["PATH"]}
    assert _popen_positional_env(_PROBE_VALUE, caller_env) == "/session/lib"
    assert "LD_LIBRARY_PATH" not in caller_env


def test_env_none_inherits_whole_environment_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _session_value(monkeypatch, "/session/lib")
    monkeypatch.setenv(_SENTINEL, "present")
    # env=None must hand the child the whole parent environment: the sentinel
    # reaches it too. A wrapper that swapped env=None for a mapping holding
    # only LD_LIBRARY_PATH would drop the sentinel and fail here.
    assert _run(_PROBE_SENTINEL, None) == "/session/lib|present"


def test_positional_env_argument_is_covered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _session_value(monkeypatch, "/session/lib")
    assert (
        _popen_positional_env(_PROBE_VALUE, {"PATH": os.environ["PATH"]})
        == "/session/lib"
    )


def test_bytes_keyed_env_gets_one_bytes_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _session_value(monkeypatch, "/session/lib")
    env = {b"PATH": os.fsencode(os.environ["PATH"])}
    assert _run(_PROBE_VALUE, env) == "/session/lib"
    assert b"LD_LIBRARY_PATH" not in env


def test_bytes_keyed_explicit_value_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _session_value(monkeypatch, "/session/lib")
    env = {
        b"PATH": os.fsencode(os.environ["PATH"]),
        b"LD_LIBRARY_PATH": b"/caller/lib",
    }
    # A str-keyed duplicate added beside the bytes key would win or lose by
    # dict order; the child must see exactly the caller's value.
    probe = 'env | grep -c "^LD_LIBRARY_PATH="; printf %s "$LD_LIBRARY_PATH"'
    assert _run(probe, env) == "1\n/caller/lib"


@pytest.mark.asyncio
async def test_asyncio_subprocess_scrubbed_env_inherits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _session_value(monkeypatch, "/session/lib")
    proc = await asyncio.create_subprocess_exec(
        _SH,
        "-c",
        _PROBE_VALUE,
        env={"PATH": os.environ["PATH"]},
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    assert proc.returncode == 0
    assert stdout.decode() == "/session/lib"


def test_scrubbed_env_child_interpreter_starts() -> None:
    """The venv interpreter starts under a scrubbed env on this runner.

    This test only discriminates on a shared-libpython CPython (the Actions
    tool cache build); on a statically linked or uv-managed interpreter it
    passes with or without the wrapper, so the build is named in the message.
    """
    shared = sysconfig.get_config_var("Py_ENABLE_SHARED")
    result = subprocess.run(
        [sys.executable, "-c", "print(1)"],
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Py_ENABLE_SHARED={shared} stderr={result.stderr!r}"
    assert result.stdout.strip() == "1"


#: Spawn APIs the ``Popen`` wrapper does not reach (see conftest).
_UNWRAPPED_SPAWNS = frozenset(
    {
        "posix_spawn",
        "posix_spawnp",
        "execve",
        "execvpe",
        "execle",
        "execlpe",
        "spawnve",
        "spawnvpe",
        "spawnle",
        "spawnlpe",
    }
)


def test_no_test_spawns_outside_the_popen_wrapper() -> None:
    """A spawn the wrapper cannot reach must be decided, not slipped in."""
    offenders: dict[str, list[int]] = {}
    for path in sorted((_REPO_ROOT / "tests").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr in _UNWRAPPED_SPAWNS
                and isinstance(node.value, ast.Name)
                and node.value.id == "os"
            ):
                rel = path.relative_to(_REPO_ROOT).as_posix()
                offenders.setdefault(rel, []).append(node.lineno)
    assert offenders == {}, (
        "os.posix_spawn/exec*e/spawn*e bypass the LD_LIBRARY_PATH wrapper in "
        "tests/conftest.py (OMN-23967); use subprocess, or extend the wrapper "
        f"and this guard: {offenders}"
    )


def _local_done_gate_scopes(contract: dict[str, Any]) -> list[str]:
    """Locations in ``dod_evidence`` scoped to the local done gate.

    ``execution_scope`` is read on the item and on each of its checks, so a
    scope moved down to a check cannot hide from the guard.
    """
    found: list[str] = []
    for item in contract["dod_evidence"]:
        if item.get("execution_scope") == "local_done_gate":
            found.append(item["id"])
        for index, check in enumerate(item.get("checks", [])):
            if check.get("execution_scope") == "local_done_gate":
                found.append(f"{item['id']}.checks[{index}]")
    return found


def test_repo_evidence_dod_verify_contract_binds_only_hosted_items() -> None:
    """No item of this test-only change is scoped to the local done gate.

    The fix lives in tests/conftest.py, so the hosted repo-evidence / dod-verify
    control overlays it onto the merge base and the bound tests pass there. A
    test-only diff is accepted only when every bound item is judged at the
    head; an item scoped to ``local_done_gate`` is never evaluated by the hosted
    job, so it makes the control refuse the whole ticket (always-pass).
    """
    contract = yaml.safe_load(_CONTRACT.read_text())
    assert contract["dod_evidence"]
    assert _local_done_gate_scopes(contract) == []


def test_execution_scope_guard_reads_item_and_check_level() -> None:
    item_level = {
        "dod_evidence": [{"id": "dod-a", "execution_scope": "local_done_gate"}]
    }
    check_level = {
        "dod_evidence": [
            {
                "id": "dod-b",
                "checks": [
                    {"check_type": "command"},
                    {"check_type": "command", "execution_scope": "local_done_gate"},
                ],
            }
        ]
    }
    hosted = {
        "dod_evidence": [
            {
                "id": "dod-c",
                "execution_scope": "hosted_and_local",
                "checks": [{"check_type": "command"}],
            }
        ]
    }
    assert _local_done_gate_scopes(item_level) == ["dod-a"]
    assert _local_done_gate_scopes(check_level) == ["dod-b.checks[1]"]
    assert _local_done_gate_scopes(hosted) == []
