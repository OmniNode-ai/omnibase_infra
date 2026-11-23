# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-25070: omnibase-core stops declaring the onex console script.

The dispatch venv will have no bin/onex; the developer wrapper runs the
dispatch venv's python -m omnibase_core.cli.cli_commands instead, and treats
a venv without omnibase-core's CLI as missing (reconcile once, then refuse)
rather than running whatever bin/onex is left there.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.scripts.onex_dispatch_fake import (
    CORE_CLI_MODULE,
    install_fake_core_cli,
    remove_core_cli,
)
from tests.scripts.test_onex_wrapper import (
    _EXIT_REFUSED,
    _SENTINEL_OK,
    _Workspace,
)

_RECORD = 'printf "ENTRYPOINT %s\\n" "$*" >> "$WITNESS"\n' + f"exit {_SENTINEL_OK}\n"


@pytest.fixture
def workspace(tmp_path: Path) -> _Workspace:
    """A throwaway workspace holding a copy of the wrapper."""
    return _Workspace(tmp_path)


def test_execs_the_dispatch_pythons_core_module(workspace: _Workspace) -> None:
    """The wrapper execs the dispatch python with the core CLI module."""
    install_fake_core_cli(workspace.dispatch_venv, _RECORD)
    assert not (workspace.venv_bin / "onex").exists()
    result = workspace.run("node", "node_example")
    assert result.returncode == _SENTINEL_OK, result.stderr
    assert workspace.witness_lines() == ["ENTRYPOINT node node_example"]


def test_a_leftover_dispatch_onex_is_never_run(workspace: _Workspace) -> None:
    """A leftover bin/onex in the dispatch venv is never executed."""
    leftover = workspace.venv_bin / "onex"
    leftover.write_text(
        '#!/usr/bin/env bash\nprintf "LEFTOVER %s\\n" "$*" >> "$WITNESS"\nexit 0\n',
        encoding="utf-8",
    )
    leftover.chmod(0o755)
    install_fake_core_cli(workspace.dispatch_venv, _RECORD)
    result = workspace.run("node", "node_example")
    assert result.returncode == _SENTINEL_OK, result.stderr
    lines = workspace.witness_lines()
    assert not any(ln.startswith("LEFTOVER") for ln in lines), (
        "a bin/onex left in the dispatch venv belongs to an older "
        "omnibase-core; the wrapper must run the installed core module "
        "instead"
    )
    assert lines == ["ENTRYPOINT node node_example"]


def test_python_without_the_core_cli_refuses_naming_the_module(
    workspace: _Workspace,
) -> None:
    """A venv without the core CLI is refused, naming the module."""
    install_fake_core_cli(workspace.dispatch_venv, _RECORD)
    remove_core_cli(workspace.dispatch_venv)
    result = workspace.run(
        "node",
        "node_example",
        extra_env={"ONEX_WRAPPER_NO_RECONCILE": "1"},
    )
    assert result.returncode == _EXIT_REFUSED, result.stderr
    assert CORE_CLI_MODULE in result.stderr
    assert str(workspace.dispatch_venv) in result.stderr
    assert not any(ln.startswith("ENTRYPOINT") for ln in workspace.witness_lines())


def test_the_fake_refuses_any_other_invocation(tmp_path: Path) -> None:
    """Positive control: the fake refuses any non-core-module invocation."""
    python = install_fake_core_cli(tmp_path / "venv", _RECORD)
    import subprocess

    witness = tmp_path / "w.log"
    result = subprocess.run(
        [str(python), "-m", "something_else", "x"],
        env={"WITNESS": str(witness), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 98
    assert witness.read_text(encoding="utf-8").startswith("WRONG_ENTRY")


def test_a_reconcile_that_leaves_the_core_cli_missing_still_refuses(
    workspace: _Workspace,
) -> None:
    """The check also runs after the one reconcile: a reconcile that leaves the interpreter but not omnibase-core's CLI must end in a refusal, not an exec that fails at import."""
    install_fake_core_cli(workspace.dispatch_venv, _RECORD)
    remove_core_cli(workspace.dispatch_venv)
    workspace.install_reconciler(creates_entrypoint=False)
    result = workspace.run("node", "node_example")
    assert result.returncode == _EXIT_REFUSED, result.stderr
    lines = workspace.witness_lines()
    assert any(ln.startswith("RECONCILE") for ln in lines), lines
    assert not any(ln.startswith("ENTRYPOINT") for ln in lines), lines
    assert CORE_CLI_MODULE in result.stderr
