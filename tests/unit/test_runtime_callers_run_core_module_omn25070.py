# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-25070 moves the onex launcher out of omnibase-core into omninode-onex.

A venv that carries omnibase-core but not omninode-onex will have no onex
next to its interpreter. Two runtime callers in this repo exec that
launcher: done_write_receipt_guard.dod_verify_argv (the evidence autoclose
sweep's verifier) and offer_companion_sources._gateway_args (the sweep's
onex scm reads). Both must run omnibase-core's CLI module with the
interpreter they are running in, the same form the sweep workflow already
uses, so neither depends on a launcher file and neither picks up an onex
from PATH that belongs to another install.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_infra.handlers import done_write_receipt_guard as guard
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect import (
    offer_companion_sources as sources,
)

_CORE_CLI = ["-m", "omnibase_core.cli.cli_commands"]


def test_dod_verify_argv_runs_the_core_module_with_its_own_interpreter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The verifier argv runs core's CLI module with the current interpreter."""
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    python = venv / "python"
    monkeypatch.setattr(sys, "executable", str(python))

    argv = guard.dod_verify_argv(
        "OMN-16759",
        contract_path="/c/OMN-16759.yaml",
        state_root="/s",
    )
    assert argv[:3] == [str(python), *_CORE_CLI]
    assert argv[3:] == [
        "skill",
        "dod_verify",
        "OMN-16759",
        "--execution-audience",
        "hosted",
        "--contract-path",
        "/c/OMN-16759.yaml",
        "--state-root",
        "/s",
    ]
    assert not (venv / "onex").exists()
    assert all(Path(a).name != "onex" for a in argv), (
        "the verifier must not be dispatched through an onex launcher file"
    )


def test_dod_verify_argv_prefix_dispatches_the_skill_command() -> None:
    """The module prefix reaches core's skill command in this venv."""
    argv = guard.dod_verify_argv("OMN-16759")
    result = subprocess.run(
        [*argv[:4], "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "SKILL_NAME" in result.stdout


def test_gateway_args_runs_the_core_module_not_a_path_onex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gateway args run core's CLI module, not an onex path on PATH."""
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    monkeypatch.setattr(sys, "executable", str(python))

    def fake_which(name: str) -> str | None:
        return "/elsewhere/bin/onex" if name == "onex" else None

    monkeypatch.setattr(shutil, "which", fake_which)

    args = sources._gateway_args(
        "pr-heads",
        "OmniNode-ai/omnibase_infra",
        ["--limit", "5"],
        "lane-a",
        "/ws",
    )
    assert "/elsewhere/bin/onex" not in args, (
        "an onex on PATH belongs to whatever install put it there "
        "(a customer's omninode-onex, another core version), "
        "not to this runtime"
    )
    assert args[:3] == [str(python), *_CORE_CLI]
    assert args[3:] == [
        "scm",
        "pr-heads",
        "OmniNode-ai/omnibase_infra",
        "--limit",
        "5",
        "--source",
        "bus",
        "--bus-lane",
        "lane-a",
        "--omnibase-path",
        "/ws",
    ]


def test_gateway_args_without_a_workspace_root_omits_omnibase_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a workspace root the gateway args omit --omnibase-path."""
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    monkeypatch.setattr(sys, "executable", str(python))

    args = sources._gateway_args(
        "pr-view",
        "OmniNode-ai/omnimarket#1",
        [],
        "lane-b",
        "",
    )
    assert args[:3] == [str(python), *_CORE_CLI]
    assert "--omnibase-path" not in args
    assert args[3:] == [
        "scm",
        "pr-view",
        "OmniNode-ai/omnimarket#1",
        "--source",
        "bus",
        "--bus-lane",
        "lane-b",
    ]
