# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for fail-closed architecture-layer wrapper behavior (OMN-17793)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.validate import run_architecture_layers


@pytest.mark.unit
def test_invalid_source_target_is_failure_not_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        assert kwargs["timeout"] == 120
        assert kwargs["shell"] is False
        return subprocess.CompletedProcess(
            ["/bin/bash"], 2, stdout="", stderr="invalid source"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert run_architecture_layers(verbose=True) is False
    command = calls[0][0]
    assert isinstance(command, list)
    assert command[0] == "/bin/bash"
    assert str(command[1]).endswith("scripts/check_architecture.sh")
    assert command[2:] == ["--no-color", "--verbose"]


@pytest.mark.unit
def test_missing_architecture_script_is_failure_not_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "exists", lambda _path: False)

    assert run_architecture_layers() is False


@pytest.mark.unit
def test_missing_fixed_shell_is_failure_not_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_missing_shell(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("/bin/bash")

    monkeypatch.setattr(subprocess, "run", raise_missing_shell)

    assert run_architecture_layers() is False
