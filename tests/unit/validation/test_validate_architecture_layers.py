# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
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
    calls: list[dict[str, object]] = []

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(kwargs)
        return subprocess.CompletedProcess(
            ["bash"], 2, stdout="", stderr="invalid source"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert run_architecture_layers(verbose=True) is False
    assert calls[0]["timeout"] == 120
    assert calls[0]["shell"] is False


@pytest.mark.unit
def test_missing_architecture_script_is_failure_not_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "exists", lambda _path: False)

    assert run_architecture_layers() is False


@pytest.mark.unit
def test_missing_bash_is_failure_not_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_missing_bash(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("bash")

    monkeypatch.setattr(subprocess, "run", raise_missing_bash)

    assert run_architecture_layers() is False
