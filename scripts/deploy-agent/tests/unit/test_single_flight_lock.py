# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the single-flight advisory lock in deploy_agent.agent."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from deploy_agent.events import DeployInProgressError
from deploy_agent.lock import single_flight_lock


@pytest.mark.unit
def test_lock_acquired_and_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("deploy_agent.lock._LOCK_PATH", tmp_path / "deploy-agent.lock")
    with single_flight_lock():
        pass  # must not raise


@pytest.mark.unit
def test_second_attempt_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("deploy_agent.lock._LOCK_PATH", tmp_path / "deploy-agent.lock")

    errors: list[Exception] = []
    ready = threading.Event()
    release = threading.Event()

    def hold_lock() -> None:
        with single_flight_lock():
            ready.set()
            release.wait(timeout=5)

    holder = threading.Thread(target=hold_lock, daemon=True)
    holder.start()
    ready.wait(timeout=5)

    try:
        with single_flight_lock():
            pass
    except DeployInProgressError as exc:
        errors.append(exc)
    finally:
        release.set()
        holder.join(timeout=5)

    assert len(errors) == 1
    assert "in flight" in str(errors[0])


@pytest.mark.unit
def test_lock_released_after_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("deploy_agent.lock._LOCK_PATH", tmp_path / "deploy-agent.lock")

    with pytest.raises(RuntimeError, match="boom"):
        with single_flight_lock():
            raise RuntimeError("boom")

    # Lock must be released — second acquisition must succeed.
    with single_flight_lock():
        pass
