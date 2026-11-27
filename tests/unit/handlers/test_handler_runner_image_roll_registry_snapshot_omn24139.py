# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-24139: the image roll reads the org runner registry from a declared snapshot on a lab host.

A lab lane's GitHub shim refuses every raw ``gh api`` read (exit 78, budget 0), so the roll could not run from
a lab host. The orchestrator supplies the registry as a snapshot file; the handler never calls ``gh`` for it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from omnibase_infra.handlers.handler_runner_image_roll import (
    HandlerRunnerImageRoll,
    RunnerRollError,
)
from omnibase_infra.models.model_runner_image_roll_request import (
    ModelRunnerImageRollRequest,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
FLEET = ROOT / "config/runner_fleet.yaml"
POOL = "hp053-ci-runner"
SHIM_REFUSAL = 78


@pytest.fixture
def token_file(tmp_path: Path) -> Path:
    path = tmp_path / "token"
    path.touch(mode=0o600)
    path.write_text("fixture-token\n")
    return path


def _snapshot(tmp_path: Path, runners: list[dict[str, object]]) -> Path:
    # The shape `gh api --paginate --slurp /orgs/<org>/actions/runners` prints: a list of pages.
    path = tmp_path / "registry.json"
    path.write_text(json.dumps([{"total_count": len(runners), "runners": runners}]))
    return path


def _request(
    token_file: Path, snapshot: Path | None, **extra: object
) -> ModelRunnerImageRollRequest:
    return ModelRunnerImageRollRequest.model_validate(
        {
            "host": "h053",
            "prefix": POOL,
            "repo_root": ROOT,
            "fleet_config": FLEET,
            "token_path": token_file,
            "registry_snapshot": str(snapshot) if snapshot else "",
            **extra,
        }
    )


def _refuse_gh(
    monkeypatch: pytest.MonkeyPatch, handler: HandlerRunnerImageRoll
) -> list[tuple[str, ...]]:
    """Stand in for the lane shim: any gh call is refused with exit 78 and recorded."""
    calls: list[tuple[str, ...]] = []

    async def command(
        *argv: str, stdin: bytes | None = None, timeout: int = 30
    ) -> tuple[int, str]:
        calls.append(argv)
        return SHIM_REFUSAL, ""

    monkeypatch.setattr(handler, "_command", command)
    return calls


@pytest.mark.asyncio
async def test_registry_comes_from_the_snapshot_and_gh_is_never_called(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, token_file: Path
) -> None:
    handler = HandlerRunnerImageRoll()
    calls = _refuse_gh(monkeypatch, handler)
    snapshot = _snapshot(
        tmp_path,
        [
            {"name": f"{POOL}-1", "status": "online", "busy": False},
            {"name": f"{POOL}-2", "status": "online", "busy": True},
        ],
    )
    request = _request(token_file, snapshot)
    assert await handler._registry("OmniNode-ai", f"{POOL}-1", request) == (
        "online",
        False,
    )
    assert await handler._registry("OmniNode-ai", f"{POOL}-2", request) == (
        "online",
        True,
    )
    assert await handler._registry("OmniNode-ai", f"{POOL}-3", request) == (
        "unknown",
        None,
    )
    assert calls == []


@pytest.mark.asyncio
async def test_no_snapshot_still_reads_through_gh_and_a_shim_refusal_stops_the_roll(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, token_file: Path
) -> None:
    """Positive control: without the snapshot the old path runs and the refusal is the stop."""
    handler = HandlerRunnerImageRoll()
    calls = _refuse_gh(monkeypatch, handler)
    with pytest.raises(RunnerRollError, match=f"gh exit {SHIM_REFUSAL}"):
        await handler._registry("OmniNode-ai", f"{POOL}-1", _request(token_file, None))
    assert [call[0] for call in calls] == ["gh"]


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["stale", "missing", "symlink", "undecodable"])
async def test_unreadable_or_fail_closed_snapshot_refuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, token_file: Path, fault: str
) -> None:
    handler = HandlerRunnerImageRoll()
    calls = _refuse_gh(monkeypatch, handler)
    snapshot = _snapshot(
        tmp_path, [{"name": f"{POOL}-1", "status": "online", "busy": False}]
    )
    if fault == "stale":
        old = time.time() - 4000
        os.utime(snapshot, (old, old))
    elif fault == "missing":
        snapshot = tmp_path / "absent.json"
    elif fault == "symlink":
        link = tmp_path / "link.json"
        link.symlink_to(snapshot)
        snapshot = link
    else:
        snapshot.write_text("not json")
    with pytest.raises(RunnerRollError, match="snapshot"):
        await handler._registry(
            "OmniNode-ai", f"{POOL}-1", _request(token_file, snapshot)
        )
    assert calls == []


@pytest.mark.asyncio
async def test_snapshot_age_limit_is_the_request_setting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, token_file: Path
) -> None:
    handler = HandlerRunnerImageRoll()
    _refuse_gh(monkeypatch, handler)
    snapshot = _snapshot(
        tmp_path, [{"name": f"{POOL}-1", "status": "online", "busy": False}]
    )
    old = time.time() - 4000
    os.utime(snapshot, (old, old))
    request = _request(token_file, snapshot, registry_snapshot_max_age_seconds=7200)
    assert await handler._registry("OmniNode-ai", f"{POOL}-1", request) == (
        "online",
        False,
    )


@pytest.mark.asyncio
async def test_roll_proves_registry_access_from_the_snapshot_before_the_host_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, token_file: Path
) -> None:
    """The first step of a real roll (not a dry run) passes under a refusing gh shim."""
    handler = HandlerRunnerImageRoll()
    calls = _refuse_gh(monkeypatch, handler)
    remote: list[str] = []

    async def refuse_lock(host: str, script: str, **_kw: object) -> tuple[int, str]:
        remote.append(script)
        return 1, ""

    monkeypatch.setattr(handler, "_remote", refuse_lock)
    snapshot = _snapshot(
        tmp_path, [{"name": f"{POOL}-1", "status": "online", "busy": False}]
    )
    result = await handler.handle(_request(token_file, snapshot, only=f"{POOL}-1"))
    assert not result.success
    assert "pool roll lock" in (result.error_message or "")
    assert calls == []
    assert len(remote) == 1


@pytest.mark.asyncio
async def test_online_readback_with_a_snapshot_is_the_listener_log_not_the_old_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, token_file: Path
) -> None:
    """A snapshot predates the recreate, so only the new container's listener line can prove it online."""
    handler = HandlerRunnerImageRoll()
    _refuse_gh(monkeypatch, handler)
    scripts: list[str] = []

    async def container(*_args: object) -> tuple[str, str]:
        return "after", "sha256:new"

    async def remote(host: str, script: str, **_kw: object) -> tuple[int, str]:
        scripts.append(script)
        return 0, ""

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(handler, "_container", container)
    monkeypatch.setattr(handler, "_remote", remote)
    monkeypatch.setattr(
        "omnibase_infra.handlers.handler_runner_image_roll.asyncio.sleep", no_sleep
    )
    snapshot = _snapshot(
        tmp_path, [{"name": f"{POOL}-1", "status": "offline", "busy": False}]
    )
    assert await handler._wait_online(
        _request(token_file, snapshot), f"{POOL}-1", "before", "sha256:new", "Org"
    )
    assert any("Listening for Jobs" in script for script in scripts)
