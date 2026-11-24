# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24139: the real roll handler reads the registry from a snapshot while a refusing ``gh`` is on PATH.

No handler method is replaced: ``gh`` is a real executable on PATH that records its call and exits 78, as a lab
lane's GitHub shim does, so a handler that still shelled out for the registry would fail here.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from omnibase_infra.handlers.handler_runner_image_roll import (
    HandlerRunnerImageRoll,
    RunnerRollError,
)
from omnibase_infra.models.model_runner_image_roll_request import (
    ModelRunnerImageRollRequest,
)

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[3]
POOL = "hp053-ci-runner"


@pytest.fixture
def refusing_gh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "gh-calls.log"
    gh = bin_dir / "gh"
    gh.write_text(f'#!/bin/sh\necho "$@" >> {log}\nexit 78\n')
    gh.chmod(gh.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    return log


def _request(tmp_path: Path, snapshot: str) -> ModelRunnerImageRollRequest:
    token = tmp_path / "token"
    token.write_text("fixture-token\n")
    token.chmod(0o600)
    return ModelRunnerImageRollRequest.model_validate(
        {
            "host": "h053",
            "prefix": POOL,
            "repo_root": ROOT,
            "fleet_config": ROOT / "config/runner_fleet.yaml",
            "token_path": token,
            "registry_snapshot": snapshot,
        }
    )


@pytest.mark.asyncio
async def test_snapshot_registry_read_never_runs_gh(
    tmp_path: Path, refusing_gh: Path
) -> None:
    snapshot = tmp_path / "registry.json"
    runners = [{"name": f"{POOL}-1", "status": "online", "busy": False}]
    snapshot.write_text(json.dumps([{"total_count": 1, "runners": runners}]))
    handler = HandlerRunnerImageRoll()
    assert await handler._registry(
        "OmniNode-ai", f"{POOL}-1", _request(tmp_path, str(snapshot))
    ) == ("online", False)
    assert not refusing_gh.exists()


@pytest.mark.asyncio
async def test_without_snapshot_the_real_gh_refusal_stops_the_roll(
    tmp_path: Path, refusing_gh: Path
) -> None:
    handler = HandlerRunnerImageRoll()
    with pytest.raises(RunnerRollError, match="gh exit 78"):
        await handler._registry("OmniNode-ai", f"{POOL}-1", _request(tmp_path, ""))
    assert refusing_gh.exists()


@pytest.mark.asyncio
async def test_missing_snapshot_file_refuses(tmp_path: Path, refusing_gh: Path) -> None:
    handler = HandlerRunnerImageRoll()
    with pytest.raises(RunnerRollError, match="snapshot"):
        await handler._registry(
            "OmniNode-ai", f"{POOL}-1", _request(tmp_path, str(tmp_path / "absent"))
        )
    assert not refusing_gh.exists()
