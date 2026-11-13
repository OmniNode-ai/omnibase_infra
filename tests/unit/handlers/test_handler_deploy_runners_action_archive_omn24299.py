# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24299: the prepare effect fills the host's action-archive cache and copies it.

The fill itself is covered over real git in
``test_handler_action_archive_cache_fill_omn24299``; here the effect around it:
which hosts get a cache, that the copy carries the declared root and the two rsync
flags, and that every problem is a warning because the cache is fail-open.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.handlers.handler_deploy_runners import HandlerDeployRunners
from omnibase_infra.models.model_deploy_runners_request import ModelDeployRunnersRequest
from tests.unit.handlers.test_handler_action_archive_cache_fill_omn24299 import (
    _make_fleet_yaml,
    _setup_fixtures,
)

HOST = "ci-host"
HOST_ROOT = "/srv/action-archives"


class RecordingHandler(HandlerDeployRunners):
    """The handler with ssh and rsync replaced by a record; no host is touched."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[tuple[str, ...]] = []
        self._fail_on = fail_on

    async def _remote(self, host: str, command: str, timeout: int) -> str:
        self.calls.append(("ssh", host, command))
        if self._fail_on == "ssh":
            raise OSError("ssh refused")
        return ""

    async def _rsync(self, source: str, host: str, target: str, timeout: int) -> None:
        self.calls.append(("rsync", source, host, target))
        if self._fail_on == "rsync":
            raise OSError("rsync exited 23")


def _fleet(tmp: Path, fx: dict[str, Any], *, with_root: bool = True) -> Path:
    path = _make_fleet_yaml(tmp, fx["staging_root"], fx["clone_root"])
    data = yaml.safe_load(path.read_text())
    row: dict[str, Any] = {
        "host": HOST,
        "arch": "amd64",
        "runner_name_prefix": "test-runner",
        "expected_count": 1,
        "classes": ["action"],
    }
    if with_root:
        row["action_archive_host_root"] = HOST_ROOT
    data["runner_host"] = HOST
    data["hosts"] = [row]
    path.write_text(yaml.dump(data))
    return path


def _request(
    fleet: Path | None, fx: dict[str, Any], host: str = HOST
) -> ModelDeployRunnersRequest:
    return ModelDeployRunnersRequest(
        host=host,
        host_dir="/h",
        compose_file="/c.yml",
        prefix="test-runner",
        count=1,
        fleet_config=str(fleet) if fleet else None,
        clone_root=str(fx["clone_root"]) if fleet else None,
    )


@pytest.mark.asyncio
async def test_golden_fills_then_creates_the_root_then_copies(tmp_path: Path) -> None:
    fx = _setup_fixtures(tmp_path)
    handler = RecordingHandler()
    warnings = await handler._sync_action_archive_cache(
        _request(_fleet(tmp_path, fx), fx), 60
    )

    assert warnings == ()
    assert [c[0] for c in handler.calls] == ["ssh", "rsync"]
    assert handler.calls[0][1:] == (HOST, f"mkdir -p -- {HOST_ROOT}")
    assert handler.calls[1] == ("rsync", str(fx["staging_root"]), HOST, HOST_ROOT)
    assert len(list(fx["staging_root"].rglob("*.tar.gz"))) == 4


@pytest.mark.asyncio
async def test_rsync_flags_never_rewrite_an_archive_and_never_ship_the_mirrors() -> (
    None
):
    import inspect

    source = inspect.getsource(HandlerDeployRunners._rsync)
    assert '"--ignore-existing"' in source
    assert '"--exclude=.sources"' in source


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fail_on", "code"),
    [("ssh", "ACTION-ARCHIVE-SYNC"), ("rsync", "ACTION-ARCHIVE-SYNC")],
)
async def test_error_chain_a_failed_copy_is_a_warning_not_a_refusal(
    tmp_path: Path, fail_on: str, code: str
) -> None:
    fx = _setup_fixtures(tmp_path)
    handler = RecordingHandler(fail_on=fail_on)
    warnings = await handler._sync_action_archive_cache(
        _request(_fleet(tmp_path, fx), fx), 60
    )
    assert [w.split(" ")[0] for w in warnings] == [code]
    assert HOST_ROOT in warnings[0]


@pytest.mark.asyncio
async def test_error_chain_an_unreachable_source_is_named_and_the_copy_still_runs(
    tmp_path: Path,
) -> None:
    fx = _setup_fixtures(tmp_path)
    fleet = _fleet(tmp_path, fx)
    data = yaml.safe_load(fleet.read_text())
    data["action_archive_cache"]["remote_url_template"] = (
        f"file://{tmp_path}/nowhere/{{repo}}.git"
    )
    fleet.write_text(yaml.dump(data))
    handler = RecordingHandler()
    warnings = await handler._sync_action_archive_cache(_request(fleet, fx), 60)
    assert any(
        w.startswith("ACTION-ARCHIVE-FILL SOURCE_UNREACHABLE actions/checkout@v4")
        for w in warnings
    )
    assert [c[0] for c in handler.calls] == ["ssh", "rsync"]


@pytest.mark.asyncio
async def test_a_host_without_a_cache_root_and_a_request_without_paths_do_nothing(
    tmp_path: Path,
) -> None:
    fx = _setup_fixtures(tmp_path)
    handler = RecordingHandler()
    assert await handler._sync_action_archive_cache(_request(None, fx), 60) == ()
    no_root = _fleet(tmp_path, fx, with_root=False)
    assert await handler._sync_action_archive_cache(_request(no_root, fx), 60) == ()
    assert (
        await handler._sync_action_archive_cache(
            _request(_fleet(tmp_path, fx), fx, "other-host"), 60
        )
        == ()
    )
    assert handler.calls == []
    assert list(fx["staging_root"].iterdir()) == []


@pytest.mark.asyncio
async def test_error_chain_an_unreadable_declaration_is_a_warning(
    tmp_path: Path,
) -> None:
    fx = _setup_fixtures(tmp_path)
    request = _request(tmp_path / "missing.yaml", fx)
    warnings = await RecordingHandler()._sync_action_archive_cache(request, 60)
    assert warnings and warnings[0].startswith("ACTION-ARCHIVE-CONFIG")
