# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Retire/reconcile of a hot-patch ledger row (OMN-17427).

Two ``merge_commit: null`` rows once blocked every dev redeploy, and the only
defined retirement was a hand edit of the ledger YAML. The reconcile handler
retires a row only on evidence it checks itself: the fix commit is an ancestor
of the deployed ref, and the container carries no ``.prepatch``. Every refusal
must leave the ledger byte-for-byte untouched.
"""

from __future__ import annotations

import importlib.util
import os
import stat
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.handlers.handler_hotpatch_ledger_reconcile import (
    HandlerHotpatchLedgerReconcile,
)
from omnibase_infra.models.model_hotpatch_ledger_reconcile_request import (
    ModelHotpatchLedgerReconcileRequest,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 10, 9, 15, 0, 0, tzinfo=UTC)
PREFLIGHT_SCRIPT = (
    Path(__file__).resolve().parents[3] / "scripts" / "preflight_hotpatch_ledger.py"
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env={
            **scrub_git_location_env(os.environ),
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        },
    ).stdout.strip()


def _commit(repo: Path, content: str) -> str:
    (repo / "f.txt").write_text(content)
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-m", content)
    return _git(repo, "rev-parse", "HEAD")


class Fixture:
    """A clones root with one repo holding a fix commit, plus a ledger."""

    def __init__(self, root: Path) -> None:
        self.clones = root / "clones"
        self.repo = self.clones / "omnibase_infra"
        self.repo.mkdir(parents=True)
        _git(self.repo, "init", "-b", "dev")
        self.base = _commit(self.repo, "one\n")
        self.fix = _commit(self.repo, "two\n")
        self.head = self.fix
        self.ledger = root / "hotpatch-ledger" / "ledger.yaml"
        self.ledger.parent.mkdir()
        self.bin = root / "bin"
        self.bin.mkdir()

    def docker(self, prepatch_lines: list[str], rc: int = 0, stderr: str = "") -> str:
        script = self.bin / "docker"
        body = "\n".join(prepatch_lines)
        script.write_text(
            '#!/bin/sh\nif [ "$1" = "exec" ]; then\n'
            f'cat >&2 <<"ERR"\n{stderr}\nERR\n'
            f'cat <<"EOF"\n{body}\nEOF\nexit {rc}\nfi\nexit 0\n'
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        return str(script)

    def write(self, rows: list[dict[str, Any]]) -> None:
        self.ledger.write_text(yaml.safe_dump({"schema": 1, "rows": rows}))

    def request(self, **overrides: Any) -> ModelHotpatchLedgerReconcileRequest:
        fields: dict[str, Any] = {
            "ledger_path": str(self.ledger),
            "clones_root": str(self.clones),
            "container": "omninode-gateway-forwarder",
            "file": "/app/x.py",
            "merge_commit": self.fix,
        }
        fields.update(overrides)
        if "docker_cmd" not in fields:
            fields["docker_cmd"] = self.docker([])
        return ModelHotpatchLedgerReconcileRequest(**fields)


def _stale_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "container": "omninode-gateway-forwarder",
        "lane": "dev",
        "file": "/app/x.py",
        "prepatch_path": "/app/x.py.prepatch",
        "source_repo": "omnibase_infra",
        "source_pr": "OmniNode-ai/omnibase_infra#4640",
        "merge_commit": None,
        "merged": False,
    }
    row.update(overrides)
    return row


def _bystander_row() -> dict[str, Any]:
    return _stale_row(
        container="omnimarket-projection-api",
        file="/app/y.py",
        prepatch_path="/app/y.py.prepatch",
        source_pr="OmniNode-ai/omnibase_infra#4641",
    )


@pytest.fixture
def fx(tmp_path: Path) -> Fixture:
    return Fixture(tmp_path)


def _handler() -> HandlerHotpatchLedgerReconcile:
    return HandlerHotpatchLedgerReconcile(clock=lambda: NOW)


def _backups(fx: Fixture) -> list[Path]:
    return sorted(fx.ledger.parent.glob("ledger.yaml.bak-*"))


def _load_preflight() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "preflight_hotpatch", PREFLIGHT_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_preflight(fx: Fixture, docker: str) -> int:
    return _load_preflight().main(
        [
            "--container",
            "omninode-gateway-forwarder",
            "--ledger",
            str(fx.ledger),
            "--clones-root",
            str(fx.clones),
            "--docker-cmd",
            docker,
        ]
    )


async def test_reconcile_retires_null_merge_commit_row(fx: Fixture) -> None:
    fx.write([_stale_row(), _bystander_row()])
    original = fx.ledger.read_text()

    result = await _handler().handle(fx.request(note="forwarder recreated"))

    assert result.success, result
    rows = yaml.safe_load(fx.ledger.read_text())["rows"]
    retired, bystander = rows
    assert retired["status"] == "reconciled"
    assert retired["reconciled_utc"] == "2026-10-09T15:00:00Z"
    assert retired["merge_commit"] == fx.fix
    assert retired["merged"] is True
    assert fx.fix[:12] in retired["reconciliation_note"]
    assert "forwarder recreated" in retired["reconciliation_note"]
    assert bystander == _bystander_row()
    assert [p.read_text() for p in _backups(fx)] == [original]
    assert result.backup_path == str(_backups(fx)[0])


async def test_reconciled_ledger_passes_the_preflight_that_refused_it(
    fx: Fixture,
) -> None:
    fx.write([_stale_row(), _bystander_row()])
    docker = fx.docker([])
    assert _run_preflight(fx, docker) == 2  # positive control: null row refuses

    result = await _handler().handle(fx.request())

    assert result.success, result
    assert _run_preflight(fx, docker) == 0


async def test_ledger_mode_is_preserved(fx: Fixture) -> None:
    fx.write([_stale_row()])
    fx.ledger.chmod(0o640)

    result = await _handler().handle(fx.request())

    assert result.success, result
    assert stat.S_IMODE(fx.ledger.stat().st_mode) == 0o640


async def test_deployed_ref_is_honoured(fx: Fixture) -> None:
    fx.write([_stale_row()])
    result = await _handler().handle(fx.request(deployed_ref=fx.base))
    assert not result.success
    assert "not an ancestor" in result.reason

    result = await _handler().handle(fx.request(deployed_ref=fx.fix))
    assert result.success, result


async def test_row_commit_list_is_used_when_none_supplied(fx: Fixture) -> None:
    fx.write([_stale_row(merge_commit=["0" * 40, fx.fix], merged=True)])

    result = await _handler().handle(fx.request(merge_commit=""))

    assert result.success, result
    row = yaml.safe_load(fx.ledger.read_text())["rows"][0]
    assert row["merge_commit"] == ["0" * 40, fx.fix]  # forensic list kept
    assert fx.fix[:12] in row["reconciliation_note"]


async def _assert_refused(
    fx: Fixture, request: ModelHotpatchLedgerReconcileRequest, needle: str
) -> None:
    original = fx.ledger.read_bytes()
    result = await _handler().handle(request)
    assert not result.success, result
    assert needle in result.reason, result.reason
    assert result.error_message == result.reason
    assert fx.ledger.read_bytes() == original
    assert _backups(fx) == []


async def test_refuses_when_commit_is_not_an_ancestor(fx: Fixture) -> None:
    _git(fx.repo, "checkout", "-b", "side", fx.base)
    side = _commit(fx.repo, "side\n")
    _git(fx.repo, "checkout", "dev")
    fx.write([_stale_row()])
    await _assert_refused(fx, fx.request(merge_commit=side), "not an ancestor")


async def test_refuses_when_commit_is_unknown_in_the_clone(fx: Fixture) -> None:
    fx.write([_stale_row()])
    await _assert_refused(fx, fx.request(merge_commit="1" * 40), "unknown in clone")


async def test_refuses_when_container_still_carries_prepatch(fx: Fixture) -> None:
    fx.write([_stale_row()])
    docker = fx.docker(["/app/x.py.prepatch"])
    await _assert_refused(fx, fx.request(docker_cmd=docker), "/app/x.py.prepatch")


async def test_refuses_when_any_prepatch_is_live_in_container(fx: Fixture) -> None:
    fx.write([_stale_row()])
    docker = fx.docker(["/app/other.py.prepatch"])
    await _assert_refused(fx, fx.request(docker_cmd=docker), "/app/other.py.prepatch")


async def test_refuses_when_container_cannot_be_probed(fx: Fixture) -> None:
    fx.write([_stale_row()])
    docker = fx.docker([], rc=1, stderr="Error: No such container: x")
    await _assert_refused(fx, fx.request(docker_cmd=docker), "cannot verify")


async def test_refuses_when_no_commit_is_available(fx: Fixture) -> None:
    fx.write([_stale_row()])
    await _assert_refused(fx, fx.request(merge_commit=""), "no merge commit")


async def test_refuses_commit_that_contradicts_the_recorded_one(fx: Fixture) -> None:
    fx.write([_stale_row(merge_commit=fx.base, merged=True)])
    await _assert_refused(
        fx, fx.request(merge_commit=fx.fix), "differs from the row's recorded"
    )


async def test_refuses_unknown_row(fx: Fixture) -> None:
    fx.write([_stale_row()])
    await _assert_refused(fx, fx.request(file="/app/nope.py"), "no ledger row")


async def test_refuses_already_reconciled_row(fx: Fixture) -> None:
    fx.write(
        [
            _stale_row(
                status="reconciled",
                reconciled_utc="2026-10-01T00:00:00Z",
                reconciliation_note="done",
            )
        ]
    )
    await _assert_refused(fx, fx.request(), "already reconciled")


async def test_refuses_ambiguous_row(fx: Fixture) -> None:
    fx.write([_stale_row(), _stale_row()])
    await _assert_refused(fx, fx.request(), "more than one")


async def test_refuses_missing_ledger(fx: Fixture) -> None:
    result = await _handler().handle(fx.request())
    assert not result.success
    assert "ledger" in result.reason
    assert not fx.ledger.exists()


async def test_refuses_unsupported_schema(fx: Fixture) -> None:
    fx.ledger.write_text(yaml.safe_dump({"schema": 2, "rows": [_stale_row()]}))
    await _assert_refused(fx, fx.request(), "schema")


def test_request_rejects_unknown_fields() -> None:
    fields = {
        "ledger_path": "l",
        "clones_root": "c",
        "container": "c",
        "file": "f",
        "surprise": "x",
    }
    with pytest.raises(ValueError):
        ModelHotpatchLedgerReconcileRequest(**fields)
