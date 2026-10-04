# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A refused canonical clone names its owner, and a second refusal tells them (OMN-20403).

On 2026-10-03 the workspace reconcile log held 66 FAILED and 62 declined ticks
and no ``ok`` one: ``omnibase_infra`` and ``omnimarket`` carried staged files
that nobody owned in the ledger, every pass refused on them, and the refusal
said only that the clone did not move. These tests drive the real
``reconcile-host.sh``, the real verifier and the real ``scripts/onex`` over the
same hermetic fixture as ``test_dispatch_floor_omn20111.py`` and pin:

* the refusal prints the dirty paths, the index mtime and the ledger lane whose
  CLAIM last named those paths, or ``unowned``;
* the second consecutive refusal of the same clone at the same index state
  appends exactly one MSG row, after saving a patch backup of the diff; a
  changed index resets the count;
* a stale ref lock older than the converge timeout is named with its age and
  the removal command, which is printed and never run;
* the wrapper's exit-3 refusal repeats the owner line from the receipt.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.scripts.test_dispatch_floor_omn20111 import (
    _install_wrapper,
    _lock,
    _no_op_delegates,
    _run_wrapper,
    _stale_floor,
)
from tests.scripts.test_reconcile_host_omn17307 import (
    EXIT_FAILED,
    Workspace,
    _advance_origin,
    _git,
    _make_clone,
    _run,
    _write_dist,
    build_workspace,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LEDGER_LOCK = _REPO_ROOT / "scripts" / "ledger_lock.py"
# ledger_lock.py loads its test-write guard from a path relative to itself and
# refuses to write without it, so the fixture carries the real one beside it.
_WRITE_GUARD = (
    _REPO_ROOT / "src" / "omnibase_infra" / "handlers" / "handler_ledger_write_guard.py"
)

_OWNER = "owner-lane-7f"
_CLAIM = (
    "2026-10-03T08:00:00Z | CLAIM | lane={lane} | ticket=OMN-1 | actor=claude | "
    "model=m | worktree=none | scope={scope}"
)


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    built = build_workspace(tmp_path)
    shutil.copy2(_LEDGER_LOCK, built.scripts / "ledger_lock.py")
    guard = built.infra / "src" / "omnibase_infra" / "handlers" / _WRITE_GUARD.name
    guard.parent.mkdir(parents=True)
    shutil.copy2(_WRITE_GUARD, guard)
    return built


def _ledger(ws: Workspace) -> Path:
    return ws.root.parent / "ledger" / "ledger.md"


def _write_ledger(ws: Workspace, *rows: str) -> Path:
    ledger = _ledger(ws)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("".join(row + "\n" for row in rows), encoding="utf-8")
    return ledger


def _msg_rows(ws: Workspace) -> list[str]:
    return [
        line
        for line in _ledger(ws).read_text(encoding="utf-8").splitlines()
        if " | MSG | " in line
    ]


def _dirty(
    ws: Workspace, name: str = "omnibase_core", path: str = "staged.txt"
) -> Path:
    """A clone behind origin with one staged path: the delegate cannot move it."""
    clone, seed = _make_clone(ws.root, name)
    _advance_origin(seed, "merged-upstream")
    (clone / path).write_text("staged\n", encoding="utf-8")
    _git(clone, "add", path)
    return clone


@pytest.fixture(autouse=True)
def _ledger_env(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONEX_LEDGER_PATH", str(_ledger(ws)))
    monkeypatch.delenv("ONEX_RECONCILE_STEP_TIMEOUT_S", raising=False)


def _tick(ws: Workspace, *args: str) -> subprocess.CompletedProcess[str]:
    _no_op_delegates(ws)
    _lock(ws)
    return _run(ws, *args)


def test_converge_dirty_clone_names_owner_and_messages_once(ws: Workspace) -> None:
    clone = _dirty(ws)
    _write_ledger(
        ws,
        _CLAIM.format(lane="someone-else", scope="docs only"),
        _CLAIM.format(lane=_OWNER, scope="omnibase_core staged.txt rework"),
    )

    first = _tick(ws)

    assert first.returncode == EXIT_FAILED, first.stderr
    assert "clone:omnibase_core: DID_NOT_MOVE" in first.stderr
    assert "staged.txt" in first.stderr
    assert "index mtime" in first.stderr
    assert f"owner: {_OWNER}" in first.stderr
    assert _msg_rows(ws) == [], "the first refusal only reports"

    second = _tick(ws)

    assert second.returncode == EXIT_FAILED, second.stderr
    rows = _msg_rows(ws)
    assert len(rows) == 1, rows
    assert f"to={_OWNER}" in rows[0]
    assert "from=reconcile-host-omnibase_core | " in rows[0]
    assert "id=" in rows[0] and "-reconcile-host-omnibase_core | " in rows[0]
    assert "omnibase_core" in rows[0] and "staged.txt" in rows[0]
    backups = list((ws.root / ".onex_state" / "dirty-clone-backups").glob("*.patch"))
    assert len(backups) == 1, backups
    assert "staged.txt" in backups[0].read_text(encoding="utf-8")
    assert str(backups[0]) in rows[0], "the MSG names where the work was saved"
    assert _git(clone, "diff", "--cached", "--name-only") == "staged.txt"

    third = _tick(ws)

    assert third.returncode == EXIT_FAILED, third.stderr
    assert len(_msg_rows(ws)) == 1, "a third refusal at the same index adds no row"
    assert f"owner: {_OWNER}" in third.stderr


def test_converge_dirty_clone_names_owner_unowned_goes_to_the_operator(
    ws: Workspace,
) -> None:
    _dirty(ws)
    _write_ledger(ws, _CLAIM.format(lane="elsewhere", scope="unrelated work"))

    first = _tick(ws)
    _tick(ws)

    assert "owner: unowned" in first.stderr
    rows = _msg_rows(ws)
    assert len(rows) == 1, rows
    assert "to=operator" in rows[0]


def test_converge_dirty_clone_names_owner_a_changed_index_resets_the_count(
    ws: Workspace,
) -> None:
    clone = _dirty(ws)
    _write_ledger(ws, _CLAIM.format(lane=_OWNER, scope="omnibase_core staged.txt"))

    _tick(ws)
    (clone / "second.txt").write_text("more\n", encoding="utf-8")
    _git(clone, "add", "second.txt")
    changed = _tick(ws)

    assert "owner:" in changed.stderr
    assert _msg_rows(ws) == [], "a changed index starts the count again"

    _tick(ws)
    assert len(_msg_rows(ws)) == 1


def test_converge_dirty_clone_names_owner_check_mode_reports_but_writes_nothing(
    ws: Workspace,
) -> None:
    _dirty(ws)
    _write_ledger(ws, _CLAIM.format(lane=_OWNER, scope="omnibase_core staged.txt"))
    before = _ledger(ws).read_text(encoding="utf-8")

    for _ in range(3):
        proc = _tick(ws, "--check")

    assert f"owner: {_OWNER}" in proc.stderr
    assert _ledger(ws).read_text(encoding="utf-8") == before
    assert not (ws.root / ".onex_state").exists()


def test_converge_dirty_clone_names_owner_without_a_ledger_says_so(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dirty(ws)
    monkeypatch.delenv("ONEX_LEDGER_PATH")
    env_less = _tick(ws)

    assert env_less.returncode == EXIT_FAILED, env_less.stderr
    assert "owner: unknown" in env_less.stderr
    assert "ONEX_LEDGER_PATH" in env_less.stderr


def test_converge_dirty_clone_names_owner_in_the_wrappers_refusal(
    ws: Workspace,
) -> None:
    """A dirty omnimarket clone blocks delegation; the refusal carries the owner."""
    market, seed = _make_clone(ws.root, "omnimarket")
    head = _git(market, "rev-parse", "HEAD")
    _advance_origin(seed, "unpulled")
    (market / "staged.txt").write_text("staged\n", encoding="utf-8")
    _git(market, "add", "staged.txt")
    _write_dist(ws.site_packages, "omnimarket", "0.4.11", commit=head)
    _write_ledger(ws, _CLAIM.format(lane=_OWNER, scope="omnimarket staged.txt"))
    _install_wrapper(ws)
    _tick(ws)
    assert not ws.floor.exists(), "a dispatch-premise failure stamps no floor"

    proc = _run_wrapper(ws, "delegate", "x")

    assert proc.returncode == 3, proc.stderr
    assert "blocking : clone:omnimarket" in proc.stderr
    assert f"owner    : {_OWNER}" in proc.stderr
    assert not (ws.root / "argv.log").exists(), "the CLI must not have run"
    receipt = json.loads(ws.receipt.read_text(encoding="utf-8"))
    assert receipt["dispatch_premise_failures"] == 1


def test_a_dirty_omnimarket_clone_still_withholds_the_floor_with_an_owner(
    ws: Workspace,
) -> None:
    """OMN-20111 preserved: the new report changes no floor decision."""
    market, seed = _make_clone(ws.root, "omnimarket")
    head = _git(market, "rev-parse", "HEAD")
    _advance_origin(seed, "unpulled")
    (market / "staged.txt").write_text("staged\n", encoding="utf-8")
    _git(market, "add", "staged.txt")
    _write_dist(ws.site_packages, "omnimarket", "0.4.11", commit=head)
    _write_ledger(ws, _CLAIM.format(lane=_OWNER, scope="omnimarket staged.txt"))
    previous = _stale_floor(ws)

    proc = _tick(ws)

    assert "blocks onex delegate: clone:omnimarket" in proc.stderr
    assert ws.floor.read_text(encoding="utf-8") == previous


def test_an_unrelated_dirty_clone_with_an_owner_still_stamps_the_floor(
    ws: Workspace,
) -> None:
    _dirty(ws)
    market, _ = _make_clone(ws.root, "omnimarket")
    head = _git(market, "rev-parse", "HEAD")
    _write_dist(ws.site_packages, "omnimarket", "0.4.11", commit=head)
    _write_ledger(ws, _CLAIM.format(lane=_OWNER, scope="omnibase_core staged.txt"))
    _stale_floor(ws)

    proc = _tick(ws)

    assert "do not block onex delegate" in proc.stderr
    assert json.loads(ws.floor.read_text(encoding="utf-8"))["omnimarket_commit"] == head


def _age(path: Path, seconds: int) -> None:
    stamp = path.stat().st_mtime - seconds
    os.utime(path, (stamp, stamp))


def test_converge_names_stale_ref_lock(ws: Workspace) -> None:
    clone = _dirty(ws)
    stale = clone / ".git" / "refs" / "heads" / "dev.lock"
    stale.write_text("", encoding="utf-8")
    _age(stale, 7200)
    young = clone / ".git" / "index.lock"
    young.write_text("", encoding="utf-8")
    _write_ledger(ws, _CLAIM.format(lane=_OWNER, scope="omnibase_core staged.txt"))

    proc = _tick(ws)

    assert proc.returncode == EXIT_FAILED, proc.stderr
    assert f"stale ref lock: {stale}" in proc.stderr
    assert re.search(r"age 72\d\ds", proc.stderr), proc.stderr
    assert f"rm -f {stale}" in proc.stderr
    assert str(young) not in proc.stderr, "a lock younger than the timeout is live work"
    assert stale.exists(), "the removal command is printed, never run"


def test_converge_names_stale_ref_lock_on_a_clean_clone_that_cannot_move(
    ws: Workspace,
) -> None:
    clone, seed = _make_clone(ws.root, "omnibase_core")
    _advance_origin(seed, "merged-upstream")
    stale = clone / ".git" / "index.lock"
    stale.write_text("", encoding="utf-8")
    _age(stale, 7200)

    proc = _tick(ws)

    assert f"stale ref lock: {stale}" in proc.stderr
    assert "dirty paths" not in proc.stderr
