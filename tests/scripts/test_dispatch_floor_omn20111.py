# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dispatch floor is scoped to the dispatch premise (OMN-20111).

On 2026-09-29 the canonical ``omnibase_core`` clone carried staged files for
about 90 minutes. Every ``reconcile-host.sh`` pass in that window ended FAILED
on that one surface, the same passes had moved the dispatch venv's omnimarket,
and the floor kept the old commit, so ``scripts/onex`` refused every
``onex delegate`` on the host.

These tests pin both halves of the fix end to end, through the real
``reconcile-host.sh`` and the real ``scripts/onex``:

* an unrelated dirty clone fails the verdict and alerts, but no longer holds the
  floor, so delegation proceeds;
* a failure on a surface the dispatch build is made from (the omnimarket clone,
  the dispatch venv) still withholds the floor, and the wrapper still refuses,
  now naming the blocking surface and the command that clears it.

Same hermetic fixture as ``test_reconcile_host_omn17307.py``: throwaway git
repositories, a hand-built ``site-packages``, stubbed delegates.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.scripts.test_reconcile_host_omn17307 import (
    EXIT_FAILED,
    Workspace,
    _advance_origin,
    _git,
    _lock,
    _make_clone,
    _run,
    _stub,
    _write_dist,
    _writer_stub,
    build_workspace,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WRAPPER = _REPO_ROOT / "scripts" / "onex"

_EXIT_BELOW_FLOOR = 3
_SENTINEL_OK = 41  # the fake CLI's exit status; no shell failure produces it

STALE_COMMIT = "e" * 40


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return build_workspace(tmp_path)


def _stale_floor(ws: Workspace) -> str:
    text = json.dumps(
        {
            "schema": "onex.workspace.floor.v1",
            "generated_at": "2026-09-29T19:55:00Z",
            "host": "test",
            "omni_home": str(ws.root),
            "distributions": {},
            "omnimarket_commit": STALE_COMMIT,
        },
        indent=2,
    )
    ws.floor.write_text(text + "\n", encoding="utf-8")
    return text + "\n"


def _dirty_core_clone(ws: Workspace) -> None:
    """The incident's clone: behind origin, with staged paths, never converged."""
    clone, seed = _make_clone(ws.root, "omnibase_core")
    _advance_origin(seed, "merged-upstream")
    (clone / "staged.txt").write_text("staged\n", encoding="utf-8")
    _git(clone, "add", "staged.txt")


def _market_at_head(ws: Workspace) -> str:
    """omnimarket clone at its target, and the dispatch venv built from it."""
    market, _ = _make_clone(ws.root, "omnimarket")
    head = _git(market, "rev-parse", "HEAD")
    _write_dist(ws.site_packages, "omnimarket", "0.4.11", commit=head)
    return head


def _no_op_delegates(ws: Workspace) -> None:
    _stub(
        ws.scripts / "runtime_build" / "reconcile_deploy_clones.sh", ws.delegate_witness
    )
    _stub(ws.scripts / "reconcile-workspace-venvs.sh", ws.delegate_witness)


def _install_wrapper(ws: Workspace) -> Path:
    """The real wrapper beside the real reconciler, and a fake CLI entrypoint."""
    wrapper = ws.scripts / "onex"
    shutil.copy2(_WRAPPER, wrapper)
    wrapper.chmod(0o755)
    entry = ws.root / ".onex-dispatch-venv" / "bin" / "onex"
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> {ws.root / "argv.log"}\n'
        f"exit {_SENTINEL_OK}\n",
        encoding="utf-8",
    )
    entry.chmod(0o755)
    return wrapper


def _run_wrapper(ws: Workspace, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["OMNI_HOME"] = str(ws.root)
    env.pop("ONEX_DISPATCH_VENV", None)
    env.pop("ONEX_RECONCILE_RECEIPT", None)
    env.pop("ONEX_WRAPPER_NO_RECONCILE", None)
    fixture_home = ws.root.parent / "fixture_home"
    fixture_home.mkdir(parents=True, exist_ok=True)
    env["HOME"] = str(fixture_home)
    env["ONEX_RECONCILE_ALERT_CMD"] = f"{_writer_stub(ws)} {ws.alert_witness}"
    return subprocess.run(
        ["bash", str(ws.scripts / "onex"), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )


# --------------------------------------------------------------------------- #
# AC1 -- an unrelated dirty clone does not block delegation
# --------------------------------------------------------------------------- #
def test_an_unrelated_dirty_clone_still_stamps_the_dispatch_floor(
    ws: Workspace,
) -> None:
    """The verdict stays FAILED and alerts; the floor moves to the installed build.

    Before OMN-20111 the floor kept ``STALE_COMMIT`` here, which is the state
    that refused every ``onex delegate`` for about 90 minutes.
    """
    _dirty_core_clone(ws)
    head = _market_at_head(ws)
    _lock(ws)
    _no_op_delegates(ws)
    _stale_floor(ws)

    proc = _run(ws)

    assert proc.returncode == EXIT_FAILED, proc.stderr
    assert "VERDICT: FAILED" in proc.stderr
    assert "clone:omnibase_core: DID_NOT_MOVE" in proc.stderr
    assert ws.alert_witness.exists(), "an unrelated failure must still alert"
    floor = json.loads(ws.floor.read_text(encoding="utf-8"))
    assert floor["omnimarket_commit"] == head
    assert "do not block onex delegate" in proc.stderr

    receipt = json.loads(ws.receipt.read_text(encoding="utf-8"))
    assert receipt["failures"] == 1
    assert receipt["dispatch_premise_failures"] == 0


def test_delegate_runs_through_the_wrapper_with_an_unrelated_dirty_clone(
    ws: Workspace,
) -> None:
    """The incident replayed through the wrapper: stale floor, dirty core clone.

    The wrapper finds the floor below the installed omnimarket, runs the real
    reconciler once, which fails on omnibase_core but stamps the dispatch floor,
    and the delegate proceeds.
    """
    _dirty_core_clone(ws)
    _market_at_head(ws)
    _lock(ws)
    _no_op_delegates(ws)
    _stale_floor(ws)
    _install_wrapper(ws)

    proc = _run_wrapper(ws, "delegate", "reply with ok")

    assert proc.returncode == _SENTINEL_OK, proc.stderr
    assert "REFUSED" not in proc.stderr
    argv_log = ws.root / "argv.log"
    assert argv_log.read_text(encoding="utf-8").strip() == "delegate reply with ok"


# --------------------------------------------------------------------------- #
# AC3 -- a dispatch-premise failure still withholds the floor
# --------------------------------------------------------------------------- #
def test_a_dirty_omnimarket_clone_withholds_the_floor(ws: Workspace) -> None:
    """omnimarket's clone is what the provider layer is built from.

    Its HEAD is the venv's target, so a clone that did not reach origin makes
    "the venv matches the clone" prove nothing about the build a dispatch should
    run. The floor must stay where it was.
    """
    market, seed = _make_clone(ws.root, "omnimarket")
    head = _git(market, "rev-parse", "HEAD")
    _advance_origin(seed, "unpulled")
    _write_dist(ws.site_packages, "omnimarket", "0.4.11", commit=head)
    _lock(ws)
    _no_op_delegates(ws)
    previous = _stale_floor(ws)

    proc = _run(ws)

    assert proc.returncode == EXIT_FAILED, proc.stderr
    assert "clone:omnimarket: DID_NOT_MOVE" in proc.stderr
    assert "blocks onex delegate: clone:omnimarket" in proc.stderr
    assert "converge-canonical-clone.sh omnimarket --execute" in proc.stderr
    assert ws.floor.read_text(encoding="utf-8") == previous


def test_a_venv_mismatch_withholds_the_floor_even_beside_an_unrelated_failure(
    ws: Workspace,
) -> None:
    """The dispatch venv's omnimarket is not the clone HEAD: a real mismatch."""
    _dirty_core_clone(ws)
    _make_clone(ws.root, "omnimarket")
    _write_dist(ws.site_packages, "omnimarket", "0.4.11", commit="b" * 40)
    _lock(ws)
    _no_op_delegates(ws)
    previous = _stale_floor(ws)

    proc = _run(ws)

    assert proc.returncode == EXIT_FAILED, proc.stderr
    assert "venv:omnimarket: DID_NOT_MOVE" in proc.stderr
    assert "blocks onex delegate: venv:omnimarket" in proc.stderr
    assert ws.floor.read_text(encoding="utf-8") == previous
    receipt = json.loads(ws.receipt.read_text(encoding="utf-8"))
    assert receipt["dispatch_premise_failures"] == 1


def test_a_lock_version_mismatch_withholds_the_floor(ws: Workspace) -> None:
    """A lock-governed distribution below its target is a dispatch failure too."""
    _make_clone(ws.root, "omnibase_core")
    _lock(ws, **{"omnibase-compat": "0.5.6"})
    _write_dist(ws.site_packages, "omnibase_compat", "0.5.5")
    _no_op_delegates(ws)
    previous = _stale_floor(ws)

    proc = _run(ws)

    assert proc.returncode == EXIT_FAILED, proc.stderr
    assert "blocks onex delegate: venv:omnibase-compat" in proc.stderr
    assert "reconcile-workspace-venvs.sh" in proc.stderr
    assert ws.floor.read_text(encoding="utf-8") == previous


def test_the_wrapper_still_refuses_a_real_venv_mismatch_and_names_it(
    ws: Workspace,
) -> None:
    """Fail-closed at the wrapper: the refusal names the surface and its repair."""
    _dirty_core_clone(ws)
    _make_clone(ws.root, "omnimarket")
    _write_dist(ws.site_packages, "omnimarket", "0.4.11", commit="b" * 40)
    _lock(ws)
    _no_op_delegates(ws)
    _stale_floor(ws)
    _install_wrapper(ws)

    proc = _run_wrapper(ws, "delegate", "x")

    assert proc.returncode == _EXIT_BELOW_FLOOR, proc.stderr
    assert "REFUSED" in proc.stderr
    assert "blocking : venv:omnimarket DID_NOT_MOVE" in proc.stderr
    assert "clears by: bash " in proc.stderr
    assert "reconcile-workspace-venvs.sh" in proc.stderr
    assert "not blocking delegation: clone:omnibase_core DID_NOT_MOVE" in proc.stderr
    assert not (ws.root / "argv.log").exists(), "the CLI must not have run"


def test_gate_purity_is_not_part_of_the_dispatch_premise(ws: Workspace) -> None:
    """An impure gate venv breaks `uv run pytest`, not the dispatch build.

    It still fails the verdict; it no longer holds delegation hostage.
    """
    head = _market_at_head(ws)
    gate_sp = ws.infra / ".venv" / "lib" / "python3.12" / "site-packages"
    _write_dist(gate_sp, "omnimarket", "0.4.11")
    _lock(ws)
    _no_op_delegates(ws)
    _stale_floor(ws)

    proc = _run(ws)

    assert proc.returncode == EXIT_FAILED, proc.stderr
    assert "venv:gate-purity: IMPURE" in proc.stderr
    assert json.loads(ws.floor.read_text(encoding="utf-8"))["omnimarket_commit"] == head
