# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Golden and error chains of the staging lock and floor promotion (OMN-16932).

Follows ``test_venv_atomic_install_chain_omn22712.py``: the typed handler of
``node_venv_atomic_install`` and its runtime command line, over a tmp venv whose
site-packages carries a real ``omnimarket`` dist-info (``direct_url.json`` with a
commit), so nothing needs a network and nothing outside the tmp tree is written.

Golden: the floor advances with the venv; the staging lock is taken, seen as
held, released. Error: a live holder declines the lock (status ``declined``,
exit 4, also from the command line); a venv with no dist-info cannot advance the
floor (status ``failed``, floor bytes untouched).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_infra.handlers.venv_atomic_install.handler_venv_atomic_install import (
    HandlerVenvAtomicInstall,
)
from omnibase_infra.models.model_venv_atomic_install_request import (
    ModelVenvAtomicInstallRequest,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME = (
    _REPO_ROOT
    / "src/omnibase_infra/nodes/node_venv_atomic_install/runtime_venv_atomic_install.py"
)
_OLD = "a" * 40
_NEW = "b" * 40


def _handler() -> HandlerVenvAtomicInstall:
    return HandlerVenvAtomicInstall(ModelONEXContainer())


def _venv_with_commit(root: Path, commit: str | None) -> Path:
    venv = root / "venv"
    dist = venv / "lib" / "python3.12" / "site-packages" / "omnimarket-0.4.11.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: omnimarket\nVersion: 0.4.11\n", encoding="utf-8"
    )
    if commit is not None:
        (dist / "direct_url.json").write_text(
            json.dumps(
                {
                    "url": "https://example.invalid/omnimarket.git",
                    "vcs_info": {"vcs": "git", "commit_id": commit},
                }
            ),
            encoding="utf-8",
        )
    return venv


def _floor(root: Path, commit: str) -> Path:
    floor = root / ".onex-workspace-floor.json"
    floor.write_text(
        json.dumps({"omnimarket_commit": commit, "distributions": {}}),
        encoding="utf-8",
    )
    return floor


def _promote(
    root: Path, venv: Path, floor: Path, *, changed: bool
) -> ModelVenvAtomicInstallRequest:
    lock = root / "uv.lock"
    lock.write_text("", encoding="utf-8")
    return ModelVenvAtomicInstallRequest(
        command="promote_floor",
        venv=venv,
        uv_lock=lock,
        workspace_root=root,
        floor_path=floor,
        changed=changed,
    )


def _lock_request(command: str, path: Path, pid: int) -> ModelVenvAtomicInstallRequest:
    return ModelVenvAtomicInstallRequest.model_validate(
        {"command": command, "lock_path": path, "holder_pid": pid}
    )


def test_golden_chain_the_floor_advances_with_the_venv(tmp_path: Path) -> None:
    venv = _venv_with_commit(tmp_path, _NEW)
    floor = _floor(tmp_path, _OLD)

    result = _handler().handle(_promote(tmp_path, venv, floor, changed=True))

    assert (result.status, result.exit_code) == ("ok", 0), result.output
    assert json.loads(floor.read_text(encoding="utf-8"))["omnimarket_commit"] == _NEW


def test_golden_chain_the_staging_lock_is_taken_seen_and_released(
    tmp_path: Path,
) -> None:
    handler = _handler()
    lock = tmp_path / "venv.rebuilding.lock"

    acquired = handler.handle(_lock_request("staging_lock_acquire", lock, os.getpid()))
    held = handler.handle(_lock_request("staging_lock_held", lock, os.getpid()))
    released = handler.handle(_lock_request("staging_lock_release", lock, os.getpid()))

    assert acquired.status == "ok", acquired.output
    assert held.status == "ok", held.output
    assert released.status == "ok", released.output
    assert not lock.is_symlink()
    assert (
        handler.handle(_lock_request("staging_lock_held", lock, os.getpid())).exit_code
        == 1
    )


def test_error_chain_a_live_holder_declines_the_lock(tmp_path: Path) -> None:
    holder = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    lock = tmp_path / "venv.rebuilding.lock"
    record = f"pid={holder.pid};host={socket.gethostname()};at=2026-11-13T00:00:00Z"
    lock.symlink_to(record)
    try:
        result = _handler().handle(
            _lock_request("staging_lock_acquire", lock, os.getpid())
        )
        assert (result.status, result.exit_code) == ("declined", 4), result.output
        assert "DECLINED" in result.output
        assert f"pid {holder.pid}" in result.output
        assert str(lock.readlink()) == record, "the declined run touched the lock"

        # The same refusal through the runtime command line the shell calls.
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        proc = subprocess.run(
            [
                sys.executable,
                str(_RUNTIME),
                "staging-lock-acquire",
                "--path",
                str(lock),
                "--pid",
                str(os.getpid()),
            ],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        assert proc.returncode == 4, proc.stdout + proc.stderr
        assert "DECLINED" in proc.stdout
    finally:
        holder.kill()
        holder.wait()


def test_error_chain_a_venv_without_a_dist_info_cannot_advance_the_floor(
    tmp_path: Path,
) -> None:
    venv = _venv_with_commit(tmp_path, None)
    floor = _floor(tmp_path, _OLD)
    before = floor.read_bytes()

    result = _handler().handle(_promote(tmp_path, venv, floor, changed=True))

    assert (result.status, result.exit_code) == ("failed", 2), result.output
    assert floor.read_bytes() == before
