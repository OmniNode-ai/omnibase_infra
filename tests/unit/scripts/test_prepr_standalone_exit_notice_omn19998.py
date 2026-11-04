# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A standalone slot bring-up says when the reaper may remove its slot.

Recorded case (friction class 449): a lane brought slot N up standalone with
``prepr_verify_lane.sh``, read "slot N is READY ... when the proof is done,
destroy it", and exercised the slot. The standalone lease stopped heartbeating
when the bring-up exited, so the reaper removed the slot five minutes later.
The bring-up now names the lease expiry and the one-command path that keeps the
lease alive; the lease and the reaper are unchanged.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from tests.helpers.prepr_pool_overlay import pool_overlay_env

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "scripts/runtime_build/prepr_teardown_slot.py"
VERIFY = ROOT / "scripts/runtime_build/prepr_verify_lane.sh"
pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("pool_overlay_env")]


@pytest.fixture
def reaper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(HELPER.parent))
    monkeypatch.setenv("ONEX_LANE_LOCK_DIR", str(tmp_path / "lane-locks"))
    spec = importlib.util.spec_from_file_location("slot_exit_notice_test", HELPER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def standalone_lease(reaper: Any, root: Path) -> dict[str, Any]:
    return reaper.claim(
        root,
        1,
        "standalone-token",
        "standalone-4242",
        str(reaper.STAGING_PARENT / "slot-1"),
    )


def test_notice_names_the_recorded_expiry_and_the_one_command_path(
    reaper: Any, tmp_path: Path
) -> None:
    record = standalone_lease(reaper, tmp_path)
    notice = reaper.exit_notice(tmp_path, 1, "standalone-token")
    expires = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record["expires"]))
    assert f"expires at {expires}" in notice
    assert "(300s after its last heartbeat)" in notice
    assert "standalone-4242" in notice
    assert "prepr_teardown_slot.py run --slot 1 --holder <lane> -- bash" in notice


def test_the_notice_matches_what_the_reaper_does_at_that_expiry(
    reaper: Any, tmp_path: Path
) -> None:
    record = standalone_lease(reaper, tmp_path)
    # The owner exits: its pid is gone, the expiry stays as last written.
    record.update(owner_pid=99999999, owner_start="gone")
    reaper.write_record(tmp_path, 1, record)
    before = reaper.reap_slot(tmp_path, 1, plan_only=True, now=record["expires"] - 1)
    after = reaper.reap_slot(tmp_path, 1, plan_only=True, now=record["expires"] + 1)
    assert before["verdict"] == "HELD"
    assert after["verdict"] == "WOULD_REAP"


def test_notice_refuses_another_runs_lease(reaper: Any, tmp_path: Path) -> None:
    standalone_lease(reaper, tmp_path)
    with pytest.raises(reaper.LeaseRefusedError):
        reaper.exit_notice(tmp_path, 1, "other-token")


def test_exit_notice_cli_reads_the_inherited_token(reaper: Any, tmp_path: Path) -> None:
    standalone_lease(reaper, tmp_path)
    env = {
        **os.environ,
        "ONEX_PREPR_LEASE_DIR": str(tmp_path),
        "ONEX_PREPR_LEASE_TOKEN": "standalone-token",
    }
    done = subprocess.run(
        [sys.executable, str(HELPER), "exit-notice", "--slot", "1"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "the reaper removes the slot once the lease expires at" in done.stdout
    refused = subprocess.run(
        [sys.executable, str(HELPER), "exit-notice", "--slot", "1"],
        env={**env, "ONEX_PREPR_LEASE_TOKEN": "other-token"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode == 8
    assert "belongs to another run" in refused.stderr


def test_only_a_standalone_bringup_prints_the_notice_after_ready() -> None:
    text = VERIFY.read_text()
    assert 'ONEX_PREPR_STANDALONE=1 exec "${PY}" "${LEASE_PY}" run' in text
    ready = text.index('log "slot ${SLOT} is READY.')
    gate = text.index('if [[ "${ONEX_PREPR_STANDALONE:-}" == "1" ]]; then')
    notice = text.index('"${PY}" "${LEASE_PY}" exit-notice --slot "${SLOT}"')
    assert ready < gate < notice
