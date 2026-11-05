# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A standalone slot bring-up says when the reaper may remove its slot.

Recorded case (friction class 449): a lane brought slot N up standalone with
``prepr_verify_lane.sh``, read "slot N is READY ... when the proof is done,
destroy it", and exercised the slot. The standalone lease stopped heartbeating
when the bring-up exited, so the reaper removed the slot five minutes later.
The completion message now names the default expiry window and the existing
one-command path that keeps the lease alive; the lease and the reaper are unchanged.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
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


def test_the_notice_matches_what_the_reaper_does_at_that_expiry(
    reaper: Any, tmp_path: Path
) -> None:
    record = standalone_lease(reaper, tmp_path)
    # The owner exits: its pid is gone, the expiry stays as last written.
    record.update(owner_pid=99999999, owner_start="gone")
    reaper.write_record(tmp_path, 1, record)
    before = reaper.reap_slot(tmp_path, 1, plan_only=True, now=record["expires"] - 1)
    at = reaper.reap_slot(tmp_path, 1, plan_only=True, now=record["expires"])
    after = reaper.reap_slot(tmp_path, 1, plan_only=True, now=record["expires"] + 1)
    assert before["verdict"] == "HELD"
    assert at["verdict"] == "WOULD_REAP"
    assert after["verdict"] == "WOULD_REAP"


@pytest.mark.parametrize("inherited_token", ["", "outer-run-token"])
def test_completion_output_names_standalone_expiry_and_safe_path(
    tmp_path: Path, inherited_token: str
) -> None:
    descriptor = tmp_path / "slot.json"
    descriptor.write_text('{"slot": 1}\n')
    # Execute the shipped completion block without building or starting a slot.
    completion = VERIFY.read_text().split('log "slot ${SLOT} is READY.', 1)[1]
    done = subprocess.run(
        [
            "bash",
            "-eu",
            "-c",
            'log() { printf "%s\\n" "$*"; }; log "slot ${SLOT} is READY.' + completion,
        ],
        env={
            **os.environ,
            "SLOT": "1",
            "DESCRIPTOR_OUT": str(descriptor),
            "ONEX_PREPR_LEASE_TOKEN": inherited_token,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    output = done.stdout
    assert "slot 1 is READY." in output
    assert "standalone" in output
    assert "last heartbeat" in output
    assert "300s" in output
    assert (
        "uv run --frozen python scripts/runtime_build/prepr_teardown_slot.py "
        "run --slot 1 --holder <lane> -- bash <proof.sh>" in output
    )
    assert "prepr_teardown_slot.sh --slot 1 --reason <text>" in output
    assert (
        output.index("READY") < output.index("standalone") < output.index('{"slot": 1}')
    )


def test_warning_describes_the_current_default_ttl(reaper: Any) -> None:
    text = VERIFY.read_text().split('log "slot ${SLOT} is READY.', 1)[1]
    assert f"{reaper.LEASE_DEFAULT_TTL:g}s" in text
    assert "by default" in text
