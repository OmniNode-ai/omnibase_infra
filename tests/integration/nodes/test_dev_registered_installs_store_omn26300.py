# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One registered-install store on a real filesystem, from the legacy store to the listing (OMN-26300).

Where the unit tests judge each rule alone, this walks the life of the store the dev deploy
agent keeps: an install registered in the old ``state/jobs`` location, the agent's first
read merging and retiring it, the listing every held install is served in, the reason
``record_hold`` writes into ``status.json``, and the restore yielding until the readback
shows the install whole. Real directories and files under ``tmp_path``; no fakes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from omnibase_infra.nodes.node_dev_roll_events_effect.runtime_dev_registered_installs import (
    RegisteredInstalls,
)

pytestmark = pytest.mark.integration

_EXPIRES = (datetime.now(UTC) + timedelta(hours=24)).isoformat()


def _register(root: Path, install_id: str, blob: bytes) -> None:
    """Register one install the way the registering lane does: a record and its content."""
    store = root / "branch-installs"
    (store / "content").mkdir(parents=True, exist_ok=True)
    records_path = store / "records.json"
    records = (
        json.loads(records_path.read_text())
        if records_path.exists()
        else {"schema_version": "dev-lane-branch-installs.v1", "installs": []}
    )
    records["installs"].append(
        {"id": install_id, "repo": "omnimarket", "expires_at": _EXPIRES}
    )
    records_path.write_text(json.dumps(records))
    (store / "content" / install_id).write_bytes(blob)


def _readback(state_dir: Path, install_id: str, state: str) -> None:
    status = state_dir / "branch-installs" / "status.json"
    status.write_text(
        json.dumps({"readbacks": [{"install_key": install_id, "state": state}]})
    )


def test_an_install_registered_in_the_old_store_is_listed_held_and_then_released(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "jobs" / "branch-installs"
    state_dir = tmp_path / "jobs-dev"
    _register(tmp_path / "jobs", "omnimarket-6510", b"fix-one")
    _register(state_dir, "omnimarket-6496", b"fix-two")
    agent = RegisteredInstalls(state_dir, legacy)

    # The first read merges the old store into the agent's and retires the old directory.
    listing = agent.listing()
    declared = listing["declared"]
    assert isinstance(declared, list)
    assert sorted(str(i["id"]) for i in declared) == [
        "omnimarket-6496",
        "omnimarket-6510",
    ]
    assert not legacy.exists()
    store = state_dir / "branch-installs"
    assert (store / "content" / "omnimarket-6510").read_bytes() == b"fix-one"
    assert agent.declared_repos() == ["omnimarket"]

    # Nothing has read the containers back yet: both are held, each with a reason.
    held = listing["held"]
    assert isinstance(held, list)
    assert sorted(h["id"] for h in held) == ["omnimarket-6496", "omnimarket-6510"]
    assert all(h["code"] == "not_checked" and h["reason"].strip() for h in held)

    # A restore with the main runtime up yields the window, and the reason is on disk.
    waiting = agent.restore_yield("companion_down")
    assert waiting is not None and "omnimarket-6510" in waiting
    agent.record_hold(
        "waiting_recreate_window", "the hourly recreate window is not open"
    )
    written = json.loads((store / "status.json").read_text())["held"]
    assert {h["id"]: h["reason"] for h in written} == {
        "omnimarket-6496": "the hourly recreate window is not open",
        "omnimarket-6510": "the hourly recreate window is not open",
    }

    # A lane that is down is restored at once: its build applies the installs itself.
    assert agent.restore_yield("down") is None

    # Both installs read back whole: nothing is held and the restore takes the window again.
    status = {
        "readbacks": [
            {"install_key": "omnimarket-6496", "state": "present"},
            {"install_key": "omnimarket-6510", "state": "reapplied"},
        ]
    }
    (store / "status.json").write_text(json.dumps(status))
    assert agent.listing()["held"] == []
    assert agent.restore_yield("companion_down") is None


def test_a_failed_apply_is_held_with_its_reason_but_is_not_waiting(
    tmp_path: Path,
) -> None:
    _register(tmp_path, "omnimarket-1", b"x")
    _readback(tmp_path, "omnimarket-1", "failed")
    agent = RegisteredInstalls(tmp_path, None)

    listed = agent.listing()["held"]
    assert isinstance(listed, list) and len(listed) == 1
    held = listed[0]
    assert held["code"] == "apply_failed" and held["reason"].strip()
    assert agent.restore_yield("companion_down") is None


def test_a_record_file_whose_installs_are_not_mappings_is_listed_unreadable(
    tmp_path: Path,
) -> None:
    store = tmp_path / "branch-installs"
    store.mkdir()
    (store / "records.json").write_text(
        json.dumps({"schema_version": "v1", "installs": ["not-a-mapping"]})
    )
    agent = RegisteredInstalls(tmp_path, None)

    listing = agent.listing()

    assert listing["installs"] is None
    assert "ValueError" in str(listing["reason"])
    assert agent.declared_repos() == []
