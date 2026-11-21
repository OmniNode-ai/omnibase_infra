# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Registered branch installs: one store, a reason for every hold, who yields (OMN-26300)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from omnibase_infra.enums.enum_dev_lane_branch_install_hold import (
    EnumDevLaneBranchInstallHold,
)
from omnibase_infra.enums.enum_dev_lane_branch_install_state import (
    EnumDevLaneBranchInstallState,
)
from omnibase_infra.models.model_dev_lane_branch_install_readback import (
    ModelDevLaneBranchInstallReadback,
)
from omnibase_infra.models.model_dev_lane_branch_install_receipt import (
    ModelDevLaneBranchInstallReceipt,
)
from omnibase_infra.nodes.node_dev_roll_events_effect import (
    runtime_dev_registered_installs as installs,
)

pytestmark = pytest.mark.unit


def _store(state: Path, ids: list[str], blobs: dict[str, bytes] | None = None) -> None:
    root = state / "branch-installs"
    root.mkdir(parents=True)
    expires = (datetime.now(UTC) + timedelta(hours=24)).isoformat()
    records = {
        "schema_version": "dev-lane-branch-installs.v1",
        "installs": [
            {"id": i, "repo": "omnimarket", "expires_at": expires} for i in ids
        ],
    }
    (root / "records.json").write_text(json.dumps(records))
    for name, data in (blobs or {}).items():
        (root / "content").mkdir(exist_ok=True)
        (root / "content" / name).write_bytes(data)


def test_the_hold_codes_the_module_writes_are_the_enum_values() -> None:
    assert {
        installs.HOLD_WINDOW,
        installs.HOLD_LANE_BUSY,
        installs.HOLD_NOT_APPLIED,
        installs.HOLD_NOT_CHECKED,
        installs.HOLD_APPLY_FAILED,
    } == {m.value for m in EnumDevLaneBranchInstallHold}


def test_the_legacy_store_is_merged_and_retired(tmp_path: Path) -> None:
    _store(tmp_path / "jobs", ["old-1", "both"], {"aa": b"old"})
    _store(tmp_path / "jobs-dev", ["new-1", "both"], {"bb": b"new"})
    legacy = tmp_path / "jobs/branch-installs"

    moved = installs.migrate_legacy_store(tmp_path / "jobs-dev", legacy)

    assert moved == ("old-1",)
    assert sorted(
        i["id"] for i in installs.read_declared(tmp_path / "jobs-dev", legacy)
    ) == [
        "both",
        "new-1",
        "old-1",
    ]
    assert (tmp_path / "jobs-dev/branch-installs/content/aa").read_bytes() == b"old"
    assert not (tmp_path / "jobs/branch-installs").exists()
    assert installs.migrate_legacy_store(tmp_path / "jobs-dev", legacy) == ()


def test_a_unit_that_declares_no_legacy_store_migrates_nothing(tmp_path: Path) -> None:
    _store(tmp_path / "jobs", ["old-1"])
    assert installs.migrate_legacy_store(tmp_path / "jobs-dev", None) == ()
    assert (tmp_path / "jobs/branch-installs").exists()


def test_every_held_install_carries_a_reason_and_a_whole_one_is_not_held(
    tmp_path: Path,
) -> None:
    _store(tmp_path, ["a", "b", "c"])
    status = tmp_path / "branch-installs" / "status.json"
    status.write_text(
        json.dumps(
            {
                "readbacks": [
                    {"install_key": "a", "state": "present"},
                    {"install_key": "b", "state": "failed"},
                ]
            }
        )
    )
    held = {h["id"]: h for h in installs.held_installs(tmp_path, None)}
    assert sorted(held) == ["b", "c"]
    assert held["b"]["code"] == "apply_failed" and held["c"]["code"] == "not_checked"
    assert all(h["reason"].strip() for h in held.values())
    assert installs.waiting(tmp_path, None) is not None
    assert "b (" not in (installs.waiting(tmp_path, None) or ""), (
        "a failed apply is not waiting"
    )


def test_only_a_running_main_runtime_makes_the_restore_yield(tmp_path: Path) -> None:
    _store(tmp_path, ["a"])
    assert installs.restore_yield("companion_down", tmp_path, None) is not None
    assert installs.restore_yield("down", tmp_path, None) is None


def test_the_receipt_names_each_install_it_holds() -> None:
    readback = ModelDevLaneBranchInstallReadback(
        install_key="k",
        repo="omnimarket",
        pr=1,
        head="a" * 40,
        ticket="OMN-1",
        expires_at=datetime.now(UTC),
        state=EnumDevLaneBranchInstallState.MISSING,
        files=(),
    )
    receipt = ModelDevLaneBranchInstallReceipt(
        checked_at=datetime.now(UTC), compose_project="p", readbacks=(readback,)
    )
    [held] = json.loads(receipt.model_dump_json())["held"]
    assert held["id"] == "k" and held["code"] == "not_applied" and held["reason"]
