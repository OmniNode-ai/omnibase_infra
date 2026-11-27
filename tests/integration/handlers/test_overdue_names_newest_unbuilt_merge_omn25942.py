# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration replay: the overdue build names the newest unbuilt runtime merge (OMN-25942).

Runs the real overdue request path against a real git history and the committed record
``tests/fixtures/overdue_build_oldest_contained_omn25942/build_decision_rows.json``; only the
bus sender is replaced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.unit.handlers import test_overdue_names_newest_unbuilt_merge_omn25942 as unit
from tests.unit.handlers.test_dev_merge_build_one_request_omn26316 import (
    _SETTINGS,
    _decision,
    _Sender,
)

pytestmark = pytest.mark.integration

_FIXTURE = unit._FIXTURE


def test_recorded_rows_replay_to_the_newest_unbuilt_runtime_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = json.loads(_FIXTURE.read_text())
    assert record["overdue_publisher"]["names"] == "9f77818e"
    caller = unit.caller
    for relative in caller._OVERDUE_MODULES:
        caller._load(relative)
    sender = _Sender()
    monkeypatch.setattr(caller, "AgentDevMergeSender", lambda *_a: sender)
    monkeypatch.setattr(caller, "_TRACKED_HEAD", "HEAD")
    subjects_in_order = [
        "no dev build by 17:56Z (#6317)",
        "newer runtime merge (#6318)",
        "newest runtime merge (#6320)",
        "docs only (#6325)",
    ]
    execution, shas = unit._history(tmp_path, subjects_in_order)
    context = caller.OverdueContext(
        (unit._running(shas[0]),),
        dict.fromkeys(shas, "omnibase_infra"),
        {
            "omnibase_infra": (
                (shas[3], "NON_RUNTIME"),
                (shas[2], "RUNTIME"),
                (shas[1], "RUNTIME"),
                (shas[0], "RUNTIME"),
            )
        },
    )
    caller._request_overdue_builds(
        execution,
        _SETTINGS,
        [_decision(shas[0])],
        dict(zip(shas, subjects_in_order, strict=True)),
        context,
    )
    assert [c.pr_number for c in sender.commands] == [6320]
    assert shas[2] in sender.commands[0].rollback_reason
