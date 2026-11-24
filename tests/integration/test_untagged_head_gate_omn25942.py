# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Run the untagged-head gate against the recorded tag run of job 7dfb1bc9 (OMN-25942).

The fixture names the dev head, the tag that the Dev Merge Tag run cut for it and the tag
before it. The gate reads them through the same seams the deploy agent hands it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from omnibase_infra.nodes.node_deploy_job_reading_effect.runtime_untagged_head_gate import (
    UntaggedHeadGate,
)

pytestmark = pytest.mark.integration

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "scripts/deploy-agent/tests/unit/fixtures/omn25942/recorded_untagged_head_7dfb1bc9.json"
)
RECORDED = json.loads(FIXTURE.read_text(encoding="utf-8"))
NOW = datetime(2026, 11, 24, 2, 46, 16, tzinfo=UTC)


def _gate(tag_shas: list[str] | None) -> tuple[UntaggedHeadGate, list[str]]:
    rows: list[str] = []
    host: Any = SimpleNamespace(
        _idle_read_head_ref=lambda repository: RECORDED["head"],
        _idle_read_dev_tags=lambda repository: (
            None if tag_shas is None else [SimpleNamespace(sha=sha) for sha in tag_shas]
        ),
        _idle_converge_now=lambda: NOW,
    )
    gate = UntaggedHeadGate(host, lambda head, now: rows.append(head))
    return gate, rows


def _cmd(requested_by: str) -> Any:
    return SimpleNamespace(requested_by=requested_by)


def test_head_without_its_tag_is_refused_and_recorded() -> None:
    gate, rows = _gate([RECORDED["previous_head"]["sha"]])
    assert gate.refuses(_cmd(RECORDED["job"]["requested_by"]))
    assert rows == [RECORDED["head"]]


def test_head_with_its_tag_is_not_refused() -> None:
    gate, rows = _gate([RECORDED["previous_head"]["sha"], RECORDED["head"]])
    assert not gate.refuses(_cmd(RECORDED["job"]["requested_by"]))
    assert rows == []


def test_unread_tags_refuse_nothing() -> None:
    gate, rows = _gate(None)
    assert not gate.refuses(_cmd(RECORDED["job"]["requested_by"]))
    assert rows == []


def test_a_signed_requester_is_never_refused() -> None:
    gate, rows = _gate([RECORDED["previous_head"]["sha"]])
    assert not gate.refuses(_cmd("operator/pinned-fix/devwindow2-2215"))
    assert rows == []
