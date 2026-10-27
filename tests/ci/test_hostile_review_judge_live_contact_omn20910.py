# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Replay a real GitHub rejection through the judge's head-bound protocol.

The gateway recording supplies the finding, rejection, comment ids and reviewed
head. Verdicts below are explicit protocol controls produced by the real code,
not claims that a model judged this finding. No API seam is replaced.
"""

from __future__ import annotations

import copy
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any, cast

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts/ci/check_hostile_review_threads.py"
)
_spec = importlib.util.spec_from_file_location("judge_contact_gate", _SCRIPT)
assert _spec is not None and _spec.loader is not None
gate = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gate
_spec.loader.exec_module(gate)


def _recorded_thread(recording: dict[str, object]) -> tuple[dict[str, Any], str]:
    """Translate the gateway's recorded projection back to GraphQL field names."""
    wire = cast("dict[str, Any]", recording["response"])
    comments = [
        {
            "databaseId": c["comment_id"],
            "body": c["body"],
            "url": c["url"],
            "author": {"login": c["author"]},
        }
        for c in wire["comments"]
    ]
    match = re.search(r"Evidence at PR head ([0-9a-f]{40})", comments[1]["body"])
    assert match is not None, "the recorded rejection identifies its code head"
    return {"id": wire["thread_id"], "comments": {"nodes": comments}}, match[1]


@pytest.mark.live_contact(
    "tests/fixtures/omn20910/infra-pr5368-recorded-rejection.json"
)
def test_recorded_rejection_round_trips_only_at_the_reviewed_head(
    recorded_response: dict[str, object],
) -> None:
    thread, head = _recorded_thread(recorded_response)
    disposition = gate.latest_disposition(thread)
    assert disposition is not None and disposition.kind == "rejected"
    assert disposition.comment_id == 4377974641
    body = gate.verdict_body(
        rejection_id=disposition.comment_id,
        digest=disposition.digest,
        head_sha=head,
        verdict="accepted",
        judge="rejection-judge",
        identity="protocol-replay-control",
        sentence="Protocol control for the recorded rejection and reviewed code.",
    )
    assert gate.parse_judge_reply(body)["head"] == head
    thread["comments"]["nodes"].append(
        {"body": body, "author": {"login": "onexbot", "__typename": "Bot"}}
    )
    assert (
        gate.judge_verdict_for(thread, disposition, head_sha=head).verdict == "accepted"
    )
    # A different head must not inherit this recorded rejection's judgement.
    other_head = ("0" if head[0] != "0" else "1") + head[1:]
    assert gate.judge_verdict_for(thread, disposition, head_sha=other_head) is None


@pytest.mark.live_contact(
    "tests/fixtures/omn20910/infra-pr5368-recorded-rejection.json"
)
def test_recorded_rejection_rejects_legacy_and_wrong_identity_controls(
    recorded_response: dict[str, object],
) -> None:
    thread, head = _recorded_thread(recorded_response)
    disposition = gate.latest_disposition(thread)
    assert disposition is not None
    body = gate.verdict_body(
        rejection_id=disposition.comment_id,
        digest=disposition.digest,
        head_sha=head,
        verdict="refused",
        judge="rejection-judge",
        identity="protocol-replay-control",
        sentence="A refused verdict must also retain the reviewed head.",
    )
    for marker, login, kind in (
        (body.replace(f"head={head} ", ""), "onexbot", "Bot"),
        (body, "github-actions", "Bot"),
        (body, "onexbot", "User"),
        (body.replace(disposition.digest, "0" * 64), "onexbot", "Bot"),
    ):
        replay = copy.deepcopy(thread)
        replay["comments"]["nodes"].append(
            {"body": marker, "author": {"login": login, "__typename": kind}}
        )
        assert gate.judge_verdict_for(replay, disposition, head_sha=head) is None
        assert gate.latest_disposition(replay) == disposition
