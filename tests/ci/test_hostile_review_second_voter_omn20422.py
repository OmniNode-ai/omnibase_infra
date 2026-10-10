# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20422: the Hostile Reviewer's second voter is a different lab model.

RULING 2026-10-09T04:15:56Z keeps Codex out of CI. The recording below is the
Studio's :8131 server as it stood on 2026-10-09. Since OMN-20910 the workflow
names no voter URL: the voters, and which endpoint each reads, are declared in
docker/lane-overlays/hostile-review-voters.yaml (see
test_hostile_reviewer_voters_overlay_omn20910.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/hostile-reviewer.yml"
FIRST_VOTER_MODEL = "Qwen3.8-27B"


@pytest.mark.live_contact(
    "tests/ci/fixtures/hostile_reviewer_second_voter_omn20422.json"
)
def test_second_voter_endpoint_serves_a_different_model_than_the_first(
    recorded_response: dict[str, object],
) -> None:
    recording = cast("dict[str, dict[str, object]]", recorded_response["response"])
    served = [
        cast("str", entry["id"])
        for entry in cast("list[dict[str, object]]", recording["models"]["data"])
    ]
    assert served == ["qwen3.6-35b-a3b"]
    assert FIRST_VOTER_MODEL not in served
    assert recording["completion"]["finish_reason"] == "stop"


def test_the_review_gate_waits_on_the_review_alone() -> None:
    """OMN-20074: no change-control preflight job sits in front of the review."""
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    assert "needs" not in jobs["hostile-review"]
    assert jobs["hostile-review-gate"]["needs"] == "hostile-review"
