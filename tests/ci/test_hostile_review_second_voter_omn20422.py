# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20422: the Hostile Reviewer's second voter is the Studio's one served model.

RULING 2026-10-09T04:15:56Z keeps Codex out of CI. RULING 2026-10-10T16:28:58Z puts one
model on the Mac Studio; the recording below is its port 8130 serving Qwen3.8-27B, the
same model the first voter serves on its own host. Since OMN-20910 the workflow names no
voter URL or model: the voters, and which endpoint each reads, are declared in a
review voters overlay that, since OMN-20923, is a private CI overlay and not part of
this repository (see test_hostile_reviewer_voters_overlay_omn20910.py); a distinct
second voter (OMN-20889) is an edit to that overlay.
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
def test_second_voter_endpoint_is_the_studio_single_served_model(
    recorded_response: dict[str, object],
) -> None:
    recording = cast("dict[str, dict[str, object]]", recorded_response["response"])
    served = [
        cast("str", entry["id"])
        for entry in cast("list[dict[str, object]]", recording["models"]["data"])
    ]
    # One model on the Studio, and it is the first voter's model on another host.
    assert served == [FIRST_VOTER_MODEL]
    assert recording["completion"]["finish_reason"] == "stop"


def test_the_review_gate_waits_on_the_review_alone() -> None:
    """OMN-20074: no change-control preflight job sits in front of the review."""
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    assert "needs" not in jobs["hostile-review"]
    assert jobs["hostile-review-gate"]["needs"] == "hostile-review"
