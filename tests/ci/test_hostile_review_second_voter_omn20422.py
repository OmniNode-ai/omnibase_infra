# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20422: the Hostile Reviewer's second voter is the Studio's one served model.

RULING 2026-10-09T04:15:56Z keeps Codex out of CI. RULING 2026-10-10T16:28:58Z puts one
model on the Mac Studio and retires the :8131 Qwen3.6-35B-A3B reviewer seat, so the
local-studio-planner key takes its URL from the repository variable
``LLM_LOCAL_STUDIO_PLANNER_URL``, set to the Studio's port 8130, which serves Qwen3.8-27B.
That is the same model qwen3-review serves on its own host: the two voters are distinct
endpoints, which is what the quorum's identity rule counts, and not distinct models. These tests pin that state honestly so a later change
to a distinct second voter (OMN-20889) has to update them on purpose.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/hostile-reviewer.yml"
FIRST_VOTER_MODEL = "Qwen3.8-27B"


def _job_env() -> dict[str, str]:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return cast("dict[str, str]", workflow["jobs"]["hostile-review"]["env"])


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


def test_review_and_preflight_share_one_second_voter_url_from_a_variable() -> None:
    """One job-level value feeds both steps; no lab address is written in this public file."""
    assert (
        _job_env()["LLM_LOCAL_STUDIO_PLANNER_URL"]
        == "${{ vars.LLM_LOCAL_STUDIO_PLANNER_URL }}"
    )


def test_the_review_gate_waits_on_the_review_alone() -> None:
    """OMN-20074: no change-control preflight job sits in front of the review."""
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    assert "needs" not in jobs["hostile-review"]
    assert jobs["hostile-review-gate"]["needs"] == "hostile-review"


def test_the_quorum_keys_are_unchanged() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "--model qwen3-review" in text
    assert "--model local-studio-planner" in text
    assert 'REVIEW_MODEL_KEYS: "qwen3-review local-studio-planner"' in text
