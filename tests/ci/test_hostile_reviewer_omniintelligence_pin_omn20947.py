# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The pinned omniintelligence accepts substitute votes and overflow endpoints (OMN-20947).

The Hostile Reviewer validates the private CI overlay with omniintelligence's
review voters overlay model, which forbids unknown keys. An overlay that sets
``substitute_votes`` or a voter's ``overflow_endpoint_urls`` is refused by every
revision older than the one pinned here ("Extra inputs are not permitted"), so
the gate fails closed on every pull request the moment such an overlay lands.

The recording is the pinned revision's own overlay schema and its loader's
result over synthetic overlays, with the previous pin as the control that
refuses them. Every workflow site that clones omniintelligence must name the
recorded revision.

The pinned gate also lets clean signed off-CI receipts stand in for a voter
lost to a timeout or an unavailable endpoint (a 503, a refused connection, an
unavailable served-model read), but reads them only for the commit named by
``--head-sha``. The review step passes the pull request's head commit, never
the merge ref the job checks out; the recording shows the pinned cli_review
accepts the flag and its classifier covers each lost-voter error the previous
pin did not.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import pytest
import yaml

pytestmark = [pytest.mark.unit]

_ROOT = Path(__file__).resolve().parents[2]
_SIBLING_PINS = _ROOT / ".github" / "sibling-pins.yaml"
_WORKFLOWS = _ROOT / ".github" / "workflows"
_NEW_SHAPES = (
    "substitute_votes",
    "overflow_endpoints",
    "substitute_votes_and_overflow",
)


def _pin() -> str:
    pins = yaml.safe_load(_SIBLING_PINS.read_text(encoding="utf-8"))["pins"]
    return cast("str", pins["omniintelligence"])


def test_every_workflow_clones_omniintelligence_at_the_sibling_pin() -> None:
    pin = _pin()
    shas: set[str] = set()
    sites = 0
    for path in sorted(_WORKFLOWS.glob("*.y*ml")):
        # OMN-20910: the Hostile Reviewer reads omniintelligence by its newest
        # dev tag (tests/ci/test_hostile_reviewer_dev_tag_key_omn20910.py).
        if path.name == "hostile-reviewer.yml":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if "omniintelligence" in line and "checkout --quiet" in line:
                shas.update(re.findall(r"\b[0-9a-f]{40}\b", line))
                sites += 1
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job in (parsed.get("jobs") or {}).values():
            for step in job.get("steps") or []:
                with_ = step.get("with") or {}
                if with_.get("repository") == "OmniNode-ai/omniintelligence":
                    shas.add(str(with_.get("ref")))
                    sites += 1
    assert sites >= 4, sites
    assert shas == {pin}


@pytest.mark.live_contact(
    "tests/ci/fixtures/hostile_reviewer_overlay_schema_pin_omn20947.json"
)
def test_pinned_omniintelligence_accepts_substitute_votes_and_overflow(
    recorded_response: dict[str, object],
) -> None:
    provenance = cast("dict[str, str]", recorded_response["_provenance"])
    pinned, control = provenance["pinned_revision"], provenance["control_revision"]
    assert pinned == _pin()
    revisions = cast(
        "dict[str, dict[str, dict[str, object]]]", recorded_response["revisions"]
    )

    schema = cast("dict[str, list[str]]", revisions[pinned]["schema"])
    assert "substitute_votes" in schema["overlay_properties"]
    assert {"overflow_endpoint_urls", "overflow_after_seconds"} <= set(
        schema["voter_properties"]
    )
    assert {
        "reviewers",
        "receipts_per_vote",
        "receipt_max_age_seconds",
        "signing_key_env",
    } <= set(schema["substitute_vote_policy_properties"])

    loaded = cast("dict[str, dict[str, object]]", revisions[pinned]["loaded"])
    refused = cast("dict[str, dict[str, object]]", revisions[control]["loaded"])
    # Positive control: the overlay every revision accepts loads at both.
    assert loaded["voters_only"]["ok"] is True
    assert refused["voters_only"]["ok"] is True
    for shape in _NEW_SHAPES:
        assert loaded[shape]["ok"] is True, loaded[shape]["error"]
        assert refused[shape]["ok"] is False, shape
        assert "Extra inputs are not permitted" in str(refused[shape]["error"])
    assert loaded["substitute_votes_and_overflow"]["overflow_endpoint_count"] == 1
    assert loaded["substitute_votes_and_overflow"]["substitute_reviewer_ids"] == [
        "offci-reviewer-a",
        "offci-reviewer-b",
    ]


_HEAD_SHA_EXPR = "${{ github.event.pull_request.head.sha }}"
_HEAD_SHA_ARG = '--head-sha "$HEAD_SHA"'


def _head_sha_problems(workflow_text: str) -> list[str]:
    """Why the review step does not hand cli_review the pull request head."""
    parsed = yaml.safe_load(workflow_text)
    steps = [
        step
        for job in (parsed.get("jobs") or {}).values()
        for step in job.get("steps") or []
        if step.get("id") == "review"
    ]
    assert len(steps) == 1, steps
    (step,) = steps
    problems: list[str] = []
    if (step.get("env") or {}).get("HEAD_SHA") != _HEAD_SHA_EXPR:
        problems.append("HEAD_SHA is not the pull request head sha")
    run = str(step.get("run", ""))
    assert "omniintelligence.review_pairing.cli_review" in run
    if _HEAD_SHA_ARG not in run:
        problems.append("cli_review is not passed --head-sha")
    return problems


def test_the_review_step_passes_the_pull_request_head_sha() -> None:
    text = (_WORKFLOWS / "hostile-reviewer.yml").read_text(encoding="utf-8")
    assert _head_sha_problems(text) == []
    # Positive control: the same step with the head removed is caught.
    before = re.sub(r"^\s+HEAD_SHA: .*\n", "", text, flags=re.MULTILINE).replace(
        f"            {_HEAD_SHA_ARG} \\\n", ""
    )
    assert _head_sha_problems(before) == [
        "HEAD_SHA is not the pull request head sha",
        "cli_review is not passed --head-sha",
    ]


@pytest.mark.live_contact(
    "tests/ci/fixtures/hostile_reviewer_overlay_schema_pin_omn20947.json"
)
def test_pinned_omniintelligence_substitutes_receipts_for_a_lost_voter(
    recorded_response: dict[str, object],
) -> None:
    provenance = cast("dict[str, str]", recorded_response["_provenance"])
    pinned, control = provenance["pinned_revision"], provenance["control_revision"]
    previous = provenance["previous_pinned_revision"]
    assert pinned == _pin()

    cli = cast("dict[str, dict[str, bool]]", recorded_response["cli_review"])
    assert cli[pinned]["accepts_head_sha"] is True
    assert cli[control]["accepts_head_sha"] is False

    substitution = cast(
        "dict[str, dict[str, dict[str, object]]]", recorded_response["substitution"]
    )
    at_pin = {k: v["substitutable_as"] for k, v in substitution[pinned].items()}
    before = {k: v["substitutable_as"] for k, v in substitution[previous].items()}
    assert at_pin == {
        "call_timed_out": "timed out",
        "call_503": "was unavailable",
        "call_connection_refused": "was unavailable",
        "models_read_503": "was unavailable",
        "call_500": None,
    }
    # Control: the previous pin substitutes only for a timeout.
    assert before == {
        "call_timed_out": "timed out",
        "call_503": None,
        "call_connection_refused": None,
        "models_read_503": None,
        "call_500": None,
    }
