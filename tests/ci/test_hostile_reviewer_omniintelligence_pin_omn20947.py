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
    assert sites >= 5, sites
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
