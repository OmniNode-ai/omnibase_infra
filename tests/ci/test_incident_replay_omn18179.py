# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Incident replay for the auto-merge hold gate (OMN-18179, OMN-15547 R5).

The regression being replayed is a false_green with no guard at all in the
path: on 2026-09-10 the "Enable Auto-Merge" job answered "arm" for a PR
whose auto-merge a human had explicitly turned off 95 minutes earlier, and
it answered that with every check green and nothing in any log saying a
decision had been overridden. The PR was `omninode_infra#1310`, `feat(OMN-14201):
scale the onex-prod namespace down to the website` -- 13 Deployments to zero
replicas -- and containment was manual: convert to draft, because draft was
the only pause anyone trusted to hold.

These tests drive the REAL decision module over the REAL captured bytes: the
GraphQL response for that PR's auto-merge timeline, exactly as the workflow
step fetches it, byte for byte as `gh api graphql` returned it. The
extraction below is the same jq expression the workflow runs, so the module
sees what it sees in production.

The discriminator matters here and is not a formality. A gate that answered
"hold" unconditionally would satisfy the first test and be useless, and it
would look identical from outside to one that works -- the auto-merge job
would simply stop arming anything, which nobody would notice quickly because
"the PR did not merge yet" is the normal state of a PR. So the same function
is driven over a second live capture, `omnibase_core#1678`, which was armed
on open and never disabled, and is required to answer "arm".
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "ci" / "check_auto_merge_hold.py"
_FIXTURES = _REPO_ROOT / "tests" / "fixtures" / "omn18179"

# The PR that was armed over a human's disable. Captured 2026-09-11.
_DISABLED_CAPTURE = _FIXTURES / "omninode_infra-1310-automerge-timeline.json.captured"
_DISABLED_SHA256 = "57879f431b99418a9b8402360be14d28d35b9171e858f0f7c200f9c34a6b267f"

# The discriminator: armed on open, never disabled. Captured 2026-09-11.
_ARMED_CAPTURE = _FIXTURES / "omnibase_core-1678-automerge-timeline.json.captured"
_ARMED_SHA256 = "ac8cf1a8d1bae8cfd5ba21daa0fceca258f4fc64fdd721603928ee92f875425a"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_auto_merge_hold", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def _extract(capture: Path) -> tuple[list[dict[str, Any]] | None, list[str] | None]:
    """Pull timeline and labels out of a capture the way the workflow does.

    This mirrors the two jq expressions in auto-merge.yml, including their
    "anything that is not an array is undetermined" shape, so the module is
    fed the same values in the replay as in production.
    """
    payload = json.loads(capture.read_text(encoding="utf-8"))
    pull_request = payload.get("data", {}).get("repository", {}).get("pullRequest", {})

    raw_timeline = pull_request.get("timelineItems", {}).get("nodes")
    timeline = raw_timeline if isinstance(raw_timeline, list) else None

    raw_labels = pull_request.get("labels", {}).get("nodes")
    labels = (
        [node["name"] for node in raw_labels] if isinstance(raw_labels, list) else None
    )
    return timeline, labels


def _assert_capture_integrity(capture: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256(capture.read_bytes()).hexdigest()
    assert digest == expected_sha256, (
        f"{capture.name} no longer matches the sha256 recorded in "
        f"tests/incident_replays/registry.yaml; a reformatted artifact is no "
        f"longer the artifact that failed (expected {expected_sha256}, got {digest})"
    )


def test_the_real_guard_holds_the_pr_that_was_armed_over_a_human_disable() -> None:
    """omninode_infra#1310: the prod scale-down that draft-mode had to contain."""
    _assert_capture_integrity(_DISABLED_CAPTURE, _DISABLED_SHA256)
    timeline, labels = _extract(_DISABLED_CAPTURE)

    # The ordering the whole gate turns on, asserted from the captured bytes
    # rather than restated: armed, then disabled by a person.
    assert timeline is not None
    assert [node["__typename"] for node in timeline] == [
        "AutoSquashEnabledEvent",
        "AutoMergeDisabledEvent",
    ]
    assert timeline[-1]["reason"] == "Manually disabled by user"

    assert mod.latest_auto_merge_decision(timeline) == "disable"
    assert mod.should_hold(timeline, labels) is True


def test_the_same_guard_arms_a_pr_nobody_disabled() -> None:
    """The discriminator: a gate stuck on "hold" would be useless and invisible.

    omnibase_core#1678 was armed on open at 2026-09-11T10:40:33Z and never
    disabled, so the correct answer is to arm. A module that simply held
    everything cannot satisfy this case.
    """
    _assert_capture_integrity(_ARMED_CAPTURE, _ARMED_SHA256)
    timeline, labels = _extract(_ARMED_CAPTURE)

    assert timeline is not None
    assert [node["__typename"] for node in timeline] == ["AutoSquashEnabledEvent"]

    assert mod.latest_auto_merge_decision(timeline) == "enable"
    assert mod.should_hold(timeline, labels) is False


def test_the_captured_enable_event_is_not_the_one_a_naive_detector_looks_for() -> None:
    """Both live captures carry AutoSquashEnabledEvent, not AutoMergeEnabledEvent.

    This is why the detector matches all three enable variants. A filter
    naming only `AutoMergeEnabledEvent` reads an empty enable history for
    every PR this workflow has ever armed, which is silent: it does not
    error, it just never sees the arm.
    """
    for capture in (_DISABLED_CAPTURE, _ARMED_CAPTURE):
        timeline, _ = _extract(capture)
        assert timeline is not None
        typenames = {node["__typename"] for node in timeline}
        assert "AutoSquashEnabledEvent" in typenames
        assert "AutoMergeEnabledEvent" not in typenames
        assert "AutoSquashEnabledEvent" in mod.ENABLE_EVENT_TYPES


def test_the_captures_are_real_graphql_responses_not_reconstructions() -> None:
    """A positive control on the capture itself.

    Both files must be a full `gh api graphql` envelope -- the `data`
    wrapper, the repository/pullRequest nesting and the labels connection
    the query asked for. A hand-written fixture of just the nodes array
    would pass every test above while proving nothing about what the API
    returns.
    """
    for capture in (_DISABLED_CAPTURE, _ARMED_CAPTURE):
        payload = json.loads(capture.read_text(encoding="utf-8"))
        assert set(payload) == {"data"}
        pull_request = payload["data"]["repository"]["pullRequest"]
        assert set(pull_request) == {"labels", "timelineItems"}
        assert isinstance(pull_request["timelineItems"]["nodes"], list)
        # The query deliberately does not select totalCount: it ignores the
        # itemTypes filter and returns the unfiltered timeline length.
        assert "totalCount" not in pull_request["timelineItems"]
