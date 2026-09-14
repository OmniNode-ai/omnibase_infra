# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18362. A criterion a later revision replaced is not a live criterion.

The sweep reads the CURRENT description text. House style, when a criterion is
re-ruled, is to write the replacement and keep the old text in the body under an
explicit ``(superseded ...)`` qualifier. Both carry the same ``AC<n>`` label, so
the reader returned both and the sweep counted a replaced declaration as one the
ticket still has to satisfy.

Every assertion below runs over a COMMITTED SCRUBBED SNAPSHOT of a real
``documentContentHistory`` payload -- OMN-18035, three revisions -- rather than a
hand-built document, because a reader proven against a hand-built body only
proves the body. Identities are synthetic; the description and its rich-text
revisions are otherwise the platform's own bytes.

The pair that matters is a matched one. ``TestLiveSetOnTheCurrentRevision``
proves the replaced criterion is dropped; ``TestFailsClosed`` proves the rule
cannot drop anything else, because dropping a criterion is the RELEASING
direction and every other bound in this module is written to hold instead.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    _ac_coverage_gap,
    _acceptance_criteria_items,
    _canonical_ac_label,
    _declares_supersession,
    _live_acceptance_criteria_items,
)

pytestmark = pytest.mark.unit

#: The scrubbed capture. `issue` carries the live description; `history` carries
#: every revision the platform holds, newest first.
_FIXTURE = (
    Path(__file__).parents[3]
    / "fixtures"
    / "autoclose"
    / "omn_18362_omn_18035_document_content_history.json"
)


def _capture() -> dict[str, Any]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _parse_timestamp(raw: str) -> datetime:
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _rich_text(node: Any, out: list[str]) -> None:
    """Every text leaf of a Linear rich-text document, in document order."""
    if isinstance(node, dict):
        if node.get("type") == "text" and isinstance(node.get("text"), str):
            out.append(node["text"])
        for value in node.values():
            _rich_text(value, out)
    elif isinstance(node, list):
        for value in node:
            _rich_text(value, out)


class TestTheCaptureIsReal:
    """The fixture is a platform payload, not a document someone wrote here."""

    def test_creation_revision_is_present_and_exact(self) -> None:
        capture = _capture()
        created_at = _parse_timestamp(capture["issue"]["createdAt"])
        snapshots = [
            _parse_timestamp(entry["contentDataSnapshotAt"])
            for entry in capture["history"]
        ]
        # The creation entry is selected the way the plan of record selects it:
        # `contentDataSnapshotAt == issue.createdAt`, to the millisecond.
        assert created_at in snapshots
        assert min(snapshots) == created_at
        # Three revisions, strictly ordered -- a positive control on the read
        # itself, since a capture that returned one entry would prove nothing
        # about what a later revision changed.
        assert len(snapshots) == 3
        assert sorted(snapshots) == snapshots[::-1]

    def test_no_identity_or_local_path_survives_the_scrub(self) -> None:
        blob = _FIXTURE.read_text(encoding="utf-8")
        assert "/Users/" not in blob
        assert "/Volumes/" not in blob


class TestCreationRevisionDeclaredOneAc3:
    """The duplicate came from a LATER revision, not from the author's pen.

    This is what makes the second AC3 a REPLACED declaration rather than an
    authoring slip, and it is why dropping it is the semantics of the closeout
    plan's section 6 rather than a convenience.
    """

    def test_creation_revision_carries_no_supersession_marker(self) -> None:
        capture = _capture()
        created_at = _parse_timestamp(capture["issue"]["createdAt"])
        creation = next(
            entry
            for entry in capture["history"]
            if _parse_timestamp(entry["contentDataSnapshotAt"]) == created_at
        )
        leaves: list[str] = []
        _rich_text(creation["contentData"], leaves)
        flat = " ".join(leaves)
        assert "supersed" not in flat.lower()

    def test_the_marker_appears_only_in_the_latest_revision(self) -> None:
        capture = _capture()
        marked = []
        for entry in capture["history"]:
            leaves: list[str] = []
            _rich_text(entry["contentData"], leaves)
            if "supersed" in " ".join(leaves).lower():
                marked.append(_parse_timestamp(entry["contentDataSnapshotAt"]))
        snapshots = [
            _parse_timestamp(entry["contentDataSnapshotAt"])
            for entry in capture["history"]
        ]
        assert marked == [max(snapshots)]


class TestLiveSetOnTheCurrentRevision:
    """RED before the change: the reader returned the replaced criterion too."""

    def test_the_raw_reader_still_sees_both_declarations(self) -> None:
        # Unchanged on purpose. `_acceptance_criteria_items` is the hash input
        # the change-control criterion reader pins shared digest vectors
        # against, so the exclusion is a filter over its output and never an
        # edit to it.
        items = _acceptance_criteria_items(_capture()["issue"]["description"])
        assert [_canonical_ac_label(item) for item in items] == [
            "AC1",
            "AC2",
            "AC3",
            "AC3",
            "AC4",
        ]

    def test_the_live_set_carries_each_label_once(self) -> None:
        items = _live_acceptance_criteria_items(_capture()["issue"]["description"])
        assert [_canonical_ac_label(item) for item in items] == [
            "AC1",
            "AC2",
            "AC3",
            "AC4",
        ]

    def test_the_surviving_ac3_is_the_replacement_not_the_replaced(self) -> None:
        items = _live_acceptance_criteria_items(_capture()["issue"]["description"])
        ac3 = [item for item in items if _canonical_ac_label(item) == "AC3"]
        assert len(ac3) == 1
        assert "re-ruled" in ac3[0]
        assert not _declares_supersession(ac3[0])


class TestFailsClosed:
    """The rule narrows and never widens. Each case here keeps the criterion."""

    def test_a_sole_declaration_carrying_the_marker_is_kept(self) -> None:
        description = (
            "## Acceptance criteria\n\n"
            "**AC1** — the first thing.\n"
            "**AC2 (superseded 2026-01-01 — replaced upstream)** — the second thing.\n"
        )
        # AC2 says it was replaced and nothing replaced it HERE. Dropping it
        # would release a ticket on a criterion no live item covers, so it is
        # counted and the ticket holds.
        assert [
            _canonical_ac_label(item)
            for item in _live_acceptance_criteria_items(description)
        ] == ["AC1", "AC2"]

    def test_a_duplicate_label_with_no_marker_keeps_both(self) -> None:
        description = (
            "## Acceptance criteria\n\n"
            "**AC1** — the first thing.\n"
            "**AC1** — a second thing written under the same label.\n"
        )
        assert len(_live_acceptance_criteria_items(description)) == 2

    def test_an_unrecognised_spelling_keeps_both(self) -> None:
        description = (
            "## Acceptance criteria\n\n"
            "**AC1 (re-ruled)** — the live one.\n"
            "**AC1** — the old one, no longer required. Superseded, see above.\n"
        )
        # The word appears, but not as a qualifier attached to the label. The
        # reader declines to guess and both items count.
        assert len(_live_acceptance_criteria_items(description)) == 2

    def test_the_marker_must_sit_on_the_label_not_in_the_body(self) -> None:
        item = "AC1 — falsified by a superseded receipt being counted."
        assert not _declares_supersession(item)

    def test_a_body_with_no_marker_anywhere_is_untouched(self) -> None:
        description = _capture()["issue"]["description"].replace(
            "superseded", "re-ruled"
        )
        raw = _acceptance_criteria_items(description)
        assert _live_acceptance_criteria_items(description) == raw


class TestCoverageGapBothDirections:
    """The count bound, watched failing in both directions (AC4)."""

    def test_no_gap_at_four_verified_probative_checks(self) -> None:
        reason, uncovered = _ac_coverage_gap(
            _capture()["issue"]["description"],
            verified_count=4,
            coverage_verified_count=4,
            coverage_non_probative_count=0,
        )
        assert reason == ""
        assert uncovered == ()

    def test_a_gap_is_still_reported_at_three(self) -> None:
        reason, uncovered = _ac_coverage_gap(
            _capture()["issue"]["description"],
            verified_count=3,
            coverage_verified_count=3,
            coverage_non_probative_count=0,
        )
        assert "4 item(s)" in reason
        assert len(uncovered) == 4
