# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18048 — the AC-coverage guard cannot see criteria written in house style.

Two independent blind spots that compound into a wrong count rather than a
missing one.

**A. The heading.** ``_AC_HEADING_TEXTS`` is a closed set of nine spellings and
``_is_ac_heading`` normalises away ``#``, ``*_``, a leading enumeration and a
trailing ``:`` — but not a trailing parenthetical. ``## Acceptance criteria
(falsifiable)`` is therefore unrecognised. When no heading anywhere is
recognised, the documented OMN-16106 fallback fires and the WHOLE BODY becomes
the section.

**B. The items.** Criteria written ``**AC1** — ...`` match neither
``_LIST_ITEM_RE`` (needs a bullet or a number) nor ``_AC_ITEM_RE`` (needs the
line to START with ``AC``, and this one starts with ``**``).

Together the guard scans the entire description and counts whatever unrelated
bullets it finds, while the real criteria stay invisible. Measured on OMN-18035:
four declared criteria, and the sweep reported **two** — two bullets from an
unrelated section. Not an over-count and not an under-count: a count of
different things. The docstring's "over-counting holds a flip, under-counting
releases one" safety argument does not cover it, and both failure directions are
reachable.

Corpus measurement (OMN-18048 AC3, 110 tickets read at full text): **14.5%**
carry an AC-looking heading the parser does not recognise, across **12 distinct
spellings**. ``## Acceptance criteria (falsifiable)`` alone is 5 of 16 misses.
Every one of the twelve is a recognised base spelling plus a qualifier — so the
repair is to normalise the qualifier away, NOT to add strings to a set that will
never close. The corpus also found a PREFIX form (``## Falsifiable acceptance
criteria``, 5 occurrences) that a trailing-parenthetical-only fix would miss.

The strings below are verbatim from that corpus, not invented.
"""

from __future__ import annotations

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    _acceptance_criteria_items,
    _is_ac_heading,
)

pytestmark = pytest.mark.unit


# Verbatim from the AC3 corpus sweep: every distinct unrecognised heading found
# across 110 full-text ticket descriptions, plus the case variant that started
# the investigation.
_UNRECOGNISED_IN_CORPUS = [
    "## Acceptance criteria (falsifiable)",
    "## Acceptance Criteria (falsifiable)",
    "## Acceptance criteria (DoD — each keeps concrete evidence)",
    "## Acceptance criteria (named here, non-implementer per operator rule)",
    "## Acceptance criteria (falsifiable, write results into this ticket)",
    "## Acceptance (in-ticket, per doctrine)",
    "## Definition of Done (mechanism-first — a shipped command + a real e2e)",
    "## Definition of Done (dod_evidence)",
    "## DoD (falsifiable)",
    "## DoD (per repo)",
    "**Acceptance (falsifiable):**",
]

# The prefix form. A fix that only strips a TRAILING parenthetical leaves all
# five corpus occurrences of this shape still invisible.
_PREFIX_QUALIFIED = [
    "## Falsifiable acceptance criteria",
    "## Falsifiable Acceptance Criteria",
]


class TestQualifiedHeadingsAreRecognised:
    """AC1 — a qualifier must not hide an otherwise-recognised heading."""

    @pytest.mark.parametrize("heading", _UNRECOGNISED_IN_CORPUS)
    def test_trailing_qualifier_does_not_hide_the_heading(self, heading: str) -> None:
        assert _is_ac_heading(heading), (
            f"{heading!r} is an acceptance-criteria heading and must be recognised; "
            "leaving it unrecognised fires the whole-body fallback and counts "
            "unrelated bullets as criteria"
        )

    @pytest.mark.parametrize("heading", _PREFIX_QUALIFIED)
    def test_prefix_qualifier_does_not_hide_the_heading(self, heading: str) -> None:
        assert _is_ac_heading(heading), (
            f"{heading!r} carries the qualifier BEFORE the phrase; a fix that only "
            "strips a trailing parenthetical misses this shape entirely"
        )


class TestBoldPrefixedItemsAreSeen:
    """AC2 — the same criteria must parse identically however they are written."""

    _BOLD = (
        "## Acceptance criteria (falsifiable)\n"
        "\n"
        "**AC1** - a complete page returns a null cursor.\n"
        "\n"
        "**AC2** - a truncated page returns a non-null cursor.\n"
        "\n"
        "**AC3** - the RED test runs against the real consumers.\n"
        "\n"
        "**AC4** - both mutations fail different tests.\n"
    )
    _BULLETED = (
        "## Acceptance criteria (falsifiable)\n"
        "\n"
        "- **AC1** - a complete page returns a null cursor.\n"
        "- **AC2** - a truncated page returns a non-null cursor.\n"
        "- **AC3** - the RED test runs against the real consumers.\n"
        "- **AC4** - both mutations fail different tests.\n"
    )

    def test_bold_prefixed_criteria_are_counted(self) -> None:
        assert len(_acceptance_criteria_items(self._BOLD)) == 4

    def test_bold_and_bulleted_forms_agree(self) -> None:
        """The written style must not change how many criteria exist."""
        assert len(_acceptance_criteria_items(self._BOLD)) == len(
            _acceptance_criteria_items(self._BULLETED)
        )

    def test_the_real_omn_18035_shape_yields_four_not_two(self) -> None:
        """The exact failure that surfaced this ticket.

        Four criteria under a qualified heading, with two unrelated bullets in a
        later section. The sweep reported 2 — those bullets — because the
        heading was unrecognised, so the fallback scanned the whole body while
        the bold criteria stayed invisible.
        """
        description = (
            "Some context about the seam.\n\n" + self._BOLD + "\n## Fence\n\n"
            "- dev lane only\n"
            "- no stability-test mutation\n"
        )
        items = _acceptance_criteria_items(description)
        assert len(items) == 4, (
            f"expected the 4 declared criteria, got {len(items)}: {items!r} — "
            "counting the Fence bullets instead of the criteria is the defect"
        )
        assert not any("dev lane only" in i for i in items)


class TestWideningDoesNotBecomeOverMatching:
    """AC5 — recognising more headings must not start inventing criteria."""

    def test_prose_beginning_with_acceptance_is_not_a_heading(self) -> None:
        """A real false positive caught during the AC3 corpus sweep.

        Sentences like this occur in ticket bodies and are not headings. If they
        were treated as one, the section would start in the middle of prose.
        """
        assert not _is_ac_heading(
            "Acceptance evidence: the result packet showing a success path."
        )
        assert not _is_ac_heading("Acceptance of this plan is the operator's call.")

    def test_unrelated_headings_stay_unrecognised(self) -> None:
        for heading in ("## Fence", "## Provenance", "## Why this exists", "## Scope"):
            assert not _is_ac_heading(heading), heading

    def test_a_real_heading_still_bounds_the_section(self) -> None:
        """With a recognised heading present, later sections must not be read."""
        description = (
            "## Acceptance criteria (falsifiable)\n"
            "- AC1: the thing works\n"
            "\n"
            "## Provenance\n"
            "- found during a sweep\n"
            "- filed the same day\n"
        )
        assert _acceptance_criteria_items(description) == ["AC1: the thing works"]


class TestItemTextIsNotPollutedByEmphasis:
    r"""OMN-18048 hostile review [MAJOR] — the captured TEXT, not just the count.

    The first revision captured `(AC[-_ ]?\d+\b.*)` after consuming a leading
    emphasis run, so `**AC1** - text` yielded `AC1** - text`: the CLOSING marker
    survived, embedded mid-string. The count was right and the string was wrong,
    which is exactly the divergence a count-only assertion cannot see.

    It matters because the same criterion written two ways must compare equal.
    Downstream the item text is used for evidence strings and de-duplication, so
    a bold spelling and a bulleted spelling of one criterion would read as two.
    """

    def test_bold_and_bulleted_forms_of_one_criterion_are_equal(self) -> None:
        bold = _acceptance_criteria_items(
            "## Acceptance criteria\n**AC1** - a complete page\n"
        )
        bulleted = _acceptance_criteria_items(
            "## Acceptance criteria\n- AC1 - a complete page\n"
        )
        assert bold == bulleted == ["AC1 - a complete page"]

    def test_no_emphasis_marker_survives_in_captured_text(self) -> None:
        items = _acceptance_criteria_items(
            "## Acceptance criteria\n"
            "**AC1** - bold wrapped\n"
            "*AC2* - italic wrapped\n"
            "__AC3__ - underscore bold\n"
            "**AC4 - emphasis spans the whole item**\n"
        )
        assert items == [
            "AC1 - bold wrapped",
            "AC2 - italic wrapped",
            "AC3 - underscore bold",
            "AC4 - emphasis spans the whole item",
        ]
        for text in items:
            assert "*" not in text and "_" not in text, text

    def test_a_trailing_underscore_in_prose_is_preserved(self) -> None:
        """The strip is conditional, and must be.

        Stripping trailing emphasis unconditionally would corrupt item text that
        legitimately ends in `_` and never used emphasis at all — turning a
        capture bug into a truncation bug.
        """
        assert _acceptance_criteria_items(
            "## Acceptance criteria\nAC1 names the column user_id_\n"
        ) == ["AC1 names the column user_id_"]

    def test_separator_after_the_token_is_preserved(self) -> None:
        """`AC3: plain` must not become `AC3 : plain`."""
        assert _acceptance_criteria_items("## Acceptance criteria\nAC3: plain\n") == [
            "AC3: plain"
        ]
