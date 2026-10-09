# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18992: refusal-count compatibility and the historical log's limits.

The real callback and PostgreSQL suites establish the B23 outcome classes.
The captured 26-line window establishes writer identities only, not whether
an ordering predicate fired. Older same-node sequences are refused by the
consumer-flow SQL predicate; equal sequences are accepted.

"""

from __future__ import annotations

import pathlib
import re

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ROWS_REFUSED_KEY,
    _extract_rows_refused,
    _extract_rows_upserted,
)

pytestmark = pytest.mark.unit

_FIXTURE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "fixtures"
    / "omn18992"
    / "zero-rows-window-430ff3434.log.captured"
)


class TestTheRefusalCountIsReadOnlyWhenSent:
    """The consumer half, which must be safe before any producer exists."""

    def test_a_result_reporting_refusals_yields_the_count(self) -> None:
        assert _extract_rows_refused({"rows_upserted": 0, ROWS_REFUSED_KEY: 3}) == 3

    def test_the_pre_existing_shape_yields_zero(self) -> None:
        """AC2. A writer that never heard of this field is read as today."""
        assert _extract_rows_refused({"rows_upserted": 0, "flow_rows": []}) == 0
        assert _extract_rows_refused({"projected": False}) == 0
        assert _extract_rows_refused({}) == 0
        assert _extract_rows_refused(None) == 0

    @pytest.mark.parametrize("value", ["three", None, [], {}, object()])
    def test_an_unreadable_count_yields_zero_rather_than_raising(
        self, value: object
    ) -> None:
        """Unknown refuses, which here means the ERROR survives."""
        assert _extract_rows_refused({ROWS_REFUSED_KEY: value}) == 0

    def test_a_negative_count_yields_zero(self) -> None:
        """A negative refusal is not a refusal; it must not silence the error."""
        assert _extract_rows_refused({ROWS_REFUSED_KEY: -5}) == 0

    def test_reading_refusals_does_not_disturb_the_row_count(self) -> None:
        """The two extractors are independent; neither may shadow the other."""
        result = {"rows_upserted": 0, ROWS_REFUSED_KEY: 2}

        assert _extract_rows_upserted(result) == 0
        assert _extract_rows_refused(result) == 2


class TestTheCapturedWindowIsClassifiedCorrectly:
    """AC4. Real captured bytes off the lane, not a hand-written sample.

    The 26 historical lines identify four consumer-flow outcomes and 22
    lane-health outcomes. Their cause cannot be recovered from the log alone.
    The real-upsert regression fixture proves the corrected B23 classes.
    """

    @staticmethod
    def _handlers() -> list[str]:
        pattern = re.compile(r"handler=(\w+)")
        return [
            match.group(1)
            for line in _FIXTURE.read_text(encoding="utf-8").splitlines()
            if (match := pattern.search(line))
        ]

    def test_the_window_holds_every_line_it_claims_to(self) -> None:
        """Positive control: the fixture is present and parses."""
        handlers = self._handlers()

        assert len(handlers) == 26, (
            f"the captured window parsed to {len(handlers)} lines, not 26 -- "
            "the fixture or the parser has drifted, and a miscount here would "
            "silently weaken every assertion below"
        )

    def test_four_lines_are_consumer_flow_and_twenty_two_are_lane_health(
        self,
    ) -> None:
        handlers = self._handlers()

        assert handlers.count("ConsumerFlowProjectionWriter") == 4
        assert handlers.count("LabLaneHealthProjectionWriter") == 22

    def test_every_line_in_the_window_was_logged_at_error(self) -> None:
        """What the fix changes, stated as the before-state.

        All 26 were ERROR. These bytes identify the writer, not why it
        returned zero. B23 requires a real SQL refusal and a window-less
        heartbeat to establish those classes; neither is inferred here.
        """
        lines = _FIXTURE.read_text(encoding="utf-8").splitlines()

        assert all("[ERROR]" in line for line in lines)
        assert not any("ordering guard" in line for line in lines)
