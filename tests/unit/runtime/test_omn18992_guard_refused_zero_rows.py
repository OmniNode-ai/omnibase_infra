# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18992 — a zero-row return has two causes and must not log as one.

A projection handler returning ``rows_upserted: 0`` was logged at ERROR
whatever the reason. Two very different things produce it:

* a writer that silently wrote nothing -- the defect the line exists for;
* an ordering guard refusing a redelivery -- correct, deliberate behaviour.

The consumer-flow writer's upsert carries
``OR ingest_sequence <= EXCLUDED.ingest_sequence`` on its conflict arm and a
``RETURNING`` clause, so an out-of-order or redelivered message writes nothing
and says so honestly. That guard exists because a read-compare-write would
race under concurrent consumers and let an older redelivery win.

Measured on the .201 dev lane at revision 430ff3434 over 33 minutes: 26 of
these ERROR lines, of which 4 came from a writer that was serving 500 rows and
updating every few seconds while it emitted them. An ERROR that fires on
correct behaviour trains people to skip the class, at which point the real
defect it exists to surface stops being visible. Silence class S5.

The consumer lands FIRST and tolerates the field's absence, so this can merge
in ``omnibase_infra`` before any writer in ``omnimarket`` sends it. That
ordering is the lesson OMN-18918 learned the expensive way: an additive field
is only additive if the consumer is already there.
"""

from __future__ import annotations

import ast
import logging
import pathlib
import re
from typing import Any

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


def _log_zero_row_outcome(
    caplog: pytest.LogCaptureFixture, result: dict[str, Any]
) -> list[logging.LogRecord]:
    """Replay the wiring's branch over one handler result.

    Mirrors the decision at the emit site rather than importing it: the branch
    lives inside a closure the runtime builds per contract, and standing up
    that whole wiring to assert two log records would test the harness. The
    shape asserted here is pinned against the real source by
    ``test_the_emit_site_still_branches_on_the_refusal_count`` below, so a
    change to one without the other is a red test rather than a stale mirror.

    The BEHAVIOURAL proof is not here. ``tests/integration/runtime/
    test_omn18992_guard_refused_zero_rows_integration.py`` drives the real
    callback from ``_make_projection_dispatch_callback`` and reads the records
    the runtime itself emitted; these cases cover the extractor edges the
    integration test would need a case each to reach.
    """
    logger = logging.getLogger("omn18992-replay")
    rows_upserted = _extract_rows_upserted(result)
    with caplog.at_level(logging.INFO, logger="omn18992-replay"):
        if rows_upserted >= 1:
            return []
        rows_refused = _extract_rows_refused(result)
        if rows_refused > 0:
            logger.info(
                "Projection handler wrote zero rows, refused by the ordering "
                "guard (expected, no terminal owed): rows_refused=%s",
                rows_refused,
            )
        else:
            logger.error(
                "Projection handler wrote zero rows (no terminal emitted): "
                "rows_upserted=%s",
                rows_upserted,
            )
    return list(caplog.records)


class TestTheBranchSeparatesTheTwoCauses:
    """AC1, both directions."""

    def test_a_guard_refusal_logs_info_and_no_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        records = _log_zero_row_outcome(
            caplog, {"rows_upserted": 0, "flow_rows": [], ROWS_REFUSED_KEY: 1}
        )

        levels = {record.levelno for record in records}
        assert logging.INFO in levels
        assert logging.ERROR not in levels
        assert "ordering guard" in records[0].getMessage()

    def test_a_genuine_zero_write_still_logs_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The half that must NOT be softened. This is the real defect."""
        records = _log_zero_row_outcome(caplog, {"rows_upserted": 0, "flow_rows": []})

        assert [record.levelno for record in records] == [logging.ERROR]
        assert "no terminal emitted" in records[0].getMessage()

    def test_a_successful_write_logs_neither(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        assert _log_zero_row_outcome(caplog, {"rows_upserted": 4}) == []


class TestTheCapturedWindowIsClassifiedCorrectly:
    """AC4. Real captured bytes off the lane, not a hand-written sample.

    26 lines from `omninode-runtime` at revision 430ff3434 between 10:15Z and
    10:48Z on 2026-09-21. The split is the finding: 4 from a writer that was
    demonstrably healthy at the time and 22 from the known OMN-18769 defect.
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

    def test_four_lines_are_the_guard_and_twenty_two_are_the_known_defect(
        self,
    ) -> None:
        handlers = self._handlers()

        assert handlers.count("ConsumerFlowProjectionWriter") == 4
        assert handlers.count("LabLaneHealthProjectionWriter") == 22

    def test_every_line_in_the_window_was_logged_at_error(self) -> None:
        """What the fix changes, stated as the before-state.

        All 26 read ERROR today. After the producer half lands, the 4 become
        INFO and the 22 stay ERROR until OMN-18769 -- and that asymmetry is
        the positive control for the lab proof. If both go quiet, the change
        suppressed the class instead of classifying it.
        """
        lines = _FIXTURE.read_text(encoding="utf-8").splitlines()

        assert all("[ERROR]" in line for line in lines)
        assert not any("ordering guard" in line for line in lines)


def test_the_emit_site_still_branches_on_the_refusal_count() -> None:
    """Keeps the replay above honest against the real source.

    The branch lives in a closure the runtime builds per contract, so the
    tests replay its shape rather than invoking it. That is only sound while
    the source still HAS the shape being replayed.

    Asserted structurally rather than as text, and the reason is a live
    near-miss: the first version of this test matched the string
    ``rows_refused = _extract_rows_refused(result)``, and disabling the branch
    with ``if False and rows_refused > 0:`` left every assertion in this file
    passing. A guard a one-word edit walks through is not a guard. The tree
    below rejects a constant folded into the test as well as the branch going
    missing.
    """
    source_path = (
        pathlib.Path(__file__).resolve().parents[3]
        / "src"
        / "omnibase_infra"
        / "runtime"
        / "auto_wiring"
        / "handler_wiring.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    branches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Name) and child.id == "rows_refused"
            for child in ast.walk(node.test)
        )
    ]

    assert len(branches) == 1, (
        f"expected exactly one branch on rows_refused, found {len(branches)} -- "
        "the emit site has been restructured and the replay in this file no "
        "longer mirrors it"
    )
    branch = branches[0]

    # A constant anywhere in the test means the branch cannot be reached on
    # merit -- `if False and ...` is the exact edit that defeated the first
    # version of this assertion.
    assert not any(
        isinstance(child, ast.Constant) and isinstance(child.value, bool)
        for child in ast.walk(branch.test)
    ), (
        "the refusal branch's test carries a boolean constant, so it is pinned open or shut"
    )

    def _log_levels(body: list[ast.stmt]) -> set[str]:
        return {
            node.func.attr
            for statement in body
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "logger"
        }

    assert _log_levels(branch.body) == {"info"}, (
        "the refusal arm must log at INFO and nothing else; an ERROR here is "
        "the defect this ticket exists to remove"
    )
    assert _log_levels(branch.orelse) == {"error"}, (
        "the non-refusal arm must still log at ERROR; softening it would "
        "suppress the real zero-write defect rather than classify it"
    )
