# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21272 -- durable_backfill re-measures a merge whose receipt is binding-only.

On 2026-10-21 the dev-lane receipt for 54ca4ac5f was a ``deployed_revision``
verdict about a deploy that never started. ``durable_backfill`` answered only
for merges with NO receipt, so once the lane reached 54ca4ac5f nothing could
measure it again. A receipt whose only failures are binding-class
(``is_binding_only_failure``) is now answered for; a PASS, a health-check FAIL
and an unreadable receipt still answer for nothing.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ci.check_dev_lane_staleness import (
    EnumReceiptPresence,
    ModelDurableBackfill,
    durable_backfill,
    read_receipt_binding_only,
)
from scripts.ci.lab_pass_receipt import (
    EnumLabLane,
    ModelLabPassCheck,
    ModelLabPassReceipt,
    ModelLaneRead,
    build_receipt,
    is_binding_only_failure,
)

pytestmark = pytest.mark.unit

_FLOOR = "a" * 40
_TARGET = "b" * 40  # the 54ca4ac5f shape: runtime-affecting, receipt PRESENT
_LANE = "c" * 40  # the lane's converged revision
_HISTORY_OLDEST_FIRST = (_FLOOR, _TARGET, _LANE)


def _below(history: Sequence[str]) -> Callable[[str, int], tuple[str, ...]]:
    newest_first = list(reversed(history))

    def read(revision: str, limit: int) -> tuple[str, ...]:
        index = newest_first.index(revision)
        return tuple(newest_first[index + 1 : index + 1 + limit])

    return read


def _all_present(sha: str) -> EnumReceiptPresence:
    return EnumReceiptPresence.PRESENT


def _receipt(*checks: ModelLabPassCheck) -> ModelLabPassReceipt:
    return build_receipt(
        sha=_TARGET,
        lane=EnumLabLane.COMPOSE_DEV,
        started_at=datetime(2026, 10, 21, 10, 56, tzinfo=UTC),
        finished_at=datetime(2026, 10, 21, 11, 22, 25, tzinfo=UTC),
        checks=list(checks),
        agent_command_id=None,
    )


_OK = ModelLabPassCheck(name="ready_main", ok=True, evidence="read fine")
_BINDING_ONLY = _receipt(
    ModelLabPassCheck(
        name="deployed_revision",
        ok=False,
        evidence="INDETERMINATE: job ended not_started=lane_lock_contended",
        indeterminate=True,
    ),
    _OK,
)
_PASS = _receipt(_OK)
_HEALTH_FAIL = _receipt(
    ModelLabPassCheck(name="ready_main", ok=False, evidence="503 not ready")
)


def _plan(binding_only: Callable[[str], bool]) -> ModelDurableBackfill:
    return durable_backfill(
        converged_revision=_LANE,
        already_answered=(),
        first_parent_below=_below(_HISTORY_OLDEST_FIRST),
        receipt_presence=_all_present,
        runtime_affecting=lambda sha: sha == _TARGET,
        receipt_binding_only=binding_only,
    )


def _reader(receipt: ModelLabPassReceipt | None) -> Callable[[str], bool]:
    """The receipt reader's contract over a fake read; ``None`` is unreadable.

    The real reader (:func:`read_receipt_binding_only`) never raises: an
    unreadable receipt is ``False``, pinned below through ``read_lane``.
    """

    def read(sha: str) -> bool:
        if sha != _TARGET or receipt is None:
            return False
        return is_binding_only_failure(receipt)

    return read


def test_backfill_remeasures_binding_failure() -> None:
    plan = _plan(_reader(_BINDING_ONLY))
    assert plan.backfill == (_TARGET,)
    assert plan.remeasured == (_TARGET,)
    assert plan.floor
    assert "re-measuring binding-only receipt(s) " + _TARGET[:12] in (
        plan.evidence_clause()
    )


def test_backfill_remeasures_binding_failure_not_a_pass() -> None:
    plan = _plan(_reader(_PASS))
    assert plan.backfill == ()
    assert plan.remeasured == ()


def test_backfill_remeasures_binding_failure_not_a_health_fail() -> None:
    plan = _plan(_reader(_HEALTH_FAIL))
    assert plan.backfill == ()
    assert plan.remeasured == ()


def test_backfill_remeasures_binding_failure_not_an_unreadable_receipt() -> None:
    plan = _plan(_reader(None))
    assert plan.backfill == ()


def test_backfill_remeasures_binding_failure_only_when_runtime_affecting() -> None:
    plan = durable_backfill(
        converged_revision=_LANE,
        already_answered=(),
        first_parent_below=_below(_HISTORY_OLDEST_FIRST),
        receipt_presence=_all_present,
        runtime_affecting=lambda sha: False,
        receipt_binding_only=lambda sha: True,
    )
    assert plan.backfill == ()


def test_backfill_remeasures_binding_failure_skips_already_answered() -> None:
    plan = durable_backfill(
        converged_revision=_LANE,
        already_answered=(_TARGET,),
        first_parent_below=_below(_HISTORY_OLDEST_FIRST),
        receipt_presence=_all_present,
        runtime_affecting=lambda sha: sha == _TARGET,
        receipt_binding_only=_reader(_BINDING_ONLY),
    )
    assert plan.backfill == ()


def test_backfill_remeasures_binding_failure_beside_an_orphan() -> None:
    # A receipt-less runtime merge above the binding-only one is still
    # answered for, and both come back oldest first.
    orphan = "d" * 40
    history = (_FLOOR, _TARGET, orphan, _LANE)
    plan = durable_backfill(
        converged_revision=_LANE,
        already_answered=(),
        first_parent_below=_below(history),
        receipt_presence=lambda sha: (
            EnumReceiptPresence.ABSENT if sha == orphan else EnumReceiptPresence.PRESENT
        ),
        runtime_affecting=lambda sha: sha in {_TARGET, orphan},
        receipt_binding_only=_reader(_BINDING_ONLY),
    )
    assert plan.backfill == (_TARGET, orphan)
    assert plan.remeasured == (_TARGET,)
    assert plan.floor == _TARGET


def test_backfill_remeasures_binding_failure_absent_without_reader() -> None:
    # Positive control: with no receipt reader the walk is exactly the
    # receipt-less one, so a PRESENT receipt is never answered for.
    plan = durable_backfill(
        converged_revision=_LANE,
        already_answered=(),
        first_parent_below=_below(_HISTORY_OLDEST_FIRST),
        receipt_presence=_all_present,
        runtime_affecting=lambda sha: sha == _TARGET,
    )
    assert plan.backfill == ()


@pytest.mark.parametrize(
    ("read", "expected"),
    [
        (
            ModelLaneRead(
                EnumLabLane.COMPOSE_DEV, _TARGET, "receipt", receipt=_BINDING_ONLY
            ),
            True,
        ),
        (
            ModelLaneRead(EnumLabLane.COMPOSE_DEV, _TARGET, "receipt", receipt=_PASS),
            False,
        ),
        (
            ModelLaneRead(
                EnumLabLane.COMPOSE_DEV, _TARGET, "receipt", receipt=_HEALTH_FAIL
            ),
            False,
        ),
        (
            ModelLaneRead(
                EnumLabLane.COMPOSE_DEV, _TARGET, "unreadable", problem="zip"
            ),
            False,
        ),
        (
            ModelLaneRead(EnumLabLane.COMPOSE_DEV, _TARGET, "absent", problem="none"),
            False,
        ),
    ],
)
def test_backfill_remeasures_binding_failure_through_the_artifact_reader(
    read: ModelLaneRead, expected: bool
) -> None:
    with patch("scripts.ci.lab_pass_receipt.read_lane", return_value=read) as fake:
        assert (
            read_receipt_binding_only("o/r", EnumLabLane.COMPOSE_DEV.value, _TARGET)
            is expected
        )
    fake.assert_called_once_with("o/r", EnumLabLane.COMPOSE_DEV, _TARGET)
