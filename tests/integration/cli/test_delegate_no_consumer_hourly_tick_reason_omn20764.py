# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The no-consumer refusal reads as no_consumer to the hourly tick (OMN-20764 AC2).

The hourly tick's held classifier lives in another repository. It classifies a
held delegate window from the failure text the delegate left, and since
OMN-20764 it has a ``no_consumer`` class, checked before ``timeout``, that
matches ``NO_CONSUMER_FAILURE`` in its sweep script. That class only fires if
the text this repository's ``onex delegate`` writes for a missing consumer
carries one of the phrases the pattern looks for.

This test pins that seam from this side: it drives the real command with no
consumer group bound and requires the receipt's ``failure_reason`` and the
operator's output to match the classifier's pattern, copied verbatim below.
The control drives an unanswerable broker, which is not a missing consumer, and
requires that it does not match.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.integration.cli.test_delegate_no_consumer_typed_terminal_omn20764 import (
    _broker_does_not_answer,
    _delegate,
    _delegate_environment,
    _no_group_bound,
    _only_receipt,
    clock,
)

pytestmark = pytest.mark.integration

# Imported for their autouse effect: the fake clock and the stand-in contract.
__all__ = ["_delegate_environment", "clock"]

# The hourly tick's NO_CONSUMER_FAILURE (an internal repository, sweep script
# linear-comments-hourly.js, added under OMN-20764), copied verbatim.
_HOURLY_TICK_NO_CONSUMER = re.compile(
    r"\bno (?:live )?consumer groups? (?:is|was|were|are) (?:bound|live)\b"
    r"|\bno service was listening\b"
    r"|\bDelegateNoConsumerRefusedError\b"
    r"|\babsent_consumer\b",
    re.IGNORECASE,
)


def test_hourly_tick_delegate_no_consumer_reason_matches_the_held_classifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC2, this side: the missing-consumer terminal carries the classifier's phrase."""
    _no_group_bound(monkeypatch)
    state_root = tmp_path / "state"
    result = _delegate(state_root, tmp_path)

    assert result.exit_code != 0
    receipt = _only_receipt(state_root)
    assert _HOURLY_TICK_NO_CONSUMER.search(receipt["failure_reason"]), receipt[
        "failure_reason"
    ]
    assert _HOURLY_TICK_NO_CONSUMER.search(result.output), result.output


def test_hourly_tick_delegate_no_consumer_reason_control_unanswerable_broker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: a broker that did not answer is not reported as a missing consumer."""
    _broker_does_not_answer(monkeypatch)
    state_root = tmp_path / "state"
    result = _delegate(state_root, tmp_path)

    assert result.exit_code != 0
    receipt = _only_receipt(state_root)
    assert not _HOURLY_TICK_NO_CONSUMER.search(receipt["failure_reason"]), receipt[
        "failure_reason"
    ]
