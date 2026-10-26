# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19043 AC-1: a dispatch that cannot reach the broker leaves a typed receipt.

The ticket's falsifier is a connection error that leaves no receipt on disk.
This drives the receipt writer with the receipt ``run_receipt_mode`` builds on
a broker connection error and reads back what a lane would read: the run
directory's ``receipt.json``. It must exist, name the transport as the class of
the failure, and be distinguishable from a run still in flight (no file) and
from a model-quality failure (no attempts, no quality verdict).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.unit.cli.test_delegate_transport_terminal_omn18925 import _receipt, _write

pytestmark = pytest.mark.unit


def test_an_unreachable_broker_leaves_a_typed_transport_receipt(
    tmp_path: Path,
) -> None:
    receipt = _receipt(is_transport=True)
    _write(receipt, tmp_path)

    receipt_path = tmp_path / "runs" / str(receipt.run_id) / "receipt.json"
    assert receipt_path.is_file(), "a connection error left no receipt on disk"
    written = json.loads(receipt_path.read_text())

    assert written["terminal_class"] == "transport"
    assert written["transport_refusal"]["reason"] == "broker_unreachable"
    assert written["attempts"] == []
    assert (
        "quality_gate_passed" not in written or written["quality_gate_passed"] is None
    )
