# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21427 AC1 falsifier over this repository's placement code.

The slot placement handler takes ``slot`` leases and keeps nothing of its own:
no ``.lock`` flock and no ``<host>.<lease>.<pid>`` marker file.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from omnibase_infra.handlers.lease import handler_lease_slot_placement

pytestmark = pytest.mark.unit

FORBIDDEN = (
    re.compile(r"\bfcntl\b"),
    re.compile(r"\bflock\b"),
    re.compile(r"\.lock\b"),
    re.compile(r"\bgetpid\b"),
    re.compile(r"\bkill\("),
    re.compile(r"\bopen\("),
    re.compile(r"\bmkdir\b"),
)


def test_slot_placement_writes_no_lock_or_marker() -> None:
    source = Path(handler_lease_slot_placement.__file__).read_text()
    hits = [p.pattern for p in FORBIDDEN if p.search(source)]
    assert hits == []
    assert "HandlerLeaseStore" in source
