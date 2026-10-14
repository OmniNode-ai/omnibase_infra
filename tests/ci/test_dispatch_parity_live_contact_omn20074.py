# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Replay recorded runtime dispatch snapshots for both declared dependency corpora."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from tests.fixtures.dispatch_parity import harness


@pytest.mark.live_contact("tests/ci/fixtures/dispatch_parity_readback_omn20074.json")
def test_live_dispatch_snapshot_matches_recorded_dependency_closure(
    recorded_response: dict[str, Any],
) -> None:
    closure = (
        "legacy"
        if "onex-change-control" in harness.REQUIRED_ONEX_NODES_PACKAGES
        else "retired"
    )
    recording = recorded_response[closure]
    live = harness.strip_volatile_header_fields(harness.build_snapshot())
    committed = harness.strip_volatile_header_fields(harness.load_selection_oracle())
    assert live == committed
    assert live["header"]["corpus"] == recording["corpus"]
    assert (
        hashlib.sha256(json.dumps(live, sort_keys=True).encode()).hexdigest()
        == recording["snapshot_sha256"]
    )
