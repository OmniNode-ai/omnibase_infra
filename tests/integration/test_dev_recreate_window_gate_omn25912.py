# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read back the dev recreate window gate built from the checkout's rendered policy (OMN-25912).

The gate is built by the node's host composition root from the real
``docker/runtime-policy.env`` and the real handler; only the bus publisher is the
node's own in-process one, so nothing shared is touched.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from omnibase_infra.nodes.node_dev_roll_events_effect.runtime_dev_roll_events import (
    recreate_gate_at,
)

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[2]


def test_gate_from_rendered_policy_holds_a_second_recreate_in_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
    last = datetime.now(UTC)
    gate = recreate_gate_at(_ROOT, last)
    assert type(gate).__name__ == "DevRecreateWindowGate"
    assert gate.allow("gha/omnibase_infra/pr-1", last) is False


def test_unreadable_policy_is_a_refusing_gate(tmp_path: Path) -> None:
    gate = recreate_gate_at(tmp_path, None)
    assert gate.allow("agent/idle-converge", None) is False
