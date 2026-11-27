# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A lane's dev build waits for the recreate window built from the rendered policy (OMN-25912).

The gate is the real ``DevRecreateWindowGate`` built from the checkout's
``docker/runtime-policy.env``, wrapped by ``lane_build_gate``; only the agent shell
around it is a namespace. A lane build a moment after a recreate is deferred with a
named reason, and a rollback to a pinned dev tag still goes at once.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from omnibase_infra.nodes.node_dev_roll_events_effect.runtime_dev_roll_events import (
    is_lane_build,
    lane_build_gate,
    recreate_gate_at,
)

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[2]
_SHA = "a" * 40


def _cmd(git_ref: str) -> SimpleNamespace:
    return SimpleNamespace(
        runtime_lane=SimpleNamespace(value="dev"),
        rollback=SimpleNamespace(actor="lane/x", reason="build my branch"),
        image_ref=None,
        image_digest=None,
        git_ref=git_ref,
        requested_by="lane/x",
    )


def test_lane_build_is_deferred_by_the_rendered_policy_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
    gate = recreate_gate_at(_ROOT, datetime.now(UTC))
    window = lane_build_gate(SimpleNamespace(_recreate_gate=gate, _load_gate=None))

    lane_build = _cmd(_SHA)
    assert is_lane_build(lane_build)
    decision = window.check(lane_build)
    assert "deferred to the recreate window" in decision.describe()


def test_pinned_dev_tag_rollback_is_not_a_lane_build() -> None:
    assert not is_lane_build(_cmd("dev-20261125010101-abcdef1234567"))
