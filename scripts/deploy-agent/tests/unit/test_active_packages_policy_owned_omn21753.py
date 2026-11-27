# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime's active package set is the policy's, so the prebuilt manifest stamp holds (OMN-21753).

The image build stamps ``active_packages`` from ``docker/runtime-policy.env``. The deploy
agent used ``setdefault`` for policy keys, so a ``ONEX_ACTIVE_RUNTIME_PACKAGES`` in the operator
store or the agent's process environment reached the container instead, and boot read a
different set than the stamp ("omnibase_infra,omnimarket" stamped, four packages seen).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from deploy_agent import executor

_ACTIVE = "ONEX_ACTIVE_RUNTIME_PACKAGES"
_FOUR = "omnibase_infra,omniclaude,omniintelligence,omnimarket"
_POLICY = Path(__file__).resolve().parents[4] / "docker" / "runtime-policy.env"


@pytest.fixture
def isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(executor, "runtime_policy_env_file", lambda: _POLICY)
    monkeypatch.setattr(executor, "_lane_env_overlay", dict)
    monkeypatch.setattr(executor, "_dev_broker_policy_env", lambda env: {})
    monkeypatch.setattr(executor, "resolve_lane_overlay_dir", lambda env: None)


def test_operator_store_cannot_widen_the_active_package_set(
    monkeypatch: pytest.MonkeyPatch, isolated: None
) -> None:
    policy = executor._load_runtime_policy_env()
    monkeypatch.setattr(executor, "operator_store_overlay", lambda: {_ACTIVE: _FOUR})
    monkeypatch.setenv(_ACTIVE, _FOUR)
    env = executor._compose_env()
    assert env[_ACTIVE] == policy[_ACTIVE]
    assert env[_ACTIVE] != _FOUR


def test_policy_value_stands_when_nothing_overrides_it(
    monkeypatch: pytest.MonkeyPatch, isolated: None
) -> None:
    policy = executor._load_runtime_policy_env()
    monkeypatch.setattr(executor, "operator_store_overlay", dict)
    monkeypatch.delenv(_ACTIVE, raising=False)
    assert executor._compose_env()[_ACTIVE] == policy[_ACTIVE]
