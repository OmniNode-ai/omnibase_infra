# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The stamp built from the policy equals the boot-time stamp under the agent's compose env (OMN-21753).

This runs in the root suite, not the deploy-agent sub-project suite: the sub-project resolves the
published ``omnibase-infra`` wheel, which does not carry ``handler_prebuilt_manifest``, so the
real ``stamp_of`` is only importable from this repository's source.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from deploy_agent import executor

from omnibase_infra.handlers.handler_prebuilt_manifest import stamp_of

_ACTIVE = "ONEX_ACTIVE_RUNTIME_PACKAGES"
_FOUR = "omnibase_infra,omniclaude,omniintelligence,omnimarket"
_POLICY = Path(__file__).resolve().parents[3] / "docker" / "runtime-policy.env"


@pytest.fixture
def isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(executor, "runtime_policy_env_file", lambda: _POLICY)
    monkeypatch.setattr(executor, "_lane_env_overlay", dict)
    monkeypatch.setattr(executor, "_dev_broker_policy_env", lambda env: {})
    monkeypatch.setattr(executor, "resolve_lane_overlay_dir", lambda env: None)


def test_stamp_built_from_policy_reads_fresh_under_the_agents_compose_env(
    monkeypatch: pytest.MonkeyPatch, isolated: None
) -> None:
    """Known good: the real stamp and the real compare agree; known bad: a missing package is named."""
    policy = executor._load_runtime_policy_env()
    monkeypatch.setattr(executor, "operator_store_overlay", lambda: {_ACTIVE: _FOUR})
    monkeypatch.setenv(_ACTIVE, _FOUR)

    monkeypatch.setenv(_ACTIVE, policy[_ACTIVE])
    built = stamp_of((), (), "main")

    booted_env = executor._compose_env()
    monkeypatch.setenv(_ACTIVE, booted_env[_ACTIVE])
    assert built.differing_fields(stamp_of((), (), "main")) == ()

    monkeypatch.setenv(_ACTIVE, _FOUR)
    differing = built.differing_fields(stamp_of((), (), "main"))
    assert any(field.startswith("active_packages") for field in differing)
