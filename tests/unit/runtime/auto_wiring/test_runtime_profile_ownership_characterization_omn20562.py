# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Characterization: which runtime owns the ten formerly allowlisted contracts (OMN-20562).

These ten node contracts were the whole of the runtime_profiles allowlist. Before
OMN-20562 they declared no ``runtime_profiles`` and so fell under the
undeclared-defaults-to-``main`` rule in ``profile_ownership``. Emptying that
allowlist means each contract declares its profile explicitly, and the only
declaration that keeps runtime behaviour identical is the one the default
already produced.

This test pins the observable fact the burn-down must not change: for every
registered runtime profile, on every registered runtime lane and on a runtime
that declares no lane, the contract is owned by ``main`` and by nothing else.
It was committed and run green against the unchanged contracts first, and it
must pass unchanged after them; that is the idempotency proof for the edit.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_core.constants.constants_runtime_lanes import REGISTERED_RUNTIME_LANES
from omnibase_core.constants.constants_runtime_profiles import (
    REGISTERED_RUNTIME_PROFILES,
)
from omnibase_infra.runtime.auto_wiring.profile_ownership import (
    runtime_profile_owns_contract,
)
from omnibase_infra.runtime.health.runtime_lane_identity import ENV_RUNTIME_LANE

pytestmark = pytest.mark.unit

NODES_ROOT = Path(__file__).resolve().parents[4] / "src" / "omnibase_infra" / "nodes"

FORMERLY_ALLOWLISTED_NODES: tuple[str, ...] = (
    "node_artifact_change_detector_effect",
    "node_baselines_batch_compute",
    "node_build_loop_write_effect",
    "node_chain_retrieval_effect",
    "node_ledger_write_effect",
    "node_llm_completion_effect",
    "node_llm_embedding_effect",
    "node_savings_estimation_compute",
    "node_topic_migration_executor_effect",
    "node_vector_store_effect",
)

_LANE_ENVIRONMENTS: tuple[dict[str, str], ...] = (
    {},
    *({ENV_RUNTIME_LANE: lane} for lane in sorted(REGISTERED_RUNTIME_LANES)),
)


def _raw_contract(node: str) -> dict[str, object]:
    loaded = yaml.safe_load((NODES_ROOT / node / "contract.yaml").read_text())
    assert isinstance(loaded, dict)
    return loaded


@pytest.mark.parametrize("node", FORMERLY_ALLOWLISTED_NODES)
def test_contract_is_owned_by_main_and_only_main(node: str) -> None:
    raw = _raw_contract(node)
    for environ in _LANE_ENVIRONMENTS:
        owners = sorted(
            profile
            for profile in REGISTERED_RUNTIME_PROFILES
            if runtime_profile_owns_contract(raw, profile, environ=environ)
        )
        assert owners == ["main"], (
            f"{node} on lane {environ or '<undeclared>'} is owned by {owners}; "
            "the burn-down must leave it owned by main alone"
        )


@pytest.mark.parametrize("node", FORMERLY_ALLOWLISTED_NODES)
def test_contract_declares_no_runtime_lane_scope(node: str) -> None:
    raw = _raw_contract(node)
    assert "runtime_lanes" not in raw
    descriptor = raw.get("descriptor")
    assert not (isinstance(descriptor, dict) and "runtime_lanes" in descriptor)
