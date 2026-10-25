# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev lane's runtime-effects lacks its context-source tenant (OMN-21099).

On the compose dev lane, runtime-effects has no
ONEX_CONTEXT_SOURCE_TENANT_ID. The omnimarket
node_context_compaction_orchestrator binds compaction_runtime.tenant_id from
"${env.ONEX_CONTEXT_SOURCE_TENANT_ID:}", and node_context_source_ledger_effect
and node_context_source_work_state_effect read tenant_ids from the same
variable in that container. Unset, every compaction command falls back with
failure_class=pack_builder_unavailable. At 2026-10-25T00:55Z, build_ms was
0.028 and a real /compact fell back to native.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_LANE = REPO_ROOT / "docker" / "docker-compose.dev-lane.yml"


def _runtime_effects_env() -> dict[str, object]:
    text = DEV_LANE.read_text().replace("!!merge ", "").replace("!override", "")
    service = yaml.safe_load(text)["services"]["runtime-effects"]
    env = service["environment"]
    assert isinstance(env, dict)
    return env


@pytest.mark.unit
def test_runtime_effects_carries_the_context_source_tenant() -> None:
    assert (
        _runtime_effects_env()["ONEX_CONTEXT_SOURCE_TENANT_ID"]
        == "${DEV_LANE_CONTEXT_SOURCE_TENANT_ID:-lab}"
    )


@pytest.mark.unit
def test_the_lane_identity_tenant_is_not_reused() -> None:
    value = _runtime_effects_env()["ONEX_CONTEXT_SOURCE_TENANT_ID"]
    assert isinstance(value, str)
    assert "ONEX_TENANT_ID" not in value
