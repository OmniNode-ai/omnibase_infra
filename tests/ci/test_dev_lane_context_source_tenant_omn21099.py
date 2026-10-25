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

Those readers live in omnimarket, which this repository does not install, so
these checks cover the shipped compose configuration: the committed entry, and
the value compose renders for runtime-effects from it. The running container is
read on the lab after the deploy agent converges.
"""

from __future__ import annotations

import re

import pytest

from tests.ci._compose_render_omn21099 import (
    DEV_LANE_OVERLAY,
    DEV_LANE_STACK,
    docker_compose_available,
    load_compose,
    render_stack,
)

pytestmark = [pytest.mark.unit]

TENANT = "ONEX_CONTEXT_SOURCE_TENANT_ID"
OPERATOR_OVERRIDE = "DEV_LANE_CONTEXT_SOURCE_TENANT_ID"
EXPECTED_ENTRY = "${DEV_LANE_CONTEXT_SOURCE_TENANT_ID:-lab}"
_INTERPOLATED_NAME_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)")

needs_compose = pytest.mark.skipif(
    not docker_compose_available(),
    reason="docker compose CLI absent; the committed-entry checks still hold",
)


def _runtime_effects_env() -> dict[str, object]:
    service = load_compose(DEV_LANE_OVERLAY)["services"]["runtime-effects"]
    env = service["environment"]
    assert isinstance(env, dict)
    return env


def _rendered_tenant(overrides: dict[str, str]) -> object:
    services = render_stack(DEV_LANE_STACK, overrides)
    return services["runtime-effects"]["environment"][TENANT]


def test_runtime_effects_carries_the_context_source_tenant() -> None:
    assert _runtime_effects_env()[TENANT] == EXPECTED_ENTRY


@needs_compose
def test_runtime_effects_renders_the_lab_tenant_by_default() -> None:
    assert _rendered_tenant({}) == "lab"


@needs_compose
def test_the_operator_env_overrides_the_tenant() -> None:
    rendered = _rendered_tenant({OPERATOR_OVERRIDE: "operator-tenant"})
    assert rendered == "operator-tenant"


def test_the_lane_identity_tenant_is_not_reused() -> None:
    value = _runtime_effects_env()[TENANT]
    assert isinstance(value, str)
    assert set(_INTERPOLATED_NAME_RE.findall(value)) == {OPERATOR_OVERRIDE}


@needs_compose
def test_the_rendered_tenant_does_not_follow_the_lane_identity() -> None:
    assert _rendered_tenant({"ONEX_TENANT_ID": "lane-identity-sentinel"}) == "lab"
