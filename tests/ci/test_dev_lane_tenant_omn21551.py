# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev lane's runtimes carry no lane tenant (OMN-21551).

omnimarket's mode-2 environment observer (node_local_environment_observe_effect)
reports for the install identity, and on a compose lane, which never runs
``onex local init``, for the runtime's configured ONEX_TENANT_ID instead.
Neither omninode-runtime nor runtime-effects on the dev lane had one, so the
observer refused every run with identity_absent (113 refusals in 30 minutes on
2026-10-29) and local-environment.v1 served no rows, which blocks OMN-19988 AC2.

The lane tenant is the house tenant. The dev lane's tenant_registry_mirror
already maps slug ``omninode`` to it, so no seed step is added: the catalog's
local-tenant-registry-seed derives its uuid from a slug and would record a
second, phantom row. The lanes that layer the dev-lane overlay keep the
variable blank, as before this change.
"""

from __future__ import annotations

import pytest

from tests.ci._compose_render_omn21099 import (
    BORROWER_STACKS,
    DEV_LANE_OVERLAY,
    DEV_LANE_STACK,
    load_compose,
    render_stack,
)

pytestmark = [pytest.mark.unit]

TENANT = "ONEX_TENANT_ID"
OPERATOR_OVERRIDE = "DEV_LANE_TENANT_ID"
HOUSE_TENANT_UUID = "820272f9-4aaf-5add-a2df-0af942852ab2"
EXPECTED_ENTRY = f"${{{OPERATOR_OVERRIDE}:-{HOUSE_TENANT_UUID}}}"
RUNTIMES = ("omninode-runtime", "runtime-effects")
OPERATOR_TENANT_UUID = "00000000-0000-4000-8000-000000000001"


@pytest.mark.parametrize("service", RUNTIMES)
def test_the_dev_lane_runtime_names_the_lane_tenant(service: str) -> None:
    env = load_compose(DEV_LANE_OVERLAY)["services"][service]["environment"]
    assert env.get(TENANT) == EXPECTED_ENTRY


@pytest.mark.parametrize("service", RUNTIMES)
def test_the_dev_lane_renders_the_house_tenant_by_default(service: str) -> None:
    rendered = render_stack(DEV_LANE_STACK)[service]["environment"]
    assert rendered.get(TENANT) == HOUSE_TENANT_UUID


def test_the_operator_env_overrides_the_lane_tenant() -> None:
    services = render_stack(DEV_LANE_STACK, {OPERATOR_OVERRIDE: OPERATOR_TENANT_UUID})
    assert {
        name: services[name]["environment"].get(TENANT) for name in RUNTIMES
    } == dict.fromkeys(RUNTIMES, OPERATOR_TENANT_UUID)


@pytest.mark.parametrize("overrides", [{}, {OPERATOR_OVERRIDE: OPERATOR_TENANT_UUID}])
def test_the_context_source_tenant_stays_separate(overrides: dict[str, str]) -> None:
    env = render_stack(DEV_LANE_STACK, overrides)["runtime-effects"]["environment"]
    assert env["ONEX_CONTEXT_SOURCE_TENANT_ID"] == "lab"


@pytest.mark.parametrize("overlay", sorted(BORROWER_STACKS))
def test_lanes_layering_the_dev_lane_keep_no_lane_tenant(overlay: str) -> None:
    services = render_stack(BORROWER_STACKS[overlay])
    assert {
        name: services[name]["environment"].get(TENANT) for name in RUNTIMES
    } == dict.fromkeys(RUNTIMES, "")
