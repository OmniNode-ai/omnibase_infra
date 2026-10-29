# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev lane's two kernel runtimes carry the house lane tenant (OMN-21554).

On 2026-10-29 ~21:00Z neither omninode-runtime nor omninode-runtime-effects on
the deployed dev lane had ONEX_TENANT_ID. omnimarket's
node_local_environment_observe_effect resolves the tenant it reports for from
the install identity, else the runtime's configured ONEX_TENANT_ID; with
neither, every tick refused with identity_absent and OMN-19988 AC2 could not
run. The local_environment writer confirms a UUID tenant against
tenant_registry_mirror by its tenant_uuid column, and the dev lane's mirror already
holds omninode -> the house UUID from a real tenant event, so no seed is wired.

omnimarket is not installed in this repository, so the house UUID is checked by
its documented derivation (omnimarket projection/tenant_isolation.py
HOUSE_TENANT_UUID = uuid5(NAMESPACE_DNS, "house-tenant.omninode.ai")), which
omnimarket's own test re-derives. The running containers are read on the lab
after the deploy agent converges.
"""

from __future__ import annotations

import os
import re
import uuid

import pytest

from tests.ci._compose_render_omn21099 import (
    DEV_LANE_OVERLAY,
    DEV_LANE_STACK,
    docker_compose_available,
    load_compose,
    render_stack,
)

pytestmark = [pytest.mark.unit]

TENANT = "ONEX_TENANT_ID"
OPERATOR_OVERRIDE = "DEV_LANE_TENANT_ID"
RUNTIMES = ("omninode-runtime", "runtime-effects")
HOUSE_TENANT_UUID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "house-tenant.omninode.ai"))
EXPECTED_ENTRY = f"${{{OPERATOR_OVERRIDE}:-{HOUSE_TENANT_UUID}}}"
_DEFAULT_RE = re.compile(r"^\$\{DEV_LANE_TENANT_ID:-([^}]*)\}$")

CONTEXT_SOURCE_TENANT = "ONEX_CONTEXT_SOURCE_TENANT_ID"
CONTEXT_SOURCE_ENTRY = "${DEV_LANE_CONTEXT_SOURCE_TENANT_ID:-lab}"

# Skipped only off CI: on a CI runner a missing docker compose CLI fails the
# rendered checks instead of skipping them, so the rendered and override
# behaviour is never reported green unverified.
needs_compose = pytest.mark.skipif(
    not os.environ.get("CI") and not docker_compose_available(),
    reason="docker compose CLI absent off CI; the committed-entry checks still hold",
)


def _committed_entry(service: str) -> object:
    env = load_compose(DEV_LANE_OVERLAY)["services"][service]["environment"]
    assert isinstance(env, dict)
    return env.get(TENANT)


@pytest.mark.parametrize("service", RUNTIMES)
def test_each_runtime_carries_the_lane_tenant(service: str) -> None:
    assert _committed_entry(service) == EXPECTED_ENTRY


@pytest.mark.parametrize("service", RUNTIMES)
def test_the_default_is_the_house_tenant_uuid(service: str) -> None:
    entry = _committed_entry(service)
    assert isinstance(entry, str), f"{service} has no {TENANT}"
    match = _DEFAULT_RE.fullmatch(entry)
    assert match is not None, entry
    default = match.group(1)
    assert str(uuid.UUID(default)) == default
    assert default == HOUSE_TENANT_UUID == "820272f9-4aaf-5add-a2df-0af942852ab2"


@needs_compose
def test_the_rendered_dev_lane_carries_the_house_tenant_on_both_runtimes() -> None:
    services = render_stack(DEV_LANE_STACK)
    for service in RUNTIMES:
        assert services[service]["environment"][TENANT] == HOUSE_TENANT_UUID


@needs_compose
def test_the_operator_env_overrides_the_lane_tenant() -> None:
    services = render_stack(DEV_LANE_STACK, {OPERATOR_OVERRIDE: "operator-tenant"})
    for service in RUNTIMES:
        assert services[service]["environment"][TENANT] == "operator-tenant"


def test_the_context_source_tenant_stays_beside_the_lane_tenant() -> None:
    env = load_compose(DEV_LANE_OVERLAY)["services"]["runtime-effects"]["environment"]
    assert isinstance(env, dict)
    assert env.get(TENANT) == EXPECTED_ENTRY
    assert env.get(CONTEXT_SOURCE_TENANT) == CONTEXT_SOURCE_ENTRY


@needs_compose
def test_the_context_source_tenant_renders_lab_beside_the_lane_tenant() -> None:
    services = render_stack(DEV_LANE_STACK, {OPERATOR_OVERRIDE: "operator-tenant"})
    env = services["runtime-effects"]["environment"]
    assert env[TENANT] == "operator-tenant"
    assert env[CONTEXT_SOURCE_TENANT] == "lab"
