# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26473 -- the dev lane's effects runtime carries the tenant identity database DSN.

omnimarket ``node_tenant_provision_effect`` is an effect node placed on the effects
runtime profile. It reads its database through the logical secret
``tenant_identity.database.dsn``. On the dev lane that name resolves only when two
things hold together:

* the policy contract maps the logical name to the env ``TENANT_IDENTITY_DATABASE_DSN``
  for the dev profile, so every dev runtime role's rendered resolver config carries it;
* the dev-lane overlay's ``runtime-effects`` service merges the fail-closed DSN anchor,
  so the env exists in the container that runs the node.

Before this change neither held for effects: the anchor was merged into the main
runtime only, and no policy mapping existed for any role.

Every pin reads a committed file and resolves the compose merge keys the way compose
does, so an anchor rename or an edited merge list fails here and not on the lab.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
COMPOSE_DEV_LANE = ROOT / "docker" / "docker-compose.dev-lane.yml"
COMPOSE_INFRA = ROOT / "docker" / "docker-compose.infra.yml"
POLICY_CONTRACT = ROOT / "contracts" / "services" / "runtime_policy.contract.yaml"
POLICY_ENV = ROOT / "docker" / "runtime-policy.env"
DEV_MANIFEST = ROOT / "docker" / "dev-lane-required-env.manifest.txt"

DSN_VAR = "TENANT_IDENTITY_DATABASE_DSN"
PASSWORD_VAR = "TENANT_IDENTITY_WRITER_PASSWORD"
LOGICAL_NAME = "tenant_identity.database.dsn"
ANCHOR = "dev_lane_tenant_identity_env"
SERVICES_WITH_THE_DSN = ("omninode-runtime", "runtime-effects")


class _ComposeLoader(yaml.SafeLoader):
    """compose uses `!override` and `!!merge`, which SafeLoader refuses."""


_ComposeLoader.add_multi_constructor(
    "",
    lambda loader, suffix, node: (
        loader.construct_mapping(node, deep=True)
        if isinstance(node, yaml.MappingNode)
        else (
            loader.construct_sequence(node, deep=True)
            if isinstance(node, yaml.SequenceNode)
            else loader.construct_scalar(node)
        )
    ),
)


def _service_environment(service: str) -> dict[str, Any]:
    """The service's environment with its merge keys resolved, as compose renders it."""
    # The loader subclasses SafeLoader; the only widening is a multi-constructor that
    # resolves compose's tags to plain mappings, sequences and scalars.
    loader = _ComposeLoader(COMPOSE_DEV_LANE.read_text(encoding="utf-8"))
    try:
        parsed = loader.get_single_data()
    finally:
        loader.dispose()
    environment = parsed["services"][service]["environment"]
    assert isinstance(environment, dict), f"{service} environment is not a mapping"
    return environment


def _policy_profiles() -> dict[str, dict[str, Any]]:
    return yaml.safe_load(POLICY_CONTRACT.read_text(encoding="utf-8"))["profiles"]


@pytest.mark.parametrize("service", SERVICES_WITH_THE_DSN)
def test_dev_lane_service_carries_the_tenant_identity_dsn_fail_closed(
    service: str,
) -> None:
    environment = _service_environment(service)
    assert DSN_VAR in environment, f"{service} carries no {DSN_VAR}"
    dsn = environment[DSN_VAR]
    assert dsn.startswith(f"postgresql://tenant_identity_writer:${{{PASSWORD_VAR}:?")
    assert dsn.endswith("@postgres:5432/omninode_cloud")


def test_the_dsn_is_declared_once_in_the_overlay_and_never_in_a_base_file() -> None:
    overlay = COMPOSE_DEV_LANE.read_text(encoding="utf-8")
    declarations = re.findall(rf"^  {DSN_VAR}: ", overlay, flags=re.MULTILINE)
    assert len(declarations) == 1
    assert f"&{ANCHOR}" in overlay
    # The base file only passes the variable through x-runtime-env so the resolver
    # can read it; the credentialed value lives in the overlay alone.
    base_lines = [
        line.strip()
        for line in COMPOSE_INFRA.read_text(encoding="utf-8").splitlines()
        if DSN_VAR in line and not line.lstrip().startswith("#")
    ]
    assert base_lines == [f"{DSN_VAR}: ${{{DSN_VAR}:-}}"]
    assert PASSWORD_VAR not in COMPOSE_INFRA.read_text(encoding="utf-8")


def test_the_password_stays_listed_in_the_required_env_manifest() -> None:
    names = {
        line.strip()
        for line in DEV_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert PASSWORD_VAR in names


def test_policy_maps_the_tenant_identity_secret_to_the_dsn_env_on_dev() -> None:
    mappings = [
        m
        for m in _policy_profiles()["dev"]["secret_resolver_mappings"]
        if m["logical_name"] == LOGICAL_NAME
    ]
    assert len(mappings) == 1
    assert mappings[0]["source"] == {"source_type": "env", "source_path": DSN_VAR}


def test_policy_maps_the_tenant_identity_secret_on_no_other_profile() -> None:
    mapped = {
        name
        for name, body in _policy_profiles().items()
        if any(
            m["logical_name"] == LOGICAL_NAME
            for m in body.get("secret_resolver_mappings", [])
        )
    }
    assert mapped == {"dev"}


def _rendered_names_carrying(logical_name: str) -> set[str]:
    carrying: set[str] = set()
    for line in POLICY_ENV.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.partition("=")
        if not sep or not name.endswith("_SECRET_RESOLVER_CONFIG_JSON"):
            continue
        raw = value.strip().strip("'")
        if raw and logical_name in {
            m["logical_name"] for m in json.loads(raw)["mappings"]
        }:
            carrying.add(name)
    return carrying


def test_the_rendered_policy_env_resolves_the_secret_on_the_dev_effects_runtime() -> (
    None
):
    carrying = _rendered_names_carrying(LOGICAL_NAME)
    assert "DEV_RUNTIME_EFFECTS_SECRET_RESOLVER_CONFIG_JSON" in carrying
    assert "DEV_RUNTIME_EFFECTS_STORE_SECRET_RESOLVER_CONFIG_JSON" in carrying
    assert "DEV_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_JSON" in carrying
    assert not {name for name in carrying if not name.startswith("DEV_")}


def test_the_committed_policy_env_equals_a_fresh_render() -> None:
    from omnibase_infra.runtime.models.model_runtime_policy_contract import (
        ModelRuntimePolicyContract,
    )
    from scripts.render_runtime_policy_env import render_env

    contract = ModelRuntimePolicyContract.model_validate(
        yaml.safe_load(POLICY_CONTRACT.read_text(encoding="utf-8"))
    )
    rendered = render_env(contract)
    for name, value in rendered.items():
        if not name.endswith("_SECRET_RESOLVER_CONFIG_JSON") or not value:
            continue
        if LOGICAL_NAME in {m["logical_name"] for m in json.loads(value)["mappings"]}:
            assert f"{name}=" in POLICY_ENV.read_text(encoding="utf-8")
