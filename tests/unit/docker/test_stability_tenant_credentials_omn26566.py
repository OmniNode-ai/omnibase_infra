# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26566: the stability runtime resolves a tenant's registered credential ref.

On the stability lane the runtime found the canary tenant key and routed to the
BYOK provider but could not resolve the registered ref: its secret resolver
config had no ``tenant_inference_credentials`` namespace and
``INFISICAL_PROJECT_ID`` was empty. The dev lane declares both. These tests pin
the same declaration for the stability runtime, by reference and never by value.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from pathlib import Path

import pytest

from scripts.render_runtime_policy_env import load_contract, render_env
from tests.helpers.lane_overlay_fixtures import load_compose

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
(STABILITY,) = (ROOT / "docker").glob("docker-compose.stability-t*.yml")
NAMESPACE = "tenant_inference_credentials"
STORE_KEY = "STABILITY_TEST_RUNTIME_MAIN_STORE_SECRET_RESOLVER_CONFIG_JSON"
EFFECTS_STORE_KEY = "STABILITY_TEST_RUNTIME_EFFECTS_STORE_SECRET_RESOLVER_CONFIG_JSON"
DEV_MAIN_STORE_KEY = "DEV_RUNTIME_MAIN_STORE_SECRET_RESOLVER_CONFIG_JSON"
SERVICES = {
    "omninode-runtime": STORE_KEY,
    "runtime-effects": EFFECTS_STORE_KEY,
}
IDENTITY = {
    "INFISICAL_ADDR": "STABILITY_TEST_RUNTIME_INFISICAL_ADDR",
    "INFISICAL_PROJECT_ID": "STABILITY_TEST_RUNTIME_INFISICAL_PROJECT_ID",
    "INFISICAL_CLIENT_ID": "STABILITY_TEST_RUNTIME_INFISICAL_CLIENT_ID",
    "INFISICAL_CLIENT_SECRET": "STABILITY_TEST_RUNTIME_INFISICAL_CLIENT_SECRET",
}


def _namespaces(rendered: Mapping[str, str], key: str) -> dict[str, dict]:
    config = json.loads(rendered[key])
    return {n["namespace"]: n for n in config.get("namespaces", [])}


def declaration_gaps(
    environment: Mapping[str, str], rendered: Mapping[str, str], store_key: str
) -> list[str]:
    """Every way a stability runtime's environment fails to resolve a tenant ref."""
    gaps: list[str] = []
    expected = f"${{{store_key}:?runtime policy contract must set {store_key}}}"
    if environment.get("ONEX_SECRET_RESOLVER_CONFIG_JSON") != expected:
        gaps.append("resolver config is not the store-backed rendering")
    if NAMESPACE not in _namespaces(rendered, store_key):
        gaps.append(f"no {NAMESPACE} namespace")
    for name, variable in IDENTITY.items():
        value = environment.get(name, "")
        if not re.fullmatch(rf"\$\{{{variable}:-\}}", value):
            gaps.append(f"{name} is not a reference to {variable}")
    if environment.get("INFISICAL_ENVIRONMENT_SLUG") != "dev":
        gaps.append("INFISICAL_ENVIRONMENT_SLUG is not pinned to dev")
    return gaps


@pytest.mark.parametrize(("service", "store_key"), sorted(SERVICES.items()))
def test_stability_runtime_declares_namespace_and_project_id(
    service: str, store_key: str
) -> None:
    environment = load_compose(STABILITY)["services"][service]["environment"]
    rendered = render_env(load_contract())
    assert declaration_gaps(environment, rendered, store_key) == []


def test_namespace_rule_equals_the_dev_lane_rule() -> None:
    rendered = render_env(load_contract())
    dev = _namespaces(rendered, DEV_MAIN_STORE_KEY)[NAMESPACE]
    assert dev, "positive control: the dev lane declares the rule"
    assert _namespaces(rendered, STORE_KEY)[NAMESPACE] == dev


def test_no_identity_value_is_written_in_the_compose_file() -> None:
    services = load_compose(STABILITY)["services"]
    for service in SERVICES:
        environment = services[service]["environment"]
        for name in IDENTITY:
            assert str(environment[name]).startswith("${"), (service, name)


def test_a_compose_without_the_declaration_fails_the_check() -> None:
    """FAILING CONTROL: the check names each gap on the pre-fix shape."""
    environment = copy.deepcopy(
        load_compose(STABILITY)["services"]["omninode-runtime"]["environment"]
    )
    rendered = dict(render_env(load_contract()))
    for name in IDENTITY:
        environment.pop(name, None)
    environment["ONEX_SECRET_RESOLVER_CONFIG_JSON"] = (
        "${STABILITY_TEST_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_JSON:?x}"
    )
    rendered[STORE_KEY] = json.dumps({"mappings": [], "namespaces": []})
    gaps = declaration_gaps(environment, rendered, STORE_KEY)
    assert "resolver config is not the store-backed rendering" in gaps
    assert f"no {NAMESPACE} namespace" in gaps
    assert any(g.startswith("INFISICAL_PROJECT_ID") for g in gaps)


def test_dev_render_is_unchanged_by_the_stability_declaration() -> None:
    """Dev keys render byte for byte as they do without the stability rule."""
    contract = load_contract()
    others = {
        name: profile.model_copy(
            update={
                "processes": {
                    process_name: process.model_copy(
                        update={"secret_resolver_namespaces": []}
                    )
                    for process_name, process in profile.processes.items()
                }
            }
        )
        for name, profile in contract.profiles.items()
        if name != "dev"
    }
    baseline = contract.model_copy(update={"profiles": {**contract.profiles, **others}})
    with_rule = render_env(contract)
    without_rule = render_env(baseline)
    dev_with = {k: v for k, v in with_rule.items() if k.startswith("DEV_")}
    dev_without = {k: v for k, v in without_rule.items() if k.startswith("DEV_")}
    assert dev_with
    assert json.dumps(dev_with, sort_keys=True) == json.dumps(
        dev_without, sort_keys=True
    )
    assert STORE_KEY in with_rule
    assert STORE_KEY not in without_rule
    assert DEV_MAIN_STORE_KEY in dev_without
