# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20533: the .201 dev lane's runtime-effects can READ a tenant's BYOK key.

onex-api writes a tenant's provider key to Infisical env ``dev``, folder
``/tenant-inference-credentials``, under a ref minted per request, and the
tenant overlay routes that tenant to the ref. The provider call runs in
runtime-effects, whose resolver config held only env mappings, so every tenant
ref failed closed: 4 of 4 lab-tenant runs on 2026-10-04 between 09:33 and
09:50Z answered provider_error with "could not be resolved from the secret
store".

The fix has four parts, and each is asserted here without Docker:

1. The contract declares the tenant-credential namespace rule on the dev
   profile's EFFECTS process only, reading the same folder onex-api writes.
2. The renderer emits it as ``DEV_RUNTIME_EFFECTS_STORE_SECRET_RESOLVER_CONFIG_JSON``
   (house mappings plus the rule) and leaves every profile-level config
   house-only.
3. ``docker-compose.dev-lane.yml`` binds that config and effects' own
   identity names (``RUNTIME_EFFECTS_INFISICAL_*``) on runtime-effects, and
   nowhere else.
4. Every other lane that layers ``docker-compose.dev-lane.yml`` rebinds
   runtime-effects to the house config, because an Infisical rule makes a
   process demand a store identity at boot and those lanes hold none.
"""

from __future__ import annotations

import json
import re
import shlex
import uuid
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.runtime.models.model_runtime_policy_contract import (
    ModelRuntimePolicyContract,
)
from omnibase_infra.runtime.models.model_runtime_profile_policy import (
    ModelRuntimeProfilePolicy,
)
from omnibase_infra.runtime.models.model_secret_resolver_config import (
    ModelSecretResolverConfig,
)
from omnibase_infra.runtime.secret_resolver import (
    SecretResolver,
    config_declares_infisical_source,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOCKER = _REPO_ROOT / "docker"
_CONTRACT = _REPO_ROOT / "contracts" / "services" / "runtime_policy.contract.yaml"
_POLICY_ENV = _DOCKER / "runtime-policy.env"
_DEV_LANE = _DOCKER / "docker-compose.dev-lane.yml"

_STORE_VAR = "DEV_RUNTIME_EFFECTS_STORE_SECRET_RESOLVER_CONFIG_JSON"
_HOUSE_VAR = "DEV_RUNTIME_EFFECTS_SECRET_RESOLVER_CONFIG_JSON"
_IDENTITY_PREFIX = "RUNTIME_EFFECTS_INFISICAL_"
_NAMESPACE = "tenant_inference_credentials"

# Every lane that layers docker-compose.dev-lane.yml under its own overlay.
# Named, not discovered: test_every_other_dev_lane_layer_is_listed below fails
# when a new overlay claims the dev-lane layering and is missing here.
_BORROWER_OVERLAYS = (
    "docker-compose.dev-105.yml",
    "docker-compose.dev-200.yml",
    "docker-compose.dev-202.yml",
    "docker-compose.prepr.yml",
)


_COMPOSE_MERGE_TAG = re.compile(r"!(?:override|reset)\b")


def _compose(name: str) -> dict[str, Any]:
    # Compose's merge tags (`!override`, `!reset`) only steer how files merge;
    # dropping them leaves the plain mapping this module asserts on.
    text = _COMPOSE_MERGE_TAG.sub("", (_DOCKER / name).read_text(encoding="utf-8"))
    loaded = yaml.safe_load(text)
    assert isinstance(loaded, dict)
    return loaded


def _effects_env(name: str) -> dict[str, Any]:
    service = _compose(name)["services"]["runtime-effects"]
    environment = service.get("environment", {})
    assert isinstance(environment, dict), name
    return environment


def _policy_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in _POLICY_ENV.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        key, raw = line.split("=", 1)
        parsed = shlex.split(raw) if raw else [""]
        env[key] = parsed[0] if parsed else ""
    return env


def _contract() -> ModelRuntimePolicyContract:
    return ModelRuntimePolicyContract.model_validate(
        yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    )


def test_the_rule_is_declared_on_dev_effects_and_nowhere_else() -> None:
    contract = _contract()
    declared = {
        (profile_name, process_name): [
            rule.namespace for rule in process.secret_resolver_namespaces
        ]
        for profile_name, profile in contract.profiles.items()
        for process_name, process in profile.processes.items()
        if process.secret_resolver_namespaces
    }
    assert declared == {("dev", "effects"): [_NAMESPACE]}
    # No profile-level rule anywhere: that would reach main, worker and
    # tenant-projection too, each of which would then demand an identity.
    assert all(
        not profile.secret_resolver_namespaces for profile in contract.profiles.values()
    )


def test_the_rule_reads_the_folder_onex_api_writes() -> None:
    rule = (
        _contract().profiles["dev"].processes["effects"].secret_resolver_namespaces[0]
    )
    onex_api_env = _compose("docker-compose.dev-lane.yml")["services"]["onex-api"][
        "environment"
    ]
    write_folder = onex_api_env["INFISICAL_TENANT_CREDENTIAL_SECRET_PATH"]
    assert rule.source_type == "infisical"
    assert rule.source_path_template == f"{write_folder}/{{ref}}"


def test_the_rendered_store_config_is_the_house_config_plus_the_rule() -> None:
    env = _policy_env()
    store = ModelSecretResolverConfig.model_validate(json.loads(env[_STORE_VAR]))
    house = ModelSecretResolverConfig.model_validate(json.loads(env[_HOUSE_VAR]))

    assert store.mappings == house.mappings
    assert [rule.namespace for rule in store.namespaces] == [_NAMESPACE]
    assert not house.namespaces
    assert config_declares_infisical_source(store)
    assert not config_declares_infisical_source(house)
    # Exactly one store variant exists: no other lane or process gained one.
    assert [key for key in env if "_STORE_SECRET_RESOLVER_CONFIG_JSON" in key] == [
        _STORE_VAR
    ]


def test_a_ref_minted_after_render_resolves_to_the_tenant_folder() -> None:
    store = ModelSecretResolverConfig.model_validate(
        json.loads(_policy_env()[_STORE_VAR])
    )
    resolver = SecretResolver(config=store)
    minted = f"cred_{uuid.uuid4()}_openrouter_{uuid.uuid4().hex}"
    spec = resolver.resolve_namespace_source(minted)
    assert spec is not None
    assert spec.source_type == "infisical"
    assert spec.source_path == f"/tenant-inference-credentials/{minted}"
    # Positive control for the refusal: a platform-shaped name is not claimed,
    # so a tenant ref can never be steered at a house key and vice versa.
    assert resolver.resolve_namespace_source("OPENROUTER_API_KEY") is None


def test_dev_lane_effects_binds_the_store_config_and_its_own_identity() -> None:
    env = _effects_env("docker-compose.dev-lane.yml")
    assert env["ONEX_SECRET_RESOLVER_CONFIG_JSON"].startswith(f"${{{_STORE_VAR}:?")
    for name in ("ADDR", "PROJECT_ID", "CLIENT_ID", "CLIENT_SECRET"):
        assert env[f"INFISICAL_{name}"] == f"${{{_IDENTITY_PREFIX}{name}:-}}"
    assert env["INFISICAL_ENVIRONMENT_SLUG"] == "dev"


def test_the_identity_and_store_config_reach_no_other_service() -> None:
    offenders: list[str] = []
    for path in sorted(_DOCKER.glob("docker-compose*.yml")):
        services = _compose(path.name).get("services") or {}
        for service_name, service in services.items():
            if path == _DEV_LANE and service_name == "runtime-effects":
                continue
            text = yaml.safe_dump(
                service.get("environment", {}) if isinstance(service, dict) else {}
            )
            if _IDENTITY_PREFIX in text or _STORE_VAR in text:
                offenders.append(f"{path.name}:{service_name}")
    assert offenders == []


@pytest.mark.parametrize("overlay", _BORROWER_OVERLAYS)
def test_every_other_dev_lane_layer_rebinds_the_house_config(overlay: str) -> None:
    env = _effects_env(overlay)
    assert env["ONEX_SECRET_RESOLVER_CONFIG_JSON"].startswith(f"${{{_HOUSE_VAR}:?")
    for name in ("ADDR", "PROJECT_ID", "CLIENT_ID", "CLIENT_SECRET"):
        assert env[f"INFISICAL_{name}"] == f"${{ONEX_RUNTIME_INFISICAL_{name}:-}}"
    assert env["INFISICAL_ENVIRONMENT_SLUG"] == ""


def test_the_rebind_check_fires_on_the_dev_lane_itself() -> None:
    """Positive control: the borrower assertion must fail on the .201 binding."""
    with pytest.raises(AssertionError):
        env = _effects_env("docker-compose.dev-lane.yml")
        assert env["ONEX_SECRET_RESOLVER_CONFIG_JSON"].startswith(f"${{{_HOUSE_VAR}:?")


def test_every_other_dev_lane_layer_is_listed() -> None:
    layering = sorted(
        path.name
        for path in _DOCKER.glob("docker-compose*.yml")
        if path != _DEV_LANE
        and "runtime-effects" in (_compose(path.name).get("services") or {})
        and (
            "-f docker/docker-compose.dev-lane.yml" in path.read_text(encoding="utf-8")
            or "over docker-compose.infra.yml and docker-compose.dev-lane.yml"
            in path.read_text(encoding="utf-8")
        )
    )
    assert layering == sorted(_BORROWER_OVERLAYS)


def test_a_process_rule_needs_the_profile_config_path() -> None:
    profile = _contract().profiles["dev"].model_dump(mode="json", by_alias=True)
    profile["secret_resolver_config_path"] = ""
    profile["secret_resolver_mappings"] = []
    with pytest.raises(ValueError, match="config path"):
        ModelRuntimeProfilePolicy.model_validate(profile)


def test_a_process_rule_may_not_reuse_a_profile_namespace() -> None:
    profile = _contract().profiles["dev"].model_dump(mode="json", by_alias=True)
    profile["secret_resolver_namespaces"] = profile["services"]["effects"][
        "secret_resolver_namespaces"
    ]
    with pytest.raises(ValueError, match="unique across the process and its profile"):
        ModelRuntimeProfilePolicy.model_validate(profile)
