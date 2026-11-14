# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21611: the h051 stability lane resolves the shared keys by path in the lab secret store.

GITHUB_TOKEN, LINEAR_API_KEY, GEMINI_API_KEY and GOOGLE_API_KEY are shared values. The
h051 stability lane reads them where they are held and never from a per-lane copy:

1. The contract declares two store-backed rules on the stability profile's MAIN process
   (the runtime that carries the effects role), naming folders and keys only.
2. The renderer emits them as ``STABILITY_TEST_RUNTIME_MAIN_STORE_SECRET_RESOLVER_CONFIG_JSON``
   and leaves the profile-level config house-only, so a lane that does not layer the
   overlay demands no store identity.
3. ``docker-compose.stability-h051-secrets.yml`` binds that config and the lane's own
   identity names on the one runtime service, and nowhere else.
4. No rendered resolver config names a literal value, and no rule points into the lane's
   own per-lane folder.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.runtime.models.model_runtime_policy_contract import (
    ModelRuntimePolicyContract,
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
_OVERLAY = "docker-compose.stability-h051-secrets.yml"

_STORE_VAR = "STABILITY_TEST_RUNTIME_MAIN_STORE_SECRET_RESOLVER_CONFIG_JSON"
_HOUSE_VAR = "STABILITY_TEST_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_JSON"
_IDENTITY_PREFIX = "STABILITY_H051_RUNTIME_INFISICAL_"

# Where each shared name is held in the lab secret store (read 2026-11-09, names only).
_SHARED_FOLDER = {
    "GITHUB_TOKEN": "/shared/env",
    "GEMINI_API_KEY": "/shared/env",
    "GOOGLE_API_KEY": "/shared/env",
    "LINEAR_API_KEY": "/shared/lab-lane-env",
}
_LANE_FOLDER = "/stability-h051"

_COMPOSE_MERGE_TAG = re.compile(r"!(?:override|reset)\b")


def _compose(name: str) -> dict[str, Any]:
    text = _COMPOSE_MERGE_TAG.sub("", (_DOCKER / name).read_text(encoding="utf-8"))
    loaded = yaml.safe_load(text)
    assert isinstance(loaded, dict)
    return loaded


def _base_compose() -> str:
    """The base compose file of the profile that declares the rules."""
    return f"docker-compose.{_declaring_processes()[0][0]}.yml"


def _policy_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in _POLICY_ENV.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        key, raw = line.split("=", 1)
        parsed = shlex.split(raw) if raw else [""]
        env[key] = parsed[0] if parsed else ""
    return env


def _store_config() -> ModelSecretResolverConfig:
    return ModelSecretResolverConfig.model_validate(
        json.loads(_policy_env()[_STORE_VAR])
    )


@pytest.mark.parametrize(("name", "folder"), sorted(_SHARED_FOLDER.items()))
def test_each_shared_name_resolves_to_its_shared_folder(name: str, folder: str) -> None:
    resolver = SecretResolver(config=_store_config())
    spec = resolver.resolve_namespace_source(name)
    assert spec is not None
    assert spec.source_type == "infisical"
    assert spec.source_path == f"{folder}/{name}"


def test_a_name_outside_the_four_is_not_claimed() -> None:
    """Positive control: the rule claims the four names and no other."""
    resolver = SecretResolver(config=_store_config())
    for other in (
        "OPENROUTER_API_KEY",
        "SLACK_BOT_TOKEN",
        "GITHUB_TOKEN_X",
        "RPK_PASS",
    ):
        assert resolver.resolve_namespace_source(other) is None


def test_no_rule_points_into_the_lane_folder_or_names_a_value() -> None:
    config = _store_config()
    for rule in config.namespaces:
        assert not rule.source_path_template.startswith(_LANE_FOLDER)
    # A literal value would sit in a mapping source path; every source path is a
    # name, an in-container path or a folder path.
    text = _policy_env()[_STORE_VAR]
    assert _LANE_FOLDER not in text
    for mapping in config.mappings:
        assert mapping.source.source_type in {"env", "file"}


def _declaring_processes() -> list[tuple[str, str]]:
    contract = ModelRuntimePolicyContract.model_validate(
        yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    )
    return [
        (profile_name, process_name)
        for profile_name, profile in contract.profiles.items()
        for process_name, process in profile.processes.items()
        if any(
            rule.namespace.startswith("shared_")
            for rule in process.secret_resolver_namespaces
        )
    ]


def test_only_the_stability_main_process_declares_the_rules() -> None:
    declared = _declaring_processes()
    assert [process for _, process in declared] == ["main"]
    assert len(declared) == 1
    # The profile key is the one the rendered variable prefix names.
    assert declared[0][0].upper().replace("-", "_") == _STORE_VAR.split("_RUNTIME_")[0]


def test_the_profile_level_config_stays_house_only() -> None:
    env = _policy_env()
    house = ModelSecretResolverConfig.model_validate(json.loads(env[_HOUSE_VAR]))
    store = _store_config()
    assert store.mappings == house.mappings
    assert not house.namespaces
    assert config_declares_infisical_source(store)
    assert not config_declares_infisical_source(house)
    for key, value in env.items():
        if (
            key.startswith("STABILITY_TEST_")
            and key.endswith("_SECRET_RESOLVER_CONFIG_JSON")
            and "_STORE_" not in key
        ):
            assert not config_declares_infisical_source(
                ModelSecretResolverConfig.model_validate(json.loads(value))
            ), key


def test_the_overlay_binds_the_store_config_and_the_lane_identity_on_the_runtime() -> (
    None
):
    services = _compose(_OVERLAY)["services"]
    assert sorted(services) == ["omninode-runtime"]
    env = services["omninode-runtime"]["environment"]
    assert env["ONEX_SECRET_RESOLVER_CONFIG_JSON"].startswith(f"${{{_STORE_VAR}:?")
    for name in ("ADDR", "PROJECT_ID", "CLIENT_ID", "CLIENT_SECRET"):
        assert env[f"INFISICAL_{name}"] == f"${{{_IDENTITY_PREFIX}{name}:-}}"
    assert env["INFISICAL_ENVIRONMENT_SLUG"] == "dev"


def test_the_store_config_and_identity_reach_no_other_compose_service() -> None:
    offenders: list[str] = []
    for path in sorted(_DOCKER.glob("docker-compose*.yml")):
        if path.name == _OVERLAY:
            continue
        for service_name, service in (
            _compose(path.name).get("services") or {}
        ).items():
            text = yaml.safe_dump(
                service.get("environment", {}) if isinstance(service, dict) else {}
            )
            if _IDENTITY_PREFIX in text or _STORE_VAR in text:
                offenders.append(f"{path.name}:{service_name}")
    assert offenders == []


def test_the_stability_base_compose_still_binds_the_house_config() -> None:
    env = _compose(_base_compose())["services"]["omninode-runtime"]["environment"]
    assert env["ONEX_SECRET_RESOLVER_CONFIG_JSON"].startswith(f"${{{_HOUSE_VAR}:?")
    assert not any(str(key).startswith("INFISICAL_") for key in env)


def test_the_binding_check_fires_when_the_overlay_binds_the_house_config() -> None:
    """Named break: an overlay that rebinds the house config fails the same assertion."""
    with pytest.raises(AssertionError):
        env = _compose(_base_compose())["services"]["omninode-runtime"]["environment"]
        assert env["ONEX_SECRET_RESOLVER_CONFIG_JSON"].startswith(f"${{{_STORE_VAR}:?")
