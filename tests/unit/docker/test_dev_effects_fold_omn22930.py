# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22930: primary dev's effects role runs inside the one runtime.

The retired ``runtime-effects`` service resolved a set of secret names from
declared sources, read its tenant credentials with its own Infisical identity,
and answered health on its own port. This file pins each of those to the folded
main runtime and pins the removal of the separate service from the dev stack,
over the real policy contract, the real dev compose files and the real
installers.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pytest
import yaml

from omnibase_infra.runtime.models.model_secret_resolver_config import (
    ModelSecretResolverConfig,
)
from omnibase_infra.runtime.secret_resolver import SecretResolver
from scripts.render_runtime_policy_env import format_dotenv, load_contract, render_env
from tests.helpers.lane_overlay_fixtures import load_compose
from tests.unit.docker.test_dev_tenant_projection_fold_omn21288 import (
    dev_stack_services,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
DOCKER = ROOT / "docker"
OVERLAY = DOCKER / "docker-compose.dev-tenant-fold.yml"
DEV_LANE = DOCKER / "docker-compose.dev-lane.yml"
SERVICE = "runtime-effects"
RETIRED_STORE_KEY = "DEV_RUNTIME_EFFECTS_STORE_SECRET_RESOLVER_CONFIG_JSON"
MAIN_STORE_KEY = "DEV_RUNTIME_MAIN_STORE_SECRET_RESOLVER_CONFIG_JSON"
PROFILE_KEY = "DEV_RUNTIME_MAIN_RUNTIME_EFFECTS"
IDENTITY_KEYS = (
    "INFISICAL_ADDR",
    "INFISICAL_PROJECT_ID",
    "INFISICAL_CLIENT_ID",
    "INFISICAL_CLIENT_SECRET",
    "INFISICAL_ENVIRONMENT_SLUG",
)


def _config(rendered: dict[str, str], key: str) -> dict:
    return json.loads(rendered[key])


def _sources(config: dict) -> dict[str, dict]:
    return {m["logical_name"]: m["source"] for m in config["mappings"]}


def _namespaces(config: dict) -> dict[str, dict]:
    return {n["namespace"]: n for n in config.get("namespaces", [])}


def _missing_names(retired: dict, main: dict) -> list[str]:
    """Every name the retired service resolved that the runtime does not, by name."""
    missing = [
        name
        for name, source in _sources(retired).items()
        if _sources(main).get(name) != source
    ]
    missing += [
        f"namespace:{name}"
        for name, rule in _namespaces(retired).items()
        if _namespaces(main).get(name) != rule
    ]
    return sorted(missing)


# -- AC1: the secret resolver ------------------------------------------------


def test_runtime_resolves_every_name_the_retired_service_declared() -> None:
    rendered = render_env(load_contract())
    retired = _config(rendered, RETIRED_STORE_KEY)
    main = _config(rendered, MAIN_STORE_KEY)
    assert _sources(retired), "positive control: the retired service declares names"
    assert _namespaces(retired), "positive control: it declares a namespace rule"
    assert _missing_names(retired, main) == []


def test_comparison_names_a_missing_secret_and_a_missing_namespace() -> None:
    """FAILING CONTROL: the comparison itself goes red, by name, on a dropped entry."""
    rendered = render_env(load_contract())
    retired = _config(rendered, RETIRED_STORE_KEY)
    main = _config(rendered, MAIN_STORE_KEY)
    dropped = sorted(_sources(main))[0]
    short = {
        **main,
        "mappings": [m for m in main["mappings"] if m["logical_name"] != dropped],
        "namespaces": [],
    }
    assert _missing_names(retired, short) == sorted(
        [dropped, "namespace:tenant_inference_credentials"]
    )


def test_main_binds_the_store_backed_config_and_the_retired_identity() -> None:
    main_env = dev_stack_services()["omninode-runtime"]["environment"]
    effects_env = load_compose(DEV_LANE)["services"][SERVICE]["environment"]
    assert main_env["ONEX_SECRET_RESOLVER_CONFIG_JSON"] == (
        f"${{{MAIN_STORE_KEY}:?runtime policy contract must set {MAIN_STORE_KEY}}}"
    )
    # Same operator-env names as the retired service, so the same sources.
    for key in IDENTITY_KEYS:
        assert main_env[key] == effects_env[key], key
    # A value is only ever an interpolation of an operator env name.
    for key in IDENTITY_KEYS[:-1]:
        assert re.fullmatch(r"\$\{RUNTIME_EFFECTS_INFISICAL_\w+:-\}", main_env[key])


def test_resolving_the_declared_names_logs_no_value(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    main = _config(render_env(load_contract()), MAIN_STORE_KEY)
    config = ModelSecretResolverConfig.model_validate(
        {
            **main,
            "namespaces": [],
            "mappings": [
                m for m in main["mappings"] if m["source"]["source_type"] == "env"
            ],
        }
    )
    # Two logical names may share one env variable: key the sentinel by variable.
    by_path = {
        path: f"omn22930-sentinel-{index:03d}-value"
        for index, path in enumerate(
            sorted({m.source.source_path for m in config.mappings})
        )
    }
    expected = {}
    for mapping in config.mappings:
        monkeypatch.setenv(
            mapping.source.source_path, by_path[mapping.source.source_path]
        )
        expected[mapping.logical_name] = by_path[mapping.source.source_path]
    assert expected, "positive control: the config has env-sourced names"
    resolver = SecretResolver(config=config)
    with caplog.at_level(logging.DEBUG):
        for name, value in expected.items():
            secret = resolver.get_secret(name)
            assert secret is not None
            assert secret.get_secret_value() == value
    assert not any(v in caplog.text for v in expected.values())


# -- AC4 (policy half): the fold is declared and the service is retired ------


def test_policy_folds_effects_into_dev_main_and_hands_over_the_groups() -> None:
    policy = load_contract()
    process = policy.profiles["dev"].processes["main"]
    assert "effects" in process.folded_profiles
    assert process.foldable_services["effects"] == SERVICE
    handover = process.fold_handovers["effects"]
    assert handover.from_instance_id == "runtime-effects"
    # Effects handlers cannot take a repeat: the handover resumes from the
    # commits of the side that last ran, never a reset.
    assert handover.handover_rule == "last_ran"
    rendered = render_env(policy)
    assert "effects" in rendered["DEV_RUNTIME_MAIN_FOLDED_PROFILES"].split(",")
    assert (
        json.loads(rendered["DEV_RUNTIME_MAIN_FOLD_HANDOVERS"])["effects"][
            "from_instance_id"
        ]
        == "runtime-effects"
    )
    assert (DOCKER / "runtime-policy.env").read_text() == format_dotenv(rendered)


def test_service_is_gated_by_the_policy_not_by_a_literal() -> None:
    document = yaml.compose(DEV_LANE.read_text())
    top = next(v for k, v in document.value if k.value == "services")
    effects = next(v for k, v in top.value if k.value == SERVICE)
    profiles = next(v for k, v in effects.value if k.value == "profiles")
    assert profiles.tag == "!override"  # a plain list would merge with runtime/full
    assert [item.value for item in profiles.value] == [
        f"${{{PROFILE_KEY}_RUNTIME_PROFILE:-runtime}}",
        f"${{{PROFILE_KEY}_FULL_PROFILE:-full}}",
    ]
    rendered = render_env(load_contract())
    assert rendered[f"{PROFILE_KEY}_RUNTIME_PROFILE"] == "folded-effects-runtime"
    assert rendered[f"{PROFILE_KEY}_FULL_PROFILE"] == "folded-effects-full"


def test_unfolded_policy_restores_the_service_to_every_bring_up() -> None:
    """Rollback of the fold in policy: the same keys render the original profiles."""
    policy = load_contract()
    process = policy.profiles["dev"].processes["main"]
    kept = tuple(p for p in process.folded_profiles if p != "effects")
    rolled_back = process.model_copy(update={"folded_profiles": kept})
    dev = policy.profiles["dev"].model_copy(
        update={"processes": {**policy.profiles["dev"].processes, "main": rolled_back}}
    )
    rendered = render_env(
        policy.model_copy(update={"profiles": {**policy.profiles, "dev": dev}})
    )
    assert rendered[f"{PROFILE_KEY}_RUNTIME_PROFILE"] == "runtime"
    assert rendered[f"{PROFILE_KEY}_FULL_PROFILE"] == "full"


def test_main_keeps_the_retired_services_state_and_port() -> None:
    main = load_compose(OVERLAY)["services"]["omninode-runtime"]
    assert "effects_data:/app/effects-data" in main["volumes"]
    assert "effects_logs:/app/effects-logs:ro" in main["volumes"]
    assert any(
        str(p).endswith(":8085") and "DEV_RUNTIME_EFFECTS_PORT" in str(p)
        for p in main["ports"]
    )


# -- AC3 and AC4 (stack half): the dev stack check ---------------------------


def test_dev_stack_has_no_effects_service() -> None:
    assert SERVICE not in dev_stack_services()
    assert "omninode-runtime" in dev_stack_services()  # positive control


def test_dev_stack_check_refuses_an_effects_service_that_stays_active(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FAILING CONTROL: with the service's gate removed it is in the dev stack."""
    from tests.unit.docker import test_dev_tenant_projection_fold_omn21288 as shape

    text = DEV_LANE.read_text()
    gate = re.search(
        rf"  {SERVICE}:\n(?:    .*\n|\n)*?    profiles: !override\n(?:      - .*\n){{2}}",
        text,
    )
    assert gate, "the gate this control removes must be present"
    ungated = tmp_path / "docker-compose.dev-lane.yml"
    ungated.write_text(
        text[: gate.start()]
        + f"  {SERVICE}:\n    build: *dev_lane_composed_runtime_build\n"
        + text[gate.end() :]
    )
    monkeypatch.setattr(
        shape,
        "DEV_COMPOSE_FILES",
        (shape.DEV_COMPOSE_FILES[0], ungated, shape.DEV_COMPOSE_FILES[2]),
    )
    assert SERVICE in dev_stack_services()


def test_primary_dev_installer_derives_its_core_services_from_the_policy() -> None:
    refresh = (ROOT / "scripts/runtime_build/refresh_dev_lane.sh").read_text()
    line = next(
        line
        for line in refresh.splitlines()
        if line.startswith("readonly CORE_SERVICES=")
    )
    # The retired container is named in the installer only behind the policy's fold.
    assert "DEV_RUNTIME_MAIN_FOLDED_PROFILES" in line
    assert "== *,effects,*" in line
