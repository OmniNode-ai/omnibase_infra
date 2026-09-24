# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The laptop profile: catalog bundle ``local`` (OMN-19496).

A new engineer boots the ONEX stack plus their own runtime with one command and
one env file, with no lab or ops secret. These tests pin the properties that
make that true in the render, so a manifest edit cannot quietly reintroduce a
lab dependency, a shared Docker object name, or a second env source.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from omnibase_infra.docker.catalog import cli as catalog_cli
from omnibase_infra.docker.catalog.enum_infra_layer import EnumInfraLayer
from omnibase_infra.docker.catalog.generator import generate_compose
from omnibase_infra.docker.catalog.resolver import DEFAULT_PROJECT, CatalogResolver
from omnibase_infra.runtime.models.enum_bifrost_lane_locale import (
    EnumBifrostLaneLocale,
)
from omnibase_infra.runtime.models.model_bifrost_lane_overlay import (
    ModelBifrostLaneOverlay,
)

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_CATALOG_DIR = str(_REPO / "docker" / "catalog")
_ENV_TEMPLATE = _REPO / "docker" / "local.env.example"
_OVERLAY_TEMPLATE = _REPO / "docker" / "lane-overlays" / "local.bifrost.example.yaml"
_PROJECT = "omnibase-infra-local"
_OVERLAY_PIN = "/app/config/delegation/local.bifrost.yaml"

#: Every name the laptop profile may ask its operator for. Two local passwords
#: and the path of the model overlay; nothing else.
_LAPTOP_REQUIRED_ENV = {
    "POSTGRES_PASSWORD",
    "VALKEY_PASSWORD",
    "ONEX_LOCAL_BIFROST_OVERLAY",
}

#: Lab and ops credentials the profile must never require.
_FORBIDDEN_FRAGMENTS = (
    "DEPLOY_AGENT",
    "GITHUB",
    "LINEAR",
    "SLACK",
    "CI_CALLBACK",
    "INFISICAL",
    "KEYCLOAK",
    "SERVICE_CLIENT",
    "LLM_CODER",
    "LLM_DEEPSEEK",
    "LLM_EMBEDDING",
    "OPENROUTER",
    "GEMINI",
)


def test_laptop_required_env_is_two_passwords_and_the_overlay_path() -> None:
    resolved = CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(["local"])
    assert resolved.required_env == _LAPTOP_REQUIRED_ENV


def test_laptop_required_env_names_no_lab_or_ops_secret() -> None:
    resolved = CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(["local"])
    offenders = sorted(
        var
        for var in resolved.required_env
        for fragment in _FORBIDDEN_FRAGMENTS
        if fragment in var
    )
    assert offenders == []


def test_laptop_required_env_check_can_fail() -> None:
    """Positive control: the full runtime bundle does require lab and ops secrets."""
    resolved = CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(["runtime"])
    assert {"DEPLOY_AGENT_HMAC_SECRET", "GITHUB_TOKEN", "LLM_CODER_URL"} <= set(
        resolved.required_env
    )


def test_local_bundle_runs_both_runtime_kernels_and_the_migration_gate() -> None:
    resolved = CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(["local"])
    assert {
        "postgres",
        "redpanda",
        "valkey",
        "forward-migration",
        "migration-gate",
        "omninode-runtime",
        "runtime-effects",
    } <= set(resolved.manifests)
    runtime = [
        n for n, m in resolved.manifests.items() if m.layer == EnumInfraLayer.RUNTIME
    ]
    assert sorted(runtime) == ["omninode-runtime", "runtime-effects"]


def test_local_render_scopes_every_docker_object_to_its_project() -> None:
    resolved = CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(["local"])
    compose = generate_compose(resolved)

    assert compose["name"] == _PROJECT
    services = compose["services"]
    assert isinstance(services, dict)
    for name, svc in services.items():
        assert svc["container_name"] == f"{_PROJECT}-{name}"
        assert svc["networks"] == ["omnibase-infra-network"]
    networks = compose["networks"]
    assert isinstance(networks, dict)
    assert networks["omnibase-infra-network"]["name"] == f"{_PROJECT}-network"
    volumes = compose["volumes"]
    assert isinstance(volumes, dict)
    assert volumes
    for key, spec in volumes.items():
        assert spec["name"] == f"{_PROJECT}-{key}"
    assert services["omninode-runtime"]["image"] == f"{_PROJECT}-runtime:latest"
    assert services["runtime-effects"]["image"] == f"{_PROJECT}-runtime:latest"
    assert "build" in services["omninode-runtime"]


def test_default_project_render_keeps_every_historical_name() -> None:
    resolved = CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(["runtime"])
    assert resolved.project == DEFAULT_PROJECT
    compose = generate_compose(resolved)
    assert compose["name"] == "omnibase-infra"
    services = compose["services"]
    assert isinstance(services, dict)
    assert services["omninode-runtime"]["container_name"] == "omninode-runtime"
    assert services["omninode-runtime"]["image"] == "runtime:latest"
    assert services["postgres"]["container_name"] == "omnibase-infra-postgres"
    networks = compose["networks"]
    assert isinstance(networks, dict)
    assert networks["omnibase-infra-network"]["name"] == "omnibase-infra-network"
    volumes = compose["volumes"]
    assert isinstance(volumes, dict)
    assert volumes["postgres_data"] == {"name": "postgres_data"}


def test_local_runtime_kernels_mount_and_pin_the_local_overlay() -> None:
    resolved = CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(["local"])
    compose = generate_compose(resolved)
    services = compose["services"]
    assert isinstance(services, dict)
    for name in ("omninode-runtime", "runtime-effects"):
        svc = services[name]
        mounts = [v for v in svc["volumes"] if v.endswith(f":{_OVERLAY_PIN}:ro")]
        assert len(mounts) == 1, f"{name}: {svc['volumes']}"
        assert mounts[0].startswith("${ONEX_LOCAL_BIFROST_OVERLAY:?")
        env = svc["environment"]
        assert env["BIFROST_LANE_OVERLAY_PATH"] == _OVERLAY_PIN
        assert env["DELEGATION_ROUTING_TIERS_PATH"]
        assert env["GITHUB_TOKEN"] == ""
        assert env["DEPLOY_AGENT_HMAC_SECRET"] == ""
    # Infrastructure entries never receive the runtime overlay mount.
    assert not any(
        v.endswith(_OVERLAY_PIN + ":ro") for v in services["postgres"]["volumes"]
    )


def test_overlay_template_is_a_typed_lab_overlay_named_for_the_local_lane() -> None:
    overlay = ModelBifrostLaneOverlay.model_validate(
        yaml.safe_load(_OVERLAY_TEMPLATE.read_text(encoding="utf-8"))
    )
    assert overlay.lane == "local"
    assert overlay.locale is EnumBifrostLaneLocale.LAB
    endpoints = {binding.endpoint_url for binding in overlay.backends}
    assert len(endpoints) == 1, "one model endpoint setting, shared by both rungs"


def test_env_template_names_every_required_var_and_nothing_secret_from_the_lab() -> (
    None
):
    keys = {
        line.partition("=")[0]
        for line in _ENV_TEMPLATE.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#") and "=" in line
    }
    assert keys >= _LAPTOP_REQUIRED_ENV
    assert not [k for k in keys for f in _FORBIDDEN_FRAGMENTS if f in k]


def test_env_file_is_the_only_operator_env_source_and_loads_runtime_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "environ", {"PATH": os.environ.get("PATH", "")})
    home_env = tmp_path / "home.env"
    home_env.write_text("OMN19496_LEAK=from-home\n", encoding="utf-8")
    monkeypatch.setattr(catalog_cli, "_HOME_ENV", home_env)
    monkeypatch.setattr(catalog_cli, "_REPO_ENV", tmp_path / "absent-repo.env")
    env_file = tmp_path / "local.env"
    env_file.write_text("OMN19496_PROBE=from-env-file\n", encoding="utf-8")

    assert catalog_cli._load_stack_env(str(env_file)) == 0

    assert os.environ["OMN19496_PROBE"] == "from-env-file"
    assert "OMN19496_LEAK" not in os.environ
    assert os.environ["ONEX_ACTIVE_RUNTIME_PACKAGES"]


def test_env_file_with_template_placeholders_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "environ", {"PATH": os.environ.get("PATH", "")})
    assert catalog_cli._load_stack_env(str(_ENV_TEMPLATE)) == 1


def test_missing_env_file_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "environ", {"PATH": os.environ.get("PATH", "")})
    assert catalog_cli._load_stack_env(str(tmp_path / "absent.env")) == 1


def test_two_bundles_naming_different_projects_are_refused(tmp_path: Path) -> None:
    services = tmp_path / "services"
    services.mkdir()
    (tmp_path / "bundles.yaml").write_text(
        yaml.safe_dump(
            {
                "one": {"description": "a", "services": [], "project": "p-one"},
                "two": {"description": "b", "services": [], "project": "p-two"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Compose project conflict"):
        CatalogResolver(catalog_dir=str(tmp_path)).resolve(["one", "two"])


def _fixture_manifest(name: str, layer: str, required: list[str]) -> dict[str, object]:
    return {
        "name": name,
        "description": f"{name} fixture",
        "image": "busybox:1.36",
        "layer": layer,
        "required_env": required,
        "hardcoded_env": {"FIXTURE": "1"},
        "operational_defaults": {},
        "ports": None,
        "healthcheck": None,
        "volumes": [],
        "depends_on": [],
    }


def test_injected_env_satisfies_a_runtime_requirement_but_never_an_infra_one(
    tmp_path: Path,
) -> None:
    services = tmp_path / "services"
    services.mkdir()
    for name, layer, required in (
        ("kernel", "runtime", ["SHARED_VAR", "RUNTIME_ONLY_VAR"]),
        ("store", "infrastructure", ["SHARED_VAR"]),
    ):
        (services / f"{name}.yaml").write_text(
            yaml.safe_dump(_fixture_manifest(name, layer, required)),
            encoding="utf-8",
        )
    (tmp_path / "bundles.yaml").write_text(
        yaml.safe_dump(
            {
                "b": {
                    "description": "fixture",
                    "services": ["kernel", "store"],
                    "inject_env": {"SHARED_VAR": "x", "RUNTIME_ONLY_VAR": "y"},
                }
            }
        ),
        encoding="utf-8",
    )
    resolved = CatalogResolver(catalog_dir=str(tmp_path)).resolve(["b"])
    assert resolved.required_env == {"SHARED_VAR"}
