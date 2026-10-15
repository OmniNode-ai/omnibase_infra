# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The laptop profile: catalog bundle ``local`` (OMN-19496).

A new engineer boots the ONEX stack plus their own runtime with one command and
one env file, with no lab or ops secret. These tests pin the properties that
make that true in the render, so a manifest edit cannot quietly reintroduce a
lab dependency, a shared Docker object name, or a second env source.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping
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

#: Every name the laptop profile may ask its operator for. Five local passwords
#: (make local-env generates all five; the fifth is the broker superuser's,
#: OMN-20820) and the path of the model overlay; nothing else.
_LAPTOP_REQUIRED_ENV = {
    "POSTGRES_PASSWORD",
    "VALKEY_PASSWORD",
    "OMNINODE_RUNTIME_PASSWORD",
    "TENANT_PROJECTION_WRITER_PASSWORD",
    "REDPANDA_SUPERUSER_PASSWORD",
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


def test_laptop_required_env_is_five_passwords_and_the_overlay_path() -> None:
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


def test_local_bundle_runs_both_runtime_kernels_the_writer_and_the_migration_gate() -> (
    None
):
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
    assert sorted(runtime) == [
        "consumer-health-projection",
        "omnimarket-projection-delegation",
        "omnimarket-projection-llm-cost",
        "omnimarket-projection-tenant-credentials",
        "omninode-runtime",
        "projection-api",
        "runtime-effects",
        "tenant-projection-writer",
    ]


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


@pytest.mark.parametrize(
    "writer",
    ["omnimarket-projection-delegation", "omnimarket-projection-tenant-credentials"],
)
def test_local_projection_writer_binds_the_lab_principals_not_the_superuser(
    writer: str,
) -> None:
    """omnimarket's standalone writer attests the connected principal per binding.

    With the superuser DSNs the laptop bundle used to inject, the delegation
    writer crash-looped on a missing topology profile, then on principal
    ``postgres`` where ``omninode_runtime`` was expected, and no
    ``delegation_events`` row was ever written.
    """
    resolved = CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(["local"])
    services = generate_compose(resolved)["services"]
    assert isinstance(services, dict)
    env = services[writer]["environment"]
    assert env["ONEX_DATABASE_TOPOLOGY_PROFILE"] == "local"
    assert env["ONEX_TENANT_DB_URL"].startswith(
        "postgresql://tenant_projection_writer:${TENANT_PROJECTION_WRITER_PASSWORD:?"
    )
    assert env["OMNINODE_INTERNAL_DB_URL"].startswith(
        "postgresql://omninode_runtime:${OMNINODE_RUNTIME_PASSWORD:?"
    )
    # forward-migration is what gives those principals a LOGIN credential.
    migration_env = services["forward-migration"]["environment"]
    assert (
        migration_env["OMNINODE_RUNTIME_PASSWORD"] == "${OMNINODE_RUNTIME_PASSWORD:-}"
    )
    assert (
        migration_env["TENANT_PROJECTION_WRITER_PASSWORD"]
        == "${TENANT_PROJECTION_WRITER_PASSWORD:-}"
    )


def test_local_tenant_credentials_projection_runs_with_its_own_healthcheck() -> None:
    resolved = CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(["local"])
    services = generate_compose(resolved)["services"]
    assert isinstance(services, dict)
    writer = services["omnimarket-projection-tenant-credentials"]
    assert writer["image"] == f"{_PROJECT}-runtime:latest"
    assert writer["command"] == [
        "python",
        "-m",
        "omnimarket.nodes.node_projection_tenant_credentials.handlers."
        "handler_tenant_credentials_projection",
    ]
    assert writer["environment"]["KAFKA_CONSUMER_GROUP"] == (
        "local.omnimarket-projections.tenant-credentials-writer.consume.v1"
    )
    assert writer["environment"]["PROJECTION_RUNNER_HEALTH_PORT"] == "8102"
    assert writer["healthcheck"]["test"] == [
        "CMD-SHELL",
        "curl -sf http://localhost:8102/ready",
    ]
    assert (
        writer["depends_on"]
        == services["omnimarket-projection-delegation"]["depends_on"]
    )
    assert set(
        resolved.manifests["omnimarket-projection-tenant-credentials"].required_env
    ) == {
        "OMNIDASH_ANALYTICS_DB_URL",
        "ONEX_TENANT_DB_URL",
        "OMNINODE_INTERNAL_DB_URL",
    }


def test_local_bundle_runs_the_tenant_projection_carrier_under_its_own_identity() -> (
    None
):
    """OMN-19972 (T3.6): the usage-by-model writer runs in this kernel.

    ``node_projection_usage_by_model_day`` and the other contracts pinned to
    ``runtime_profiles: [tenant-projection]`` are dropped from ``main`` and
    ``effects``; without this process the laptop consumes them with nothing. It
    must be its own claimant: main's group id or capabilities would put a
    second owner on main's topics.
    """
    resolved = CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(["local"])
    services = generate_compose(resolved)["services"]
    assert isinstance(services, dict)
    carrier = services["tenant-projection-writer"]
    env = carrier["environment"]
    assert carrier["image"] == f"{_PROJECT}-runtime:latest"
    assert env["RUNTIME_PROFILE"] == "tenant-projection"
    kernels = ("omninode-runtime", "runtime-effects")
    assert env["ONEX_GROUP_ID"] not in {
        services[k]["environment"]["ONEX_GROUP_ID"] for k in kernels
    }
    assert env["ONEX_RUNTIME_CAPABILITIES"] not in {
        services[k]["environment"]["ONEX_RUNTIME_CAPABILITIES"] for k in kernels
    }
    # The binding the process exists for, as the laptop's own principal.
    assert env["ONEX_TENANT_DB_URL"].startswith(
        "postgresql://tenant_projection_writer:${TENANT_PROJECTION_WRITER_PASSWORD:?"
    )
    assert env["ONEX_DATABASE_TOPOLOGY_PROFILE"] == "local"
    # Lab and ops integrations stay off, as on the other two kernels.
    assert env["KEYCLOAK_ADMIN_URL"] == ""
    assert "healthcheck" in carrier
    assert carrier["depends_on"]["migration-gate"] == {"condition": "service_healthy"}


def test_the_runtime_family_that_resolves_a_tenant_key_shares_one_credentials_store() -> (
    None
):
    """The main runtime hosts the LLM call effect; runtime-effects hosts intake.

    Measured on the lab (2026-09-30): with the store on runtime-effects alone,
    the main runtime could not resolve the tenant's minted reference ("could
    not be resolved from the secret store") and the tenant route failed.
    """
    resolved = CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(["local"])
    compose = generate_compose(resolved)
    services = compose["services"]
    assert isinstance(services, dict)
    target = "/home/omniinfra/.omninode/delegation"
    owners = {
        name: [v for v in svc.get("volumes", []) if f":{target}" in v]
        for name, svc in services.items()
        if any(f":{target}" in v for v in svc.get("volumes", []))
    }
    assert owners == {
        "omninode-runtime": [f"delegation_credentials:{target}"],
        "runtime-effects": [f"delegation_credentials:{target}"],
    }
    volumes = compose["volumes"]
    assert isinstance(volumes, dict)
    assert volumes["delegation_credentials"] == {
        "name": f"{_PROJECT}-delegation_credentials"
    }


def _provider_credential_env_names(environment: Mapping[str, object]) -> list[str]:
    return sorted(
        name
        for name in environment
        if any(
            fragment in name.upper()
            for fragment in ("GEMINI", "OPENROUTER", "GLM_", "OPENAI", "ANTHROPIC")
        )
        or name.lower().startswith("llm.")
    )


def test_local_render_has_no_provider_credential_environment_names() -> None:
    resolved = CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(["local"])
    services = generate_compose(resolved)["services"]
    assert isinstance(services, dict)
    assert {
        name: hits
        for name, svc in services.items()
        if (hits := _provider_credential_env_names(svc.get("environment", {})))
    } == {}


def test_provider_credential_environment_scan_can_fail() -> None:
    forbidden = {
        "GEMINI_API_KEY": "",
        "OPENAI_API_KEY": "",
        "OPENROUTER_TOKEN": "",
        "GLM_TOKEN": "",
        "llm.example.secret": "",
    }
    assert _provider_credential_env_names({**forbidden, "ONEX_TENANT_ID": ""}) == (
        sorted(forbidden)
    )


def test_local_lane_tenant_is_optional_and_injected_into_every_runtime_service() -> (
    None
):
    resolved = CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(["local"])
    services = generate_compose(resolved)["services"]
    assert isinstance(services, dict)
    assert "ONEX_TENANT_ID" not in resolved.required_env
    for name, manifest in resolved.manifests.items():
        assert "ONEX_TENANT_ID" not in manifest.required_env
        env = services[name].get("environment", {})
        if (
            manifest.layer == EnumInfraLayer.RUNTIME
            or name == "local-tenant-registry-seed"
        ):
            assert env["ONEX_TENANT_ID"] == "${ONEX_TENANT_ID:-}"
        else:
            assert "ONEX_TENANT_ID" not in env


def test_secret_local_make_target_passes_only_provider_and_tenant_as_arguments() -> (
    None
):
    makefile = (_REPO / "Makefile").read_text(encoding="utf-8")
    target = makefile.split("\nsecret-local:", 1)[1].split("\n\n", 1)[0]
    assert 'test -n "$(PROVIDER)"' in target
    assert "ONEX_TENANT_ID=" in target
    assert '"$(LOCAL_ENV_FILE)"' in target
    assert 'test -n "$$tenant"' in target
    # The key is registered only for the tenant the RUNNING runtime serves.
    assert "printenv ONEX_TENANT_ID" in target
    commands = [
        line
        for line in target.splitlines()
        if "docker exec -u" in line and " -i " in line
    ]
    assert commands
    for command in commands:
        assert (
            "docker exec -u omniinfra -e HOME=/home/omniinfra -i "
            "$(LOCAL_PROJECT)-runtime-effects onex secret "
            'register-tenant-key "$(PROVIDER)" --tenant "$$tenant"'
        ) in command
        assert not re.search(r"\$[({][^)}]*(?:KEY|VALUE)[^)}]*[)}]", command)
    assert "read -r -s" in target  # Terminal input must be hidden.


def test_delegate_local_make_target_runs_the_cli_as_the_runtime_user() -> None:
    # `docker exec` defaults to root. The CLI creates <state root> on its first
    # run, so a root-run delegation leaves a root-owned directory in the
    # runtime's data volume and every later delegation by the runtime user
    # fails with "Permission denied: .../.onex_state/delegation".
    makefile = (_REPO / "Makefile").read_text(encoding="utf-8")
    target = makefile.split("\ndelegate-local:", 1)[1].split("\n\n", 1)[0]
    commands = [line for line in target.splitlines() if "docker exec" in line]
    assert len(commands) == 1
    assert (
        commands[0]
        .strip()
        .startswith(
            "docker exec -u omniinfra -e HOME=/home/omniinfra "
            "$(LOCAL_PROJECT)-runtime-effects onex delegate"
        )
    )


def _make(target: str, env_file: Path, overlay_file: Path) -> None:
    result = subprocess.run(
        [
            "make",
            target,
            f"LOCAL_ENV_FILE={env_file}",
            f"LOCAL_OVERLAY_FILE={overlay_file}",
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_local_env_leaves_the_runtime_without_a_tenant(tmp_path: Path) -> None:
    """A tenant makes every delegation BYOK-only, so it is never the default."""
    env_file = tmp_path / "local.env"
    _make("local-env", env_file, tmp_path / "local.bifrost.yaml")
    lines = env_file.read_text(encoding="utf-8").splitlines()
    assert "ONEX_TENANT_ID=" in lines


@pytest.mark.parametrize("existing_tenant", [None, "absent", "", "local-kept"])
def test_tenant_local_generates_a_tenant_once(
    tmp_path: Path, existing_tenant: str | None
) -> None:
    env_file = tmp_path / "local.env"
    overlay_file = tmp_path / "local.bifrost.yaml"
    if existing_tenant is not None:
        contents = "POSTGRES_PASSWORD=kept\n"
        if existing_tenant != "absent":
            contents += f"ONEX_TENANT_ID={existing_tenant}\n"
        env_file.write_text(contents, encoding="utf-8")
    _make("tenant-local", env_file, overlay_file)
    tenant_lines = [
        line
        for line in env_file.read_text().splitlines()
        if line.startswith("ONEX_TENANT_ID=")
    ]
    assert len(tenant_lines) == 1
    tenant = tenant_lines[0].partition("=")[2]
    if existing_tenant == "local-kept":
        assert tenant == existing_tenant
    else:
        assert re.fullmatch(r"local-[0-9a-f]{12}", tenant)
    _make("tenant-local", env_file, overlay_file)
    assert tenant_lines[0] in env_file.read_text().splitlines()


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


def test_up_precleanup_removes_anonymous_volumes_of_removed_containers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measured on .105: without -v every re-up orphaned the one-shots' volumes."""
    services = tmp_path / "catalog" / "services"
    services.mkdir(parents=True)
    (services / "store.yaml").write_text(
        yaml.safe_dump(_fixture_manifest("store", "infrastructure", [])),
        encoding="utf-8",
    )
    (tmp_path / "catalog" / "bundles.yaml").write_text(
        yaml.safe_dump({"b": {"description": "fixture", "services": ["store"]}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(os, "environ", {"PATH": os.environ.get("PATH", "")})
    monkeypatch.setattr(catalog_cli, "_CATALOG_DIR", str(tmp_path / "catalog"))
    monkeypatch.setattr(catalog_cli, "_DEFAULT_OUTPUT", str(tmp_path / "compose.yml"))
    monkeypatch.setattr(catalog_cli, "_STACK_FILE", str(tmp_path / "stack.yml"))
    env_file = tmp_path / "stack.env"
    env_file.write_text("FIXTURE_ONLY=1\n", encoding="utf-8")
    calls: list[list[str]] = []

    class _Done:
        returncode = 0

    def _fake_run(command: list[str], **_: object) -> _Done:
        calls.append(list(command))
        return _Done()

    monkeypatch.setattr(catalog_cli.subprocess, "run", _fake_run)

    assert catalog_cli.cmd_up(["b", "--env-file", str(env_file)]) == 0

    rm_calls = [c for c in calls if "rm" in c]
    assert rm_calls == [
        [
            "docker",
            "compose",
            "-f",
            str(tmp_path / "compose.yml"),
            "rm",
            "-f",
            "--stop",
            "-v",
        ]
    ]


# --- OMN-19972: the laptop profile publishes on loopback only and names no lab host.
#
# Failure modes these pin (spec, workflow/records/plans/OMN-19972):
#   1. a published port binds all interfaces in the ``local`` render;
#   2. a lab address (the .201 LAN IP, the tailnet domain, the lab hostname)
#      appears anywhere in the ``local`` render;
#   3. the loopback bind or the default override leaks into another bundle,
#      which the lab lanes render from the same shared manifests.

_LOOPBACK = "127.0.0.1"
_LAB_HOST_MARKERS = ("192.168.86.", "tail75df5e", "omninode-pc")


def _published_ports(compose: dict[str, object]) -> dict[str, list[str]]:
    services = compose["services"]
    assert isinstance(services, dict)
    return {
        name: [str(p) for p in svc["ports"]]
        for name, svc in services.items()
        if svc.get("ports")
    }


def _lab_host_hits(compose: dict[str, object]) -> list[str]:
    text = yaml.safe_dump(compose, sort_keys=True)
    return [marker for marker in _LAB_HOST_MARKERS if marker in text]


def test_local_render_publishes_every_port_on_loopback_only() -> None:
    compose = generate_compose(
        CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(["local"])
    )
    published = _published_ports(compose)
    # The five host ports the laptop guide names; a render that published none
    # would pass the loopback check vacuously.
    assert {
        "postgres",
        "redpanda",
        "valkey",
        "omninode-runtime",
        "runtime-effects",
    } <= set(published)
    not_loopback = {
        name: ports
        for name, ports in published.items()
        if any(not p.startswith(f"{_LOOPBACK}:") or p.count(":") != 2 for p in ports)
    }
    assert not_loopback == {}


def test_local_render_names_no_lab_host() -> None:
    compose = generate_compose(
        CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(["local"])
    )
    assert _lab_host_hits(compose) == []


def test_lab_host_check_can_fail() -> None:
    """Positive control: the lab-host check finds a marker when one is present."""
    assert _lab_host_hits({"command": f"{_LAB_HOST_MARKERS[0]}1"}) != []


@pytest.mark.parametrize("bundle", ["core", "runtime"])
def test_other_bundles_keep_all_interface_ports_and_name_no_lab_host(
    bundle: str,
) -> None:
    """OMN-20931: the shared manifests carry the neutral advertise default.

    A deployment's own env file sets REDPANDA_ADVERTISE_HOST; no render of the
    shipped catalog names a lab host.
    """
    compose = generate_compose(
        CatalogResolver(catalog_dir=_CATALOG_DIR).resolve([bundle])
    )
    published = _published_ports(compose)
    assert published
    for ports in published.values():
        for port in ports:
            assert port.count(":") == 1, port
            assert not port.startswith(f"{_LOOPBACK}:"), port
    services = compose["services"]
    assert isinstance(services, dict)
    command = " ".join(services["redpanda"]["command"])
    assert "${REDPANDA_ADVERTISE_HOST:-localhost}" in command
    assert _lab_host_hits(compose) == []


# --- OMN-19972 hostile-review follow-ups: env_default_overrides must fail loudly
# rather than render a broken compose file, and must not trip over a non-string
# command part.


def _override_catalog(
    tmp_path: Path, command: list[object], overrides: dict[str, str]
) -> CatalogResolver:
    services = tmp_path / "services"
    services.mkdir()
    manifest = _fixture_manifest("svc", "infrastructure", [])
    manifest["command"] = command
    (services / "svc.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (tmp_path / "bundles.yaml").write_text(
        yaml.safe_dump(
            {
                "b": {
                    "description": "b",
                    "services": ["svc"],
                    "env_default_overrides": overrides,
                }
            }
        ),
        encoding="utf-8",
    )
    return CatalogResolver(catalog_dir=str(tmp_path))


def test_override_of_a_nested_default_is_refused(tmp_path: Path) -> None:
    """``${A:-${B:-x}}`` cannot be rewritten safely; a half-rewrite breaks compose."""
    resolver = _override_catalog(
        tmp_path, ["run", "--addr ${ADV:-${OTHER:-lab}}:1"], {"ADV": "localhost"}
    )
    with pytest.raises(ValueError, match="nested"):
        generate_compose(resolver.resolve(["b"]))


@pytest.mark.parametrize("bad", ["a}b", "a$b"])
def test_override_value_that_would_break_interpolation_is_refused(
    tmp_path: Path, bad: str
) -> None:
    resolver = _override_catalog(tmp_path, ["run"], {"ADV": bad})
    with pytest.raises(ValueError, match="env_default_overrides"):
        resolver.resolve(["b"])


def test_override_leaves_a_non_string_command_part_untouched(tmp_path: Path) -> None:
    resolver = _override_catalog(
        tmp_path, ["sleep", 5, "--addr ${ADV:-lab}"], {"ADV": "localhost"}
    )
    compose = generate_compose(resolver.resolve(["b"]))
    services = compose["services"]
    assert isinstance(services, dict)
    assert services["svc"]["command"] == ["sleep", 5, "--addr ${ADV:-localhost}"]


# --- OMN-19972 demo half (plan T4.1): the services the six pages read ----------
#
# Failure modes these tests are written against, each shown failing on dev
# 83fa0e0c3 before the change:
#   1. an added service is missing from the laptop render, or runs with no
#      health signal, so the CI boot cannot tell it is dead;
#   2. the laptop's projection API keeps port 3002, the lab lanes' port;
#   3. the projection API starts before the kernel has provisioned the exposure
#      topics (measured on the lakshman lane 2026-09-30: it waits 300 s, exits
#      with "no partition metadata" and loops);
#   4. a laptop-only override leaks into another bundle rendered from the same
#      shared manifests;
#   5. the llm-cost writer names an image no catalog render builds.
#   6. the projection API is given no Kafka broker and exits at startup (measured in
#      CI run 36729468068 attempt 2: "projection-api requires Kafka bootstrap
#      servers", 16 restarts, never healthy).

_PAGE_SERVICES = (
    "projection-api",
    "omnimarket-projection-llm-cost",
    "consumer-health-projection",
)
_LAPTOP_PROJECTION_API_PORT = 3102


def _render(*bundles: str) -> dict[str, object]:
    return generate_compose(
        CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(list(bundles))
    )


def _services(compose: dict[str, object]) -> dict[str, dict[str, object]]:
    services = compose["services"]
    assert isinstance(services, dict)
    return services


def _seed_script() -> str:
    """The seed one-shot's shell, as the container's ``sh -c`` receives it."""
    svc = _services(_render("local"))["local-tenant-registry-seed"]
    assert svc["entrypoint"] == ["/bin/sh", "-c"]
    command = svc["command"]
    assert isinstance(command, list) and len(command) == 1
    return str(command[0]).replace("$$", "$")


def _run_seed(tmp_path: Path, tenant: str) -> subprocess.CompletedProcess[str]:
    """Run the seed shell against a stub ``psql`` that logs its arguments."""
    calls = tmp_path / "psql-calls"
    psql = tmp_path / "psql"
    psql.write_text(
        '#!/bin/sh\necho "$*" >> "$PSQL_CALLS"\ncat > /dev/null\n'
        "echo 00000000-0000-0000-0000-000000000001\n",
        encoding="utf-8",
    )
    psql.chmod(0o755)
    return subprocess.run(
        ["sh", "-c", _seed_script()],
        env={
            **os.environ,
            "ONEX_TENANT_ID": tenant,
            "POSTGRES_PASSWORD": "unused",
            "POSTGRES_HOST": "postgres",
            "POSTGRES_PORT": "5432",
            "POSTGRES_USER": "postgres",
            "NODE_POSTGRES_DB": "omnidash_analytics",
            "PATH": f"{tmp_path}:{os.environ.get('PATH', '')}",
            "PSQL_CALLS": str(calls),
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_local_render_seeds_its_tenant_after_migrations() -> None:
    svc = _services(_render("local"))["local-tenant-registry-seed"]
    assert svc["restart"] == "no"
    assert svc["depends_on"] == {"migration-gate": {"condition": "service_healthy"}}
    env = svc["environment"]
    assert isinstance(env, dict)
    assert env["ONEX_TENANT_ID"] == "${ONEX_TENANT_ID:-}"
    assert env["NODE_POSTGRES_DB"] == "omnidash_analytics"


def test_only_local_bundle_lists_the_tenant_registry_seed() -> None:
    bundles = yaml.safe_load(
        (_REPO / "docker" / "catalog" / "bundles.yaml").read_text(encoding="utf-8")
    )
    assert [
        name
        for name, bundle in bundles.items()
        if "local-tenant-registry-seed" in bundle.get("services", [])
    ] == ["local"]


def test_local_tenant_registry_seed_preserves_existing_mappings() -> None:
    script = _seed_script()
    assert "INSERT INTO tenant_registry_mirror" in script
    assert "ON CONFLICT (tenant_slug) DO NOTHING" in script
    assert "md5('omninode-local-tenant:' || :'slug')::uuid" in script


def test_local_tenant_registry_seed_records_the_tenant(tmp_path: Path) -> None:
    result = _run_seed(tmp_path, "local-dd95787510f8")
    assert result.returncode == 0, result.stderr
    assert "tenant local-dd95787510f8 -> " in result.stdout
    calls = (tmp_path / "psql-calls").read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert all("slug=local-dd95787510f8" in call for call in calls)


@pytest.mark.parametrize("tenant", ["", "Bad Slug!"])
def test_local_tenant_registry_seed_skips_database_for_empty_or_invalid_tenant(
    tmp_path: Path, tenant: str
) -> None:
    result = _run_seed(tmp_path, tenant)
    if tenant:
        assert result.returncode != 0
        assert "invalid ONEX_TENANT_ID" in result.stderr
    else:
        assert result.returncode == 0
        assert "no tenant, nothing to record" in result.stdout
    assert not (tmp_path / "psql-calls").exists()


def test_local_render_carries_the_services_the_pages_read() -> None:
    assert set(_PAGE_SERVICES) <= set(_services(_render("local")))


def test_every_long_running_local_service_has_a_healthcheck() -> None:
    services = _services(_render("local"))
    unwatched = sorted(
        name
        for name, svc in services.items()
        if svc.get("restart") != "no" and "healthcheck" not in svc
    )
    assert unwatched == []


def test_local_projection_api_publishes_a_laptop_port_not_the_lab_port() -> None:
    ports = _services(_render("local"))["projection-api"]["ports"]
    assert ports == [f"{_LOOPBACK}:{_LAPTOP_PROJECTION_API_PORT}:3002"]


def test_other_bundles_keep_the_projection_api_on_3002() -> None:
    ports = _services(_render("runtime-observability-projections"))["projection-api"][
        "ports"
    ]
    assert ports == ["3002:3002"]


def test_local_projection_api_waits_for_the_kernel_that_provisions_its_topics() -> None:
    depends_on = _services(_render("local"))["projection-api"]["depends_on"]
    assert isinstance(depends_on, dict)
    assert depends_on.get("omninode-runtime") == {"condition": "service_healthy"}


def test_other_bundles_do_not_gain_the_kernel_dependency() -> None:
    depends_on = _services(_render("runtime-observability-projections"))[
        "projection-api"
    ]["depends_on"]
    assert isinstance(depends_on, dict)
    assert "omninode-runtime" not in depends_on


def test_local_projection_api_is_given_the_compose_broker() -> None:
    env = _services(_render("local"))["projection-api"]["environment"]
    assert isinstance(env, dict)
    assert env.get("KAFKA_BROKERS") == "redpanda:9092"


def test_llm_cost_writer_builds_from_the_runtime_image_and_reports_ready() -> None:
    for bundle in ("local", "omnimarket-projections"):
        svc = _services(_render(bundle))["omnimarket-projection-llm-cost"]
        assert "omnimarket-projection:latest" not in str(svc["image"]), bundle
        test = svc["healthcheck"]["test"]  # type: ignore[index]
        assert "/ready" in " ".join(test), bundle
        env = svc["environment"]
        assert isinstance(env, dict)
        assert env.get("PROJECTION_RUNNER_HEALTH_PORT"), bundle


def test_two_bundles_overriding_one_service_port_differently_are_refused(
    tmp_path: Path,
) -> None:
    import shutil

    catalog = tmp_path / "catalog"
    shutil.copytree(_CATALOG_DIR, catalog)
    bundles_file = catalog / "bundles.yaml"
    bundles = yaml.safe_load(bundles_file.read_text())
    bundles["other-laptop"] = {
        "description": "fixture: a second bundle overriding the same port",
        "services": ["projection-api"],
        "port_overrides": {"projection-api": 3999},
    }
    bundles_file.write_text(yaml.safe_dump(bundles, sort_keys=False))
    with pytest.raises(ValueError, match=r"[Pp]ort override conflict"):
        CatalogResolver(catalog_dir=str(catalog)).resolve(["local", "other-laptop"])


# The resolver's other override guards (hostile review, both models: each was
# untested). Each test adds fixture bundles to a copy of the real catalog.


def _catalog_with(tmp_path: Path, extra: dict[str, object]) -> CatalogResolver:
    import shutil

    catalog = tmp_path / "catalog"
    shutil.copytree(_CATALOG_DIR, catalog)
    bundles_file = catalog / "bundles.yaml"
    bundles = yaml.safe_load(bundles_file.read_text())
    bundles.update(extra)
    bundles_file.write_text(yaml.safe_dump(bundles, sort_keys=False))
    return CatalogResolver(catalog_dir=str(catalog))


def test_two_bundles_waiting_differently_on_one_dependency_are_refused(
    tmp_path: Path,
) -> None:
    resolver = _catalog_with(
        tmp_path,
        {
            "other-laptop": {
                "description": "fixture: waits on the kernel as merely started",
                "services": ["projection-api"],
                "extra_depends_on": {
                    "projection-api": {"omninode-runtime": "service_started"}
                },
            }
        },
    )
    with pytest.raises(ValueError, match="Extra dependency conflict"):
        resolver.resolve(["local", "other-laptop"])


def test_extra_dependency_on_a_service_the_stack_does_not_run_is_refused(
    tmp_path: Path,
) -> None:
    resolver = _catalog_with(
        tmp_path,
        {
            "probe": {
                "description": "fixture: redpanda alone, waiting on postgres",
                "services": ["redpanda"],
                "extra_depends_on": {"redpanda": {"postgres": "service_healthy"}},
            }
        },
    )
    with pytest.raises(ValueError, match="Extra dependency names 'postgres'"):
        resolver.resolve(["probe"])


def test_port_override_for_a_service_the_stack_does_not_run_is_refused(
    tmp_path: Path,
) -> None:
    resolver = _catalog_with(
        tmp_path,
        {
            "probe": {
                "description": "fixture: overrides a service it does not run",
                "services": ["redpanda"],
                "port_overrides": {"projection-api": 3999},
            }
        },
    )
    with pytest.raises(ValueError, match="Port override for 'projection-api'"):
        resolver.resolve(["probe"])


def test_port_override_for_a_service_that_publishes_no_port_is_refused(
    tmp_path: Path,
) -> None:
    resolver = _catalog_with(
        tmp_path,
        {
            "probe": {
                "description": "fixture: overrides a writer that publishes no port",
                "services": ["omnimarket-projection-llm-cost"],
                "port_overrides": {"omnimarket-projection-llm-cost": 3999},
            }
        },
    )
    with pytest.raises(ValueError, match="publishes no port"):
        resolver.resolve(["probe"])


def test_unknown_dependency_condition_is_refused(tmp_path: Path) -> None:
    resolver = _catalog_with(
        tmp_path,
        {
            "probe": {
                "description": "fixture: a condition compose does not know",
                "services": ["projection-api"],
                "extra_depends_on": {"projection-api": {"redpanda": "service_happy"}},
            }
        },
    )
    # Resolved alone, so no other bundle's condition can conflict with it: only
    # the condition check can refuse (with "local", the conflict guard answered
    # first and this test passed with the condition check removed).
    with pytest.raises(ValueError, match="'service_happy' is not a valid"):
        resolver.resolve(["probe"])


# --- OMN-20820: SASL on the laptop broker (bus-mint plan T1) -------------------
#
# The laptop broker used to accept any client, so a developer's stack proved
# nothing about the authenticated path every staging and production client
# takes. `redpanda-local-sasl` creates one SCRAM superuser, enables SASL and
# proves the refusal; every Kafka client of the bundle authenticates as that
# user and waits for the one-shot to finish.

_SASL_SERVICE = "redpanda-local-sasl"
_SASL_USER = "onex-local"
_SASL_ENV_NAMES = (
    "KAFKA_SECURITY_PROTOCOL",
    "KAFKA_SASL_MECHANISM",
    "KAFKA_SASL_USERNAME",
    "KAFKA_SASL_PASSWORD",
)
_SASL_COMPLETED = {"condition": "service_completed_successfully"}


def _sasl_clients(compose: dict[str, object]) -> list[str]:
    """Every service of the render whose environment names the broker."""
    clients = []
    for name, svc in _services(compose).items():
        env = svc.get("environment", {})
        assert isinstance(env, dict)
        if any("redpanda:9092" in str(value) for value in env.values()):
            clients.append(name)
    return sorted(clients)


def test_local_render_has_the_redpanda_local_sasl_one_shot_sasl() -> None:
    svc = _services(_render("local"))[_SASL_SERVICE]
    assert svc["restart"] == "no"
    assert svc["container_name"] == f"{_PROJECT}-{_SASL_SERVICE}"
    assert svc["entrypoint"] == ["/bin/sh", "-c"]
    assert "healthcheck" not in svc
    depends_on = svc["depends_on"]
    assert isinstance(depends_on, dict)
    assert depends_on["redpanda"] == {"condition": "service_healthy"}
    env = svc["environment"]
    assert isinstance(env, dict)
    assert env["KAFKA_SASL_USERNAME"] == _SASL_USER
    assert str(env["KAFKA_SASL_PASSWORD"]).startswith("${REDPANDA_SUPERUSER_PASSWORD:?")


def test_local_kafka_clients_carry_the_four_sasl_variables_and_wait_sasl() -> None:
    compose = _render("local")
    services = _services(compose)
    clients = _sasl_clients(compose)
    runtime_layer = sorted(
        name
        for name, m in CatalogResolver(catalog_dir=_CATALOG_DIR)
        .resolve(["local"])
        .manifests.items()
        if m.layer == EnumInfraLayer.RUNTIME
    )
    # Every runtime-layer service of the bundle is a broker client.
    assert clients == runtime_layer
    assert len(clients) >= 7
    for name in clients:
        env = services[name]["environment"]
        assert isinstance(env, dict)
        assert env["KAFKA_SECURITY_PROTOCOL"] == "SASL_PLAINTEXT", name
        assert env["KAFKA_SASL_MECHANISM"] == "SCRAM-SHA-256", name
        assert env["KAFKA_SASL_USERNAME"] == _SASL_USER, name
        assert str(env["KAFKA_SASL_PASSWORD"]).startswith(
            "${REDPANDA_SUPERUSER_PASSWORD:?"
        ), name
        depends_on = services[name]["depends_on"]
        assert isinstance(depends_on, dict)
        assert depends_on[_SASL_SERVICE] == _SASL_COMPLETED, name
        # The broker wait the service already had is kept.
        assert depends_on["redpanda"] == {"condition": "service_healthy"}, name


def test_local_sasl_variables_stay_off_everything_that_is_not_a_client_sasl() -> None:
    """Positive control: the check above can fail, and infrastructure is untouched."""
    services = _services(_render("local"))
    for name in ("postgres", "redpanda", "valkey", "forward-migration"):
        env = services[name].get("environment", {})
        assert isinstance(env, dict)
        assert not [k for k in env if k in _SASL_ENV_NAMES], name
    assert _SASL_SERVICE not in services["redpanda"].get("depends_on", {})


def test_local_sasl_password_is_never_rendered_inline_sasl() -> None:
    rendered = yaml.safe_dump(_render("local"))
    assert "REDPANDA_SUPERUSER_PASSWORD" in rendered
    for match in re.finditer(r"KAFKA_SASL_PASSWORD: (.*)", rendered):
        assert match.group(1).strip().startswith("${REDPANDA_SUPERUSER_PASSWORD:?")


def _sasl_script() -> str:
    svc = _services(_render("local"))[_SASL_SERVICE]
    command = svc["command"]
    assert isinstance(command, list) and len(command) == 1
    return str(command[0]).replace("$$", "$")


_STUB_RPK = """#!/bin/sh
# Stateful stub of rpk for the one-shot: state lives in $STUB_DIR.
echo "$*" >> "$STUB_DIR/calls"
args="$*"
case "$args" in
  "security user list"*)
    [ -f "$STUB_DIR/users" ] && sed 's/$/  SCRAM-SHA-256/' "$STUB_DIR/users"
    echo "USERNAME  MECHANISM"
    exit 0 ;;
  "security user create"*)
    echo "$4" >> "$STUB_DIR/users"; exit 0 ;;
  "cluster config set superusers"*)
    echo "$args" > "$STUB_DIR/superusers"; exit 0 ;;
  "cluster config set enable_sasl true"*)
    # The flip is only safe after the superuser grant (OMN-18012).
    [ -f "$STUB_DIR/superusers" ] || { echo "enable_sasl before superusers" >> "$STUB_DIR/order-violation"; }
    touch "$STUB_DIR/sasl-on"; exit 0 ;;
  "cluster config get superusers"*)
    cat "$STUB_DIR/superusers"; exit 0 ;;
  "cluster config get enable_sasl"*)
    [ -f "$STUB_DIR/sasl-on" ] && echo true || echo false; exit 0 ;;
  "topic list"*)
    if [ -n "${RPK_USER:-}" ] && [ -n "${RPK_PASS:-}" ]; then
      [ "$RPK_PASS" = "$STUB_PASSWORD" ] || { echo "SASL authentication failed" >&2; exit 1; }
      exit 0
    fi
    if [ -f "$STUB_DIR/sasl-on" ] && [ -z "${STUB_BROKEN_SASL:-}" ]; then
      echo "SASL is required but not provided" >&2; exit 1
    fi
    exit 0 ;;
esac
echo "stub rpk: unexpected $args" >&2; exit 2
"""


def _run_sasl_script(
    tmp_path: Path, *, broken_sasl: bool = False, rounds: int = 1, supplied: str = "pw"
) -> tuple[subprocess.CompletedProcess[str], Path]:
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir(exist_ok=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    # The script names rpk by absolute path, as the container image has it.
    script = _sasl_script()
    rpk = bin_dir / "rpk"
    rpk.write_text(_STUB_RPK, encoding="utf-8")
    rpk.chmod(0o755)
    script = script.replace("/usr/bin/rpk", str(rpk))
    env = {
        **os.environ,
        "STUB_DIR": str(stub_dir),
        "STUB_PASSWORD": "pw",
        "KAFKA_SASL_USERNAME": _SASL_USER,
        "KAFKA_SASL_PASSWORD": supplied,
    }
    if broken_sasl:
        env["STUB_BROKEN_SASL"] = "1"
    result = subprocess.run(
        ["sh", "-c", script], env=env, capture_output=True, text=True, check=False
    )
    for _ in range(rounds - 1):
        result = subprocess.run(
            ["sh", "-c", script], env=env, capture_output=True, text=True, check=False
        )
    return result, stub_dir


def test_local_sasl_one_shot_creates_the_user_then_flips_sasl_and_proves_refusal_sasl(
    tmp_path: Path,
) -> None:
    result, stub_dir = _run_sasl_script(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (stub_dir / "users").read_text().split() == [_SASL_USER]
    assert (stub_dir / "sasl-on").exists()
    assert not (stub_dir / "order-violation").exists()
    assert _SASL_USER in (stub_dir / "superusers").read_text()


def test_local_sasl_one_shot_is_idempotent_on_the_second_boot_sasl(
    tmp_path: Path,
) -> None:
    result, stub_dir = _run_sasl_script(tmp_path, rounds=2)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (stub_dir / "users").read_text().split() == [_SASL_USER], "created once"


def test_local_sasl_one_shot_fails_when_an_anonymous_client_is_accepted_sasl(
    tmp_path: Path,
) -> None:
    """Positive control: a broker that stays open must turn the one-shot red."""
    result, _ = _run_sasl_script(tmp_path, broken_sasl=True)
    assert result.returncode != 0
    assert "UNAUTHENTICATED client reached" in result.stderr


def test_local_sasl_one_shot_fails_when_the_superuser_password_is_wrong_sasl(
    tmp_path: Path,
) -> None:
    """A dead or mismatched principal must not read as a secured broker."""
    result, _ = _run_sasl_script(tmp_path, supplied="not-the-value")
    assert result.returncode != 0


_OTHER_BUNDLE_DIGESTS = {
    "auth": "1bf4de388a77a7d94e7a44cf785c7caba8a051c8aaf5501cc0b082cb5a25d28d",
    "canary": "caa5d10df59f3f5a596bb4f565fd2a466612a46a156d67a398b51ef8134cb53f",
    "core": "7c8f0182db15b6af0a70c6d05d6ffa7883db4725af31143b50fbdc83157a196b",
    "fault-injection": "79c02d5711737a44a78bc6212d5a6625a119a5232bfd670a42ff6e547996e371",
    "memgraph": "d55629512c84083550a018a068c3e394f352e06414192eff5a4f9f0967bd8bc3",
    "observability": "78d05c2b98b885da7867967c2189cc412cb2f1d408ac665a27170ec65fb1d936",
    "omnidash": "60e9cefe84b508bf5dc65281e7da56f1f338535d7501329161083bf76c1dbc36",
    "omnimarket-projections": "56e85c3eb64c72283ba5f7c98747520dadc79e95475272717ce5d6f2a0ac24ee",
    "omnimemory": "deff725fc3a399734a9052e09823d6f7e2b1c65e4935dd2b9e16d9b275294671",
    "runtime": "b9ecfd08f3d38127d16a98e603938c7e53c3b190ca1002cb493fa5d44be13926",
    "runtime-core": "450f161ad18bc1f73ad0e30eed2931207d544985f5579f6444d476a5636c809d",
    "runtime-infrastructure": "a2a0d96b0230f3fcb932f5f490c680c1bca16d6f7ebfeed8550585e2f45933ff",
    "runtime-integrations": "1b498251ede09da9d3eeb9481777b10831100d4d71b7e6092e201dea0a25e6fe",
    "runtime-observability-projections": "abdf89e68dfe87d77d24582604a514f9e90e186e264f64d0ebb78d1585a6a52e",
    "secrets": "7c8f0182db15b6af0a70c6d05d6ffa7883db4725af31143b50fbdc83157a196b",
    "tenant-path": "078b216879af2ccdedc8cc65c3ed4d6ee7a32e8f796b78dd8b2c6f63bf1232f0",
    "tracing": "78d05c2b98b885da7867967c2189cc412cb2f1d408ac665a27170ec65fb1d936",
}


def _digest(bundle: str) -> str:
    rendered = json.dumps(_render(bundle), sort_keys=True, default=str)
    return hashlib.sha256(rendered.encode()).hexdigest()


def test_every_bundle_but_local_renders_byte_identically_to_its_pre_sasl_digest_sasl() -> (
    None
):
    """OMN-20820 AC2. The digests were taken on dev 59e8bcb1e, before the change.

    A bundle whose render changes for a reason of its own updates its line here
    in the same change; the failure names the bundle.
    """
    bundles = yaml.safe_load((Path(_CATALOG_DIR) / "bundles.yaml").read_text())
    assert sorted(set(bundles) - {"local"}) == sorted(_OTHER_BUNDLE_DIGESTS)
    drifted = sorted(
        name
        for name, pinned in _OTHER_BUNDLE_DIGESTS.items()
        if _digest(name) != pinned
    )
    assert drifted == []


def test_the_sasl_one_shot_is_in_no_other_bundle_sasl() -> None:
    """Positive control for the digest test: the local render does change."""
    assert _SASL_SERVICE in _services(_render("local"))
    for name in _OTHER_BUNDLE_DIGESTS:
        assert _SASL_SERVICE not in _services(_render(name)), name


def _run_make_local_env(
    home: Path, *, env_file: Path, overlay: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", "local-env"],
        cwd=_REPO,
        env={**os.environ, "HOME": str(home)},
        capture_output=True,
        text=True,
        check=False,
    )


_PRE_SASL_ENV = (
    "# kept comment\n"
    "POSTGRES_PASSWORD=pg-kept\n"
    "VALKEY_PASSWORD=vk-kept\n"
    "OMNINODE_RUNTIME_PASSWORD=rt-kept\n"
    "TENANT_PROJECTION_WRITER_PASSWORD=tp-kept\n"
    "ONEX_TENANT_ID=\n"
    "ONEX_LOCAL_BIFROST_OVERLAY=/somewhere/overlay.yaml\n"
)


def test_local_env_adds_only_the_superuser_password_to_an_existing_0644_file_sasl(
    tmp_path: Path,
) -> None:
    """OMN-20820 AC3, in a temporary HOME so the default path is the one exercised."""
    home = tmp_path / "home"
    env_file = home / ".omnibase" / "local.env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(_PRE_SASL_ENV, encoding="utf-8")
    env_file.chmod(0o644)
    before = env_file.read_text().splitlines()

    result = _run_make_local_env(
        home, env_file=env_file, overlay=home / ".omnibase" / "local.bifrost.yaml"
    )
    assert result.returncode == 0, result.stdout + result.stderr

    after = env_file.read_text().splitlines()
    assert after[: len(before)] == before, "no existing line changed or moved"
    added = after[len(before) :]
    assert len(added) == 1, added
    name, _, value = added[0].partition("=")
    assert name == "REDPANDA_SUPERUSER_PASSWORD"
    assert re.fullmatch(r"[0-9a-f]{64}", value)
    assert env_file.stat().st_mode & 0o777 == 0o600
    assert value not in result.stdout + result.stderr, "the value is never printed"

    # Second run: nothing further to add, mode stays 0600.
    again = _run_make_local_env(
        home, env_file=env_file, overlay=home / ".omnibase" / "local.bifrost.yaml"
    )
    assert again.returncode == 0, again.stdout + again.stderr
    assert env_file.read_text().splitlines() == after


def test_local_env_fresh_file_carries_the_superuser_password_at_0600_sasl(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    result = _run_make_local_env(
        home,
        env_file=home / ".omnibase" / "local.env",
        overlay=home / ".omnibase" / "local.bifrost.yaml",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    env_file = home / ".omnibase" / "local.env"
    assert env_file.stat().st_mode & 0o777 == 0o600
    lines = [
        line
        for line in env_file.read_text().splitlines()
        if line.startswith("REDPANDA_SUPERUSER")
    ]
    assert len(lines) == 1
    assert re.fullmatch(r"REDPANDA_SUPERUSER_PASSWORD=[0-9a-f]{64}", lines[0])


_STUB_DOCKER = """#!/bin/sh
# Stub of `docker` for the laptop Makefile targets. STUB_MODE is secured|open.
[ "$1" = "info" ] && exit 0
[ "$1" = "inspect" ] && { echo healthy; exit 0; }
[ "$1" = "exec" ] || { echo "stub docker: unexpected $*" >&2; exit 2; }
shift
while [ "$1" = "-e" ]; do
  case "$2" in
    *=*) export "$2" ;;
    *) export "$2=$(printenv "$2")" ;;
  esac
  shift 2
done
shift  # the container name
echo "$* user=${RPK_USER:-}" >> "$STUB_CALLS"
case "$*" in
  "rpk cluster info"*|"rpk group list"*) ;;
  *) echo "stub docker: unexpected rpk $*" >&2; exit 2 ;;
esac
if [ "$STUB_MODE" = "open" ]; then
  echo "0  local.omnimarket.node_delegate_skill_orchestrator.consume  Stable"
  exit 0
fi
if [ -n "${RPK_USER:-}" ] && [ "${RPK_PASS:-}" = "$STUB_PASSWORD" ]; then
  echo "BROKER  GROUP  STATE"
  echo "0  local.omnimarket.node_delegate_skill_orchestrator.consume  Stable"
  exit 0
fi
echo "unable to request metadata: SASL is required but not provided" >&2
exit 1
"""


def _run_make_with_stub_docker(
    tmp_path: Path, target: str, *, mode: str, env_text: str | None
) -> tuple[subprocess.CompletedProcess[str], str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    docker = bin_dir / "docker"
    docker.write_text(_STUB_DOCKER, encoding="utf-8")
    docker.chmod(0o755)
    # status-local also reads the kernels' /health on host ports; no host port
    # is assumed to be free or open here.
    curl = bin_dir / "curl"
    curl.write_text('#!/bin/sh\necho \'{"status": "healthy"}\'\n', encoding="utf-8")
    curl.chmod(0o755)
    env_file = tmp_path / "local.env"
    if env_text is not None:
        env_file.write_text(env_text, encoding="utf-8")
    calls = tmp_path / "calls"
    calls.write_text("", encoding="utf-8")
    result = subprocess.run(
        ["make", target, f"LOCAL_ENV_FILE={env_file}"],
        cwd=_REPO,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
            "STUB_MODE": mode,
            "STUB_PASSWORD": "stub-secret",
            "STUB_CALLS": str(calls),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    return result, calls.read_text(encoding="utf-8")


_SECURED_ENV = "REDPANDA_SUPERUSER_PASSWORD=stub-secret\n"


def test_check_sasl_local_passes_on_a_secured_broker_sasl(tmp_path: Path) -> None:
    result, calls = _run_make_with_stub_docker(
        tmp_path, "check-sasl-local", mode="secured", env_text=_SECURED_ENV
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "stub-secret" not in result.stdout + result.stderr + calls
    # One anonymous probe and one authenticated probe, in that order.
    assert calls.splitlines() == [
        "rpk cluster info -X brokers=localhost:19092 user=",
        "rpk cluster info -X brokers=localhost:19092 user=onex-local",
    ]


def test_check_sasl_local_fails_on_a_broker_that_accepts_anonymous_clients_sasl(
    tmp_path: Path,
) -> None:
    """The pre-change stack: an open broker must turn the target red."""
    result, _ = _run_make_with_stub_docker(
        tmp_path, "check-sasl-local", mode="open", env_text=_SECURED_ENV
    )
    assert result.returncode != 0
    assert "anonymous client was accepted" in result.stderr


def test_check_sasl_local_fails_when_the_superuser_is_refused_sasl(
    tmp_path: Path,
) -> None:
    result, _ = _run_make_with_stub_docker(
        tmp_path,
        "check-sasl-local",
        mode="secured",
        env_text="REDPANDA_SUPERUSER_PASSWORD=some-other-value\n",
    )
    assert result.returncode != 0
    assert "was refused" in result.stderr


@pytest.mark.parametrize("env_text", [None, "POSTGRES_PASSWORD=x\n"])
def test_check_sasl_local_fails_without_the_superuser_password_sasl(
    tmp_path: Path, env_text: str | None
) -> None:
    result, calls = _run_make_with_stub_docker(
        tmp_path, "check-sasl-local", mode="secured", env_text=env_text
    )
    assert result.returncode != 0
    assert calls == ""


def test_status_local_group_listing_authenticates_as_the_local_superuser_sasl(
    tmp_path: Path,
) -> None:
    """The group listing is the part of status-local the SASL broker would refuse."""
    result, calls = _run_make_with_stub_docker(
        tmp_path, "status-local", mode="secured", env_text=_SECURED_ENV
    )
    # The migration-gate inspect and the /health curls are not under test; the
    # listing is, and it must carry the credentials and print the delegate group.
    assert "rpk group list user=onex-local" in calls
    assert "node_delegate_skill_orchestrator" in result.stdout
    assert "stub-secret" not in result.stdout + result.stderr + calls
