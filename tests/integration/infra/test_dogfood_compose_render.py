# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Non-mutating compose render checks for the OMN-18693 dogfood lane.

The checks prove that the standalone configuration selects the dogfood services,
uses private resources, and renders with its required operator inputs. Nothing
here starts a container; ``docker compose config`` is a pure render.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent.parent
COMPOSE_FILE = "docker/docker-compose.dogfood.yml"
ENV_EXAMPLE = "docker/dogfood.env.example"
DOGFOOD_NETWORK = "omnibase-infra-dogfood-network"
COMPOSE_PROJECT = "omnibase-infra-dogfood"

#: Render-only stand-ins for the credentials above.
#:
#: These live only in the test. They exercise compose interpolation without
#: putting an operator value in a tracked configuration example.
RENDER_ONLY_SECRETS = {
    "POSTGRES_PASSWORD": "render-only-not-a-secret",
    "VALKEY_PASSWORD": "render-only-not-a-secret",
    "OMNINODE_RUNTIME_PASSWORD": "render-only-not-a-secret",
    "TENANT_PROJECTION_WRITER_PASSWORD": "render-only-not-a-secret",
    "DOGFOOD_REDPANDA_ADVERTISED_HOST": "192.168.86.201",
    "OMNICLAUDE_SKILLS_DIR": str(
        (REPO_ROOT.parent / "omnimarket/plugins/onex/skills").resolve()
    ),
}

#: The operator input is an absolute source path so staging the compose file
#: under an isolated deployment root cannot change the bind-mount target.
EXPECTED_SKILLS_SOURCE = (REPO_ROOT.parent / "omnimarket/plugins/onex/skills").resolve()
SKILLS_MOUNT_TARGET = "/app/skills"

EXPECTED_RENDERED_SERVICES = {
    "postgres",
    "redpanda",
    "redpanda-partition-cap",
    "valkey",
    "forward-migration",
    "migration-gate",
    "intelligence-migration",
    "omninode-runtime",
    "runtime-effects",
    "projection-api",
}
#: Services that must never appear on this lane. ``keycloak`` and ``infisical``
#: are the OMN-13581 cross-lane displacement risk (they carry no profile in the
#: base infra compose, so a layered render would start them under THIS project
#: with DEV names on the DEV network); the rest belong to other lanes.
OUT_OF_SCOPE_SERVICES = {
    "keycloak",
    "infisical",
    "runtime-worker",
    "agent-actions-consumer",
    "skill-lifecycle-consumer",
    "context-audit-consumer",
    "intelligence-api",
    "omninode-contract-resolver",
    "phoenix",
    "autoheal",
}
#: The OMN-18693 reserved block, and only it.
EXPECTED_PUBLISHED_PORTS = {
    "postgres": {"49436"},
    "redpanda": {"47092", "47644"},
    "redpanda-partition-cap": set(),
    "valkey": {"49379"},
    "forward-migration": set(),
    "migration-gate": set(),
    "intelligence-migration": set(),
    "omninode-runtime": {"49085"},
    "runtime-effects": {"49086"},
    "projection-api": {"49302"},
}
#: Ports belonging to the four lanes that predate this one. Publishing any of
#: them here would displace a live lane on the shared .201 host (OMN-13581).
OTHER_LANE_PUBLISHED_PORTS = {
    "5436",
    "15436",
    "35436",
    "19092",
    "39092",
    "49092",
    "9644",
    "29644",
    "49644",
    "16379",
    "26379",
    "56379",
    "8085",
    "18085",
    "28085",
    "48085",
    "8086",
    "18086",
    "28086",
    "48086",
    "3002",
    "13002",
    "23002",
    "43002",
    "28080",
    "38080",
    "8881",
}


def _docker_compose_available() -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(
        ["docker", "compose", "version"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _compose_render_env() -> dict[str, str]:
    """A deliberately minimal environment.

    The ambient shell must not be able to satisfy an interpolation the compose
    file is supposed to demand, or the render would prove nothing about a fresh
    clone — which is the state the lane owner is actually in.
    """
    python_path = os.pathsep.join(
        path
        for path in (
            str(REPO_ROOT / "src"),
            str(REPO_ROOT),
            os.environ.get("PYTHONPATH", ""),
        )
        if path
    )
    return {
        "HOME": os.environ.get("HOME", ""),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": python_path,
        "USER": os.environ.get("USER", ""),
        # Explicit render-only stand-ins, not inherited from the ambient shell.
        # The tracked example keeps guarded values empty, so this fixture
        # supplies non-secret values only for offline compose rendering.
        **RENDER_ONLY_SECRETS,
    }


def _docker_compose_command(*args: str) -> list[str]:
    return [
        "docker",
        "compose",
        "-p",
        COMPOSE_PROJECT,
        "--env-file",
        "docker/runtime-policy.env",
        "--env-file",
        ENV_EXAMPLE,
        "-f",
        COMPOSE_FILE,
        "--profile",
        "dogfood",
        *args,
    ]


def _run_compose_config(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _docker_compose_command("config", *args),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        env=_compose_render_env(),
        text=True,
    )


def _compose_config_json() -> dict[str, Any]:
    rendered_config = json.loads(_run_compose_config("--format", "json").stdout)
    assert isinstance(rendered_config, dict)
    return cast("dict[str, Any]", rendered_config)


def _published_ports(service_config: dict[str, Any]) -> set[str]:
    ports = cast("list[dict[str, Any]]", service_config.get("ports", []))
    return {str(port["published"]) for port in ports}


pytestmark = pytest.mark.skipif(
    not _docker_compose_available(),
    reason="docker compose is required for non-mutating compose render validation",
)


@pytest.mark.integration
def test_dogfood_profile_selects_the_lane_services() -> None:
    """THE regression this lane was blocked on: the profile must select services.

    ``--profile dogfood`` previously matched nothing, because no file declared
    the profile. A zero-service match is silent — ``up -d`` exits 0 and starts
    nothing — so this asserts the exact set rather than merely "non-empty".
    """
    rendered_services = set(_run_compose_config("--services").stdout.splitlines())

    assert rendered_services == EXPECTED_RENDERED_SERVICES
    assert rendered_services.isdisjoint(OUT_OF_SCOPE_SERVICES)


@pytest.mark.integration
def test_dogfood_lane_publishes_only_the_reserved_port_block() -> None:
    """Every published host port is from the OMN-18693 reservation, and no other
    lane's port is published — the OMN-13581 displacement class, checked."""
    services = _compose_config_json()["services"]

    observed = {name: _published_ports(cfg) for name, cfg in services.items()}
    assert observed == EXPECTED_PUBLISHED_PORTS

    all_published = {port for ports in observed.values() for port in ports}
    collisions = all_published & OTHER_LANE_PUBLISHED_PORTS
    assert not collisions, (
        f"the dogfood lane publishes port(s) {sorted(collisions)} that "
        "belong to another .201 lane; on a shared host that displaces the live "
        "lane (OMN-13581)"
    )


@pytest.mark.integration
def test_dogfood_lane_render_is_isolated_from_every_other_lane() -> None:
    """Project, network, container names and volumes are all lane-scoped.

    Volume namespacing is what makes a teardown-with-volumes a safe reset for
    this lane's owner: it must wipe this lane's state and nothing else on the
    host.
    """
    rendered_config = _compose_config_json()

    assert rendered_config["name"] == COMPOSE_PROJECT
    assert set(rendered_config["networks"]) == {DOGFOOD_NETWORK}

    for name, service in rendered_config["services"].items():
        assert set(service.get("networks", {})) == {DOGFOOD_NETWORK}, (
            f"service {name!r} is attached to a network other than the lane's own"
        )
        assert "dogfood" in service["container_name"], (
            f"service {name!r} renders container_name "
            f"{service['container_name']!r}, which does not name this lane and "
            "could collide with another lane's container"
        )

    volume_names = {v["name"] for v in rendered_config["volumes"].values()}
    assert volume_names
    for volume_name in volume_names:
        assert "dogfood" in volume_name, (
            f"volume {volume_name!r} is not lane-scoped, so a profile-scoped "
            "teardown with volume removal would wipe another lane's data"
        )


@pytest.mark.integration
def test_dogfood_lane_render_carries_its_own_runtime_identity() -> None:
    rendered_config = _compose_config_json()
    services = rendered_config["services"]

    assert services["omninode-runtime"]["container_name"] == "omninode-dogfood-runtime"
    assert services["runtime-effects"]["container_name"] == (
        "omninode-dogfood-runtime-effects"
    )
    assert services["projection-api"]["container_name"] == (
        "omnimarket-dogfood-projection-api"
    )

    for service_name in ("omninode-runtime", "runtime-effects"):
        environment = services[service_name]["environment"]
        assert environment["ONEX_ENVIRONMENT"] == "dogfood"
        assert environment["KAFKA_ENVIRONMENT"] == "dogfood"
        assert environment["ONEX_DATABASE_TOPOLOGY_PROFILE"] == "dogfood"
        assert environment["BUS_ID"] == "dogfood"
        assert environment["ONEX_STATE_ROOT"] == "/app/data/.onex_state_dogfood"
        assert environment["ONEX_STATE_DIR"] == "/app/data/.onex_state_dogfood"
        assert environment["KAFKA_INSTANCE_ID"].startswith("dogfood-")
        assert environment["ONEX_RUNTIME_ADDRESS"].startswith(
            "runtime://omninode-pc/dogfood/"
        )
        assert environment["ONEX_RUNTIME_ID"].startswith("dogfood-")
        assert environment["ONEX_SECRET_RESOLVER_CONFIG_PATH"] == (
            "/app/data/delegation/secret_resolver.yaml"
        )
        resolver_config = json.loads(environment["ONEX_SECRET_RESOLVER_CONFIG_JSON"])
        assert resolver_config["enable_convention_fallback"] is False

    assert services["omninode-runtime"]["environment"]["ONEX_GROUP_ID"] == (
        "onex-dogfood-runtime-main"
    )
    assert services["runtime-effects"]["environment"]["ONEX_GROUP_ID"] == (
        "onex-dogfood-runtime-effects"
    )


@pytest.mark.integration
def test_dogfood_lane_renders_no_dangling_credential_broker() -> None:
    """OMN-12966 / OMN-13037: this lane runs neither Keycloak nor Infisical, so
    no service may carry an address for either. A blank address fails visibly;
    another lane's address fails silently by succeeding against the wrong lane.
    """
    services = _compose_config_json()["services"]

    assert set(services).isdisjoint(OUT_OF_SCOPE_SERVICES)
    for name, service in services.items():
        depends_on = service.get("depends_on", {})
        assert "keycloak" not in depends_on, name
        assert "infisical" not in depends_on, name
        environment = service.get("environment", {})
        assert environment.get("INFISICAL_ADDR", "") == "", name
        assert environment.get("KEYCLOAK_ISSUER", "") == "", name


@pytest.mark.integration
def test_dogfood_lane_boots_without_any_cloud_model_credential() -> None:
    """Correction (b): a mutable sandbox must render with zero cloud secrets.

    The judge lane hard-requires GEMINI_API_KEY/GOOGLE_API_KEY (``:?``) because
    it exists to reproduce a cloud-routed judgement. This lane must reach a
    healthy ``/health`` with none of them set — wiring a model backend is a
    separate, later step. If any of these became fail-closed, the render below
    would abort and the lane owner would be blocked on a credential they do not
    need yet.
    """
    environment = _compose_config_json()["services"]["omninode-runtime"]["environment"]

    for optional_ref in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
        "LLM_GLM_API_KEY",
    ):
        assert environment.get(optional_ref, "") == "", (
            f"{optional_ref} rendered non-empty from the example env file; the "
            "example must never carry a real credential"
        )


@pytest.mark.integration
def test_dogfood_lane_infrastructure_credentials_stay_fail_closed() -> None:
    """Correction (d): mutable is not the same as sloppy.

    Model credentials are optional; the database and cache passwords are not. A
    lane that boots with a blank Postgres password is a worse outcome than one
    that refuses to boot, so the ``:?`` guards must survive.
    """
    raw = (REPO_ROOT / COMPOSE_FILE).read_text(encoding="utf-8")

    for required_ref in (
        "POSTGRES_PASSWORD:?",
        "VALKEY_PASSWORD:?",
        "OMNINODE_RUNTIME_PASSWORD:?",
        "TENANT_PROJECTION_WRITER_PASSWORD:?",
        "OMNICLAUDE_SKILLS_DIR:?",
    ):
        assert required_ref in raw, (
            f"{required_ref!r} lost its fail-closed guard; the lane would boot "
            "with a blank credential or an empty skills mount"
        )


@pytest.mark.integration
def test_skills_mount_resolves_to_the_sibling_omnimarket_checkout() -> None:
    """The /app/skills bind mount must use the absolute operator source path."""
    services = _compose_config_json()["services"]
    mounted_by = {
        name: volume["source"]
        for name, config in services.items()
        for volume in config.get("volumes", [])
        if volume.get("target") == SKILLS_MOUNT_TARGET
    }
    assert mounted_by, (
        f"No service mounts {SKILLS_MOUNT_TARGET}. The runtime services need the "
        "skills tree; a lane that mounts nothing there is silently skill-less."
    )
    wrong = {
        name: source
        for name, source in mounted_by.items()
        if Path(source).resolve() != EXPECTED_SKILLS_SOURCE
    }
    assert not wrong, (
        f"These services resolve {SKILLS_MOUNT_TARGET} to the wrong host path: "
        f"{wrong}. Expected the sibling checkout {EXPECTED_SKILLS_SOURCE}. "
        "Compose resolves a relative source against docker/, so a sibling "
        "checkout needs TWO levels up (../../omnimarket/...), not one."
    )
