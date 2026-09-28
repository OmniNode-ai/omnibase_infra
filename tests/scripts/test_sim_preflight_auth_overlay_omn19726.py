# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Rendered, no-launch checks for the disposable auth/Gateway candidate."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILES = (
    ROOT / "docker/docker-compose.dogfood.yml",
    ROOT / "docker/docker-compose.sim-202.yml",
    ROOT / "docker/docker-compose.sim-preflight.yml",
    ROOT / "docker/docker-compose.sim-preflight-auth.yml",
)
AUTH = COMPOSE_FILES[-1]
REQUIRED_REF = re.compile(r"\$\{([A-Z][A-Z0-9_]*):\?")


def _render(
    *, omit: str | None = None, isolated: bool = False
) -> subprocess.CompletedProcess[str]:
    assert shutil.which("docker") is not None
    # No inherited credentials, Compose project override, or active Docker
    # context: config is a local, read-only merge of disposable test values.
    env = {"PATH": os.environ["PATH"]}
    compose_files = COMPOSE_FILES + (
        (
            ROOT / "docker/docker-compose.sim-preflight-graph.yml",
            ROOT / "docker/docker-compose.sim-preflight-isolated.yml",
        )
        if isolated
        else ()
    )
    required = {
        match
        for path in compose_files
        for match in REQUIRED_REF.findall(path.read_text(encoding="utf-8"))
    }
    env.update(dict.fromkeys(required, "test-only-placeholder"))
    if isolated:
        env.update(
            {
                name: "/private/test-only/" + name
                for name in required
                if name.endswith("_FILE")
            }
        )
    env.update(
        {
            "SIM_202_RUNTIME_IMAGE": "example.invalid/runtime@sha256:" + "a" * 64,
            "SIM_PREFLIGHT_ONEX_API_IMAGE": "example.invalid/onex-api@sha256:"
            + "b" * 64,
            "SIM_PREFLIGHT_CLOUD_MIGRATE_IMAGE": "example.invalid/cloud-migrate@sha256:"
            + "c" * 64,
            "OMNICLAUDE_SKILLS_DIR": str(ROOT / "tests"),
            "DOGFOOD_RUNTIME_MAIN_PORT": "8085",
            "DOGFOOD_RUNTIME_EFFECTS_PORT": "8086",
            "OMNIMEMORY_MEMGRAPH_PORT": "7687",
            "DOGFOOD_TOPIC_PROVISIONER_MAX_PARTITIONS": "1",
            "DOGFOOD_RUNTIME_MAIN_BIFROST_VERIFY_ENDPOINTS": "0",
            "DOGFOOD_RUNTIME_EFFECTS_BIFROST_VERIFY_ENDPOINTS": "0",
            "DOGFOOD_RUNTIME_MAIN_OMNIMEMORY_ENABLED": "false",
            "DOGFOOD_RUNTIME_EFFECTS_OMNIMEMORY_ENABLED": "false",
        }
    )
    if omit:
        env.pop(omit, None)
    cmd = [
        "docker",
        "compose",
        "-p",
        "omnibase-infra-sim-preflight",
        "--env-file",
        str(ROOT / "docker/runtime-policy.env"),
    ]
    for path in compose_files:
        cmd.extend(("-f", str(path)))
    cmd.extend(
        (
            "--profile",
            "dogfood",
            "--profile",
            "sim-preflight-auth",
            "config",
            "--format",
            "json",
        )
    )
    return subprocess.run(
        cmd, cwd=ROOT, env=env, text=True, capture_output=True, check=False, timeout=20
    )


@pytest.mark.unit
def test_isolated_overlay_passes_network_boundary_and_removes_business_surfaces() -> (
    None
):
    from scripts.runtime_build.verify_sim_preflight_isolation import verify

    result = _render(isolated=True)
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    assert verify(config) == {"isolated": True, "service_count": 13}
    assert "projection-api" not in config["services"]
    for name in ("omninode-runtime", "runtime-effects"):
        runtime = config["services"][name]
        targets = {mount["target"] for mount in runtime["volumes"]}
        assert "/app/contracts" not in targets
        assert "/app/skills" not in targets
        assert runtime["environment"]["BUS_ID"] == "sim-preflight"
        assert runtime["environment"]["OTEL_SDK_DISABLED"] == "true"
        assert runtime["environment"]["BIFROST_CONTRACT_PATH"] == ""


@pytest.mark.unit
def test_auth_overlay_renders_isolated_single_issuer_and_migration_chain() -> None:
    result = _render()
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    assert config["name"] == "omnibase-infra-sim-preflight"
    services = config["services"]
    keycloak = services["keycloak"]
    gateway = services["onex-api"]
    cloud = services["cloud-migration"]
    files = services["cloud-migration-files"]
    assert keycloak["image"] == "quay.io/keycloak/keycloak:26.3.3"
    assert keycloak["environment"]["KC_HOSTNAME"] == "http://auth.localhost:28080"
    assert keycloak["environment"]["KC_HTTP_PORT"] == "28080"
    assert keycloak["environment"]["KC_HOSTNAME_BACKCHANNEL_DYNAMIC"] == "true"
    assert (
        "auth.localhost"
        in keycloak["networks"]["omnibase-infra-dogfood-network"]["aliases"]
    )
    assert keycloak["ports"] == [
        {
            "mode": "ingress",
            "target": 28080,
            "published": "28080",
            "protocol": "tcp",
            "host_ip": "127.0.0.1",
        }
    ]
    assert gateway["ports"][0]["host_ip"] == "127.0.0.1"
    assert gateway["ports"][0]["published"] == "8090"
    assert (
        gateway["environment"]["KEYCLOAK_ISSUER_URL"]
        == "http://auth.localhost:28080/realms/omninode"
    )
    assert gateway["environment"]["KEYCLOAK_AUDIENCE"] == "onex-api"
    assert gateway["environment"]["GATEWAY_P0B_PER_TENANT_CREDENTIALS"] == "disabled"
    assert gateway["environment"]["GATEWAY_ATTACH_TOKEN_EXCHANGE"] == "disabled"
    for name in (
        "STRIPE_CHECKOUT_SUCCESS_URL",
        "STRIPE_CHECKOUT_CANCEL_URL",
        "STRIPE_PORTAL_RETURN_URL",
    ):
        assert gateway["environment"][name] == "test-only-placeholder"
    assert "build" not in gateway
    assert gateway["image"].endswith("b" * 64)
    assert files["image"].endswith("c" * 64)
    assert (
        cloud["depends_on"]["cloud-migration-files"]["condition"]
        == "service_completed_successfully"
    )
    assert (
        cloud["depends_on"]["forward-migration"]["condition"]
        == "service_completed_successfully"
    )
    assert (
        gateway["depends_on"]["cloud-migration"]["condition"]
        == "service_completed_successfully"
    )
    assert cloud["environment"]["DB_USER"] == "role_omninode"
    assert cloud["environment"]["DB_NAME"] == "omninode_cloud"
    assert (
        services["postgres"]["environment"]["ROLE_OMNINODE_PASSWORD"]
        == "test-only-placeholder"
    )
    assert (
        services["forward-migration"]["environment"]["ROLE_OMNINODE_PASSWORD"]
        == "test-only-placeholder"
    )
    assert (
        config["networks"]["omnibase-infra-dogfood-network"]["name"]
        == "omnibase-infra-sim-preflight-network"
    )
    assert (
        config["volumes"]["sim_preflight_cloud_migrations"]["name"]
        == "omnibase-infra-sim-preflight-cloud-migrations"
    )
    assert all(
        port["host_ip"] == "127.0.0.1" for port in keycloak["ports"] + gateway["ports"]
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "missing",
    [
        "SIM_PREFLIGHT_CLOUD_ROLE_PASSWORD",
        "SIM_PREFLIGHT_KEYCLOAK_ADMIN_USERNAME",
        "SIM_PREFLIGHT_KEYCLOAK_ADMIN_PASSWORD",
        "SIM_PREFLIGHT_ONEX_API_IMAGE",
        "SIM_PREFLIGHT_CLOUD_MIGRATE_IMAGE",
        "SIM_PREFLIGHT_STRIPE_API_KEY",
        "SIM_PREFLIGHT_STRIPE_CHECKOUT_SUCCESS_URL",
        "SIM_PREFLIGHT_STRIPE_CHECKOUT_CANCEL_URL",
        "SIM_PREFLIGHT_STRIPE_PORTAL_RETURN_URL",
    ],
)
def test_auth_overlay_refuses_missing_required_inputs(missing: str) -> None:
    result = _render(omit=missing)
    assert result.returncode != 0
    assert missing in result.stderr
    assert "test-only-placeholder" not in result.stderr


@pytest.mark.unit
def test_overlay_keeps_gateway_catalog_fenced_and_has_no_lifecycle_command() -> None:
    raw = AUTH.read_text(encoding="utf-8")
    assert "workflow-contracts.yaml" not in raw
    assert "seed-keycloak" not in raw
    assert "docker compose up" not in raw
    assert "docker compose down" not in raw
    assert "sim-preflight-auth" in raw


@pytest.mark.unit
def test_existing_forward_corpus_creates_keycloak_and_cloud_role() -> None:
    keycloak_migration = (
        ROOT / "docker/migrations/forward/042_create_keycloak_db.sql"
    ).read_text(encoding="utf-8")
    database_bootstrap = (
        ROOT / "docker/migrations/forward/000_create_multiple_databases.sh"
    ).read_text(encoding="utf-8")
    forward_runner = (ROOT / "scripts/run-forward-migrations.sh").read_text(
        encoding="utf-8"
    )
    assert "-- onex-create-database: keycloak" in keycloak_migration
    assert '"omninode_cloud:role_omninode:ROLE_OMNINODE_PASSWORD"' in database_bootstrap
    assert "onex-create-database" in forward_runner
    assert "ROLE_OMNINODE_PASSWORD" in forward_runner


@pytest.mark.unit
def test_gateway_credentials_use_disposable_input_names_only() -> None:
    raw = AUTH.read_text(encoding="utf-8")
    for name in (
        "ROLE_OMNINODE_PASSWORD",
        "KEYCLOAK_ADMIN_USERNAME",
        "KEYCLOAK_ADMIN_PASSWORD",
        "STRIPE_API_KEY",
        "STRIPE_WEBHOOK_SECRET",
        "STRIPE_CHECKOUT_SUCCESS_URL",
        "STRIPE_CHECKOUT_CANCEL_URL",
        "STRIPE_PORTAL_RETURN_URL",
        "TENANT_BOOTSTRAP_ADMIN_SECRET",
        "TENANT_TOPICS_ADMIN_SECRET",
        "TENANT_CLIENTS_ADMIN_SECRET",
        "TENANT_OFFBOARD_ADMIN_SECRET",
        "ALPHA_INVITE_ADMIN_SECRET",
    ):
        assert f"${{{name}" not in raw
    assert "${SIM_PREFLIGHT_CLOUD_ROLE_PASSWORD:?" in raw
    assert "${SIM_PREFLIGHT_KEYCLOAK_ADMIN_PASSWORD:?" in raw
    for name in (
        "SIM_PREFLIGHT_STRIPE_CHECKOUT_SUCCESS_URL",
        "SIM_PREFLIGHT_STRIPE_CHECKOUT_CANCEL_URL",
        "SIM_PREFLIGHT_STRIPE_PORTAL_RETURN_URL",
    ):
        assert f"${{{name}:?" in raw
