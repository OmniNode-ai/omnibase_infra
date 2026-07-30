# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Credential and network-boundary invariants for the edge gateway compose."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parents[3]
GATEWAY_COMPOSE = REPO_ROOT / "docker" / "docker-compose.gateway.yml"
GATEWAY_SYSTEMD_UNIT = (
    REPO_ROOT / "docker" / "gateway" / "onex-gateway-forwarder.service"
)


def _gateway_service() -> dict[str, object]:
    compose = yaml.safe_load(GATEWAY_COMPOSE.read_text(encoding="utf-8"))
    return compose["services"]["gateway-forwarder"]


def test_gateway_uses_roles_anywhere_instead_of_static_access_keys() -> None:
    """The permanent edge must consume only short-lived AWS credentials."""
    service = _gateway_service()
    environment = service["environment"]
    volumes = service["volumes"]
    devices = service["devices"]
    raw_compose = GATEWAY_COMPOSE.read_text(encoding="utf-8")

    assert environment["AWS_CONFIG_FILE"] == "/run/aws/config"
    assert environment["AWS_SDK_LOAD_CONFIG"] == "1"
    assert "AWS_SHARED_CREDENTIALS_FILE" not in environment
    assert "GATEWAY_AWS_CREDENTIALS_FILE" not in raw_compose
    assert not any("AWS_ACCESS_KEY_ID" in value for value in environment)
    assert not any("AWS_SECRET_ACCESS_KEY" in value for value in environment)
    assert any(":/run/aws/config:ro" in mount for mount in volumes)
    assert any(":/run/aws/certificate.pem:ro" in mount for mount in volumes)
    assert any(":/run/aws/private-key.tss:ro" in mount for mount in volumes)
    assert any(":/usr/local/bin/aws_signing_helper:ro" in mount for mount in volumes)
    assert devices == [
        "${GATEWAY_TPM_DEVICE:?GATEWAY_TPM_DEVICE is required}:/dev/tpmrm0"
    ]
    assert service["user"] == (
        "${GATEWAY_CONTAINER_UID:?GATEWAY_CONTAINER_UID is required}:"
        "${GATEWAY_CONTAINER_GID:?GATEWAY_CONTAINER_GID is required}"
    )
    assert service["group_add"] == [
        "${GATEWAY_TPM_GROUP_ID:?GATEWAY_TPM_GROUP_ID is required}"
    ]


def test_gateway_has_no_inbound_application_port() -> None:
    """The hybrid edge remains outbound-only after credential hardening."""
    service = _gateway_service()

    assert "ports" not in service
    assert service["image"] == "${GATEWAY_IMAGE:?GATEWAY_IMAGE is required}"
    assert service["pull_policy"] == "never"
    assert service["restart"] == "unless-stopped"


def test_gateway_systemd_unit_waits_for_container_health() -> None:
    """The host supervisor must fail if the gateway never becomes healthy."""
    unit = GATEWAY_SYSTEMD_UNIT.read_text(encoding="utf-8")

    assert "After=network-online.target docker.service tailscaled.service" in unit
    assert "ExecStartPre=/usr/bin/docker image inspect ${GATEWAY_IMAGE}" in unit
    assert "up -d --no-build --wait --wait-timeout 120 gateway-forwarder" in unit
    assert "WantedBy=multi-user.target" in unit
