# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Emit only a strict non-secret summary of a sim-preflight Compose render."""

from __future__ import annotations

import json
import sys
from typing import Any, cast

_PROJECT = "omnibase-infra-sim-preflight"
_SERVICES = frozenset(
    {
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
)
_CONTAINERS = frozenset(
    {
        "omnibase-infra-sim-preflight-postgres",
        "omnibase-infra-sim-preflight-redpanda",
        "omnibase-infra-sim-preflight-redpanda-partition-cap",
        "omnibase-infra-sim-preflight-valkey",
        "omnibase-infra-sim-preflight-forward-migration",
        "omnibase-infra-sim-preflight-migration-gate",
        "omnibase-infra-sim-preflight-intelligence-migration",
        "omninode-sim-preflight-runtime",
        "omninode-sim-preflight-runtime-effects",
        "omnimarket-sim-preflight-projection-api",
    }
)
_NETWORKS = frozenset({"omnibase-infra-sim-preflight-network"})
_VOLUMES = frozenset(
    {
        "omnibase-infra-sim-preflight-postgres-data",
        "omnibase-infra-sim-preflight-redpanda-data",
        "omnibase-infra-sim-preflight-valkey-data",
        "omninode-sim-preflight-runtime-logs",
        "omninode-sim-preflight-runtime-data",
        "omninode-sim-preflight-effects-logs",
        "omninode-sim-preflight-effects-data",
    }
)
_PORTS = frozenset({"65036", "65092", "65444", "65379", "65085", "65086", "65002"})
_RUNTIME_SERVICES = frozenset({"omninode-runtime", "runtime-effects", "projection-api"})
_EXPECTED_BIND_MOUNT_COUNT = 8
_CREDENTIAL_KEYS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    "LLM_GLM_API_KEY",
    "LOCAL_LLM_SHARED_SECRET",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "LINEAR_API_KEY",
)


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"compose {label} must be an object")
    return cast("dict[str, Any]", value)


def _resource_names(config: dict[str, Any], key: str) -> list[str]:
    resources = _mapping(config.get(key), label=key)
    names: list[str] = []
    for resource_key, value in resources.items():
        if not isinstance(value, dict) or not isinstance(value.get("name"), str):
            raise ValueError(f"compose {key}.{resource_key} must name a resource")
        names.append(value["name"])
    return sorted(names)


def main() -> int:
    config = _mapping(json.load(sys.stdin), label="root")
    services = _mapping(config.get("services"), label="services")
    container_names: list[str] = []
    published_ports: list[dict[str, str | int]] = []
    credential_fields_blank: dict[str, bool] = {}
    all_bind_mounts_read_only = True
    bind_mount_count = 0
    db_hosts_are_postgres = True
    for service_name, service in services.items():
        if not isinstance(service, dict):
            raise ValueError(f"compose services.{service_name} must be an object")
        container_name = service.get("container_name")
        if not isinstance(container_name, str):
            raise ValueError(f"compose services.{service_name} lacks container name")
        container_names.append(container_name)
        if service_name in _RUNTIME_SERVICES:
            environment = _mapping(service.get("environment"), label="environment")
            credential_fields_blank[service_name] = all(
                environment.get(key) == "" for key in _CREDENTIAL_KEYS
            )
            db_hosts_are_postgres = db_hosts_are_postgres and all(
                "@postgres:5432/" in str(environment.get(key, ""))
                for key in ("OMNIBASE_INFRA_DB_URL", "OMNIDASH_ANALYTICS_DB_URL")
            )
            mounts = service.get("volumes", [])
            if not isinstance(mounts, list):
                raise ValueError(
                    f"compose services.{service_name}.volumes must be a list"
                )
            for mount in mounts:
                if not isinstance(mount, dict):
                    raise ValueError(
                        f"compose services.{service_name}.volumes entry invalid"
                    )
                if mount.get("type") == "bind":
                    bind_mount_count += 1
                    all_bind_mounts_read_only = all_bind_mounts_read_only and (
                        mount.get("read_only") is True
                    )
        ports = service.get("ports", [])
        if isinstance(ports, list):
            for port in ports:
                if not isinstance(port, dict):
                    raise ValueError(
                        f"compose services.{service_name}.ports entry invalid"
                    )
                host_ip = port.get("host_ip")
                published = port.get("published")
                if isinstance(host_ip, str) and isinstance(published, (str, int)):
                    published_ports.append({"host_ip": host_ip, "published": published})
                else:
                    raise ValueError(
                        f"compose services.{service_name}.ports entry lacks host metadata"
                    )
        else:
            raise ValueError(f"compose services.{service_name}.ports must be a list")
    if config.get("name") != _PROJECT:
        raise ValueError("compose project does not match sim-preflight")
    if set(services) != _SERVICES or set(container_names) != _CONTAINERS:
        raise ValueError(
            "compose services or containers do not match sim-preflight allowlist"
        )
    if set(_resource_names(config, "networks")) != _NETWORKS:
        raise ValueError("compose networks do not match sim-preflight allowlist")
    if set(_resource_names(config, "volumes")) != _VOLUMES:
        raise ValueError("compose volumes do not match sim-preflight allowlist")
    if (
        len(published_ports) != len(_PORTS)
        or {str(port["published"]) for port in published_ports} != _PORTS
        or any(port["host_ip"] != "127.0.0.1" for port in published_ports)
    ):
        raise ValueError("compose published ports do not match sim-preflight allowlist")
    if set(credential_fields_blank) != _RUNTIME_SERVICES:
        raise ValueError("compose runtime service set does not match credential check")
    if not all(credential_fields_blank.values()):
        raise ValueError("compose runtime credentials are not blank")
    if not db_hosts_are_postgres:
        raise ValueError("compose runtime database hosts are not postgres")
    if not all_bind_mounts_read_only:
        raise ValueError("compose bind mounts are not read-only")
    if bind_mount_count != _EXPECTED_BIND_MOUNT_COUNT:
        raise ValueError("compose bind mount count does not match sim-preflight")
    print(
        json.dumps(
            {
                "all_bind_mounts_read_only": all_bind_mounts_read_only,
                "bind_mount_count": bind_mount_count,
                "credential_fields_blank": all(credential_fields_blank.values()),
                "container_count": len(container_names),
                "db_hosts_are_postgres": db_hosts_are_postgres,
                "network_count": len(_NETWORKS),
                "port_count": len(published_ports),
                "project_is_expected": True,
                "service_count": len(services),
                "volume_count": len(_VOLUMES),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
