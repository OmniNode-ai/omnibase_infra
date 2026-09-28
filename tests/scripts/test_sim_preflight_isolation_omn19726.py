# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""No-launch tests for the approved graph-only disposable network boundary."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from scripts.runtime_build.verify_sim_preflight_isolation import verify


def _config() -> dict[str, Any]:
    names = (
        "postgres",
        "redpanda",
        "redpanda-partition-cap",
        "valkey",
        "forward-migration",
        "migration-gate",
        "intelligence-migration",
        "omninode-runtime",
        "runtime-effects",
        "keycloak",
        "cloud-migration-files",
        "cloud-migration",
        "onex-api",
    )
    services = {
        name: {
            "container_name": (
                "omninode-sim-preflight-runtime"
                if name == "omninode-runtime"
                else "omninode-sim-preflight-runtime-effects"
                if name == "runtime-effects"
                else f"omnibase-infra-sim-preflight-{name}"
            ),
            "networks": {"omnibase-infra-dogfood-network": {}},
            "dns": ["127.0.0.1"],
            "environment": {},
        }
        for name in names
    }
    for name in ("omninode-runtime", "runtime-effects"):
        services[name].update(
            {
                "cap_drop": ["ALL"],
                "cap_add": ["CHOWN", "DAC_OVERRIDE", "FOWNER", "SETGID", "SETUID"],
                "security_opt": ["no-new-privileges:true"],
                "volumes": [
                    {
                        "type": "bind",
                        "source": "/private/runtime.yaml",
                        "target": "/app/contracts/runtime/runtime_config.yaml",
                        "read_only": True,
                    }
                ],
                "environment": {
                    "ONEX_BOX_ID": "sim-preflight",
                    "ONEX_RUNTIME_LANE": "sim-202",
                    "KAFKA_BOOTSTRAP_SERVERS": "redpanda:9092",
                    "OMNIBASE_INFRA_DB_URL": "postgresql://test@postgres:5432/omnibase_infra",
                    "OMNIDASH_ANALYTICS_DB_URL": "postgresql://test@postgres:5432/omnidash_analytics",
                    "VALKEY_HOST": "valkey",
                    "BIFROST_VERIFY_ENDPOINTS": "false",
                    "OTEL_SDK_DISABLED": "true",
                },
            }
        )
    return {
        "name": "omnibase-infra-sim-preflight",
        "services": services,
        "networks": {
            "omnibase-infra-dogfood-network": {
                "name": "omnibase-infra-sim-preflight-network",
                "internal": True,
                "driver": "bridge",
            }
        },
        "volumes": {},
    }


def test_exact_internal_network_and_graph_service_boundary() -> None:
    assert verify(_config()) == {"isolated": True, "service_count": 13}


@pytest.mark.parametrize(
    "mutation",
    [
        "external_network",
        "public_port",
        "host_network",
        "host_alias",
        "docker_socket",
        "privileged",
        "extra_service",
        "writable_bind",
        "external_dns",
        "missing_config",
        "extra_capability",
        "missing_bootstrap_capability",
        "shared_volume",
        "wrong_broker",
        "missing_security_opt",
    ],
)
def test_refuses_escape_hatches(mutation: str) -> None:
    config = copy.deepcopy(_config())
    runtime = config["services"]["runtime-effects"]
    if mutation == "external_network":
        config["networks"]["omnibase-infra-dogfood-network"]["internal"] = False
    elif mutation == "public_port":
        runtime["ports"] = [{"host_ip": "0.0.0.0", "published": "65086"}]  # noqa: S104 -- refusal case
    elif mutation == "host_network":
        runtime["network_mode"] = "host"
    elif mutation == "host_alias":
        runtime["extra_hosts"] = ["host.docker.internal=host-gateway"]
    elif mutation == "docker_socket":
        runtime["volumes"].append(
            {
                "type": "bind",
                "source": "/var/run/docker.sock",
                "target": "/socket",
                "read_only": True,
            }
        )
    elif mutation == "privileged":
        runtime["privileged"] = True
    elif mutation == "extra_service":
        config["services"]["projection-api"] = {}
    elif mutation == "writable_bind":
        runtime["volumes"][0]["read_only"] = False
    elif mutation == "external_dns":
        runtime["dns"] = ["8.8.8.8"]
    elif mutation == "missing_config":
        runtime["volumes"] = []
    elif mutation == "extra_capability":
        runtime["cap_add"] = ["NET_ADMIN"]
    elif mutation == "missing_bootstrap_capability":
        runtime["cap_add"].remove("FOWNER")
    elif mutation == "shared_volume":
        config["volumes"]["v"] = {"name": "omnibase-infra-postgres-data"}
    elif mutation == "wrong_broker":
        runtime["environment"]["KAFKA_BOOTSTRAP_SERVERS"] = (
            "192.168.86.201:9092"  # kafka-fallback-ok: negative fixture must reject shared-lab broker access.
        )
    elif mutation == "missing_security_opt":
        runtime["security_opt"] = []
    with pytest.raises(ValueError):
        verify(config)
