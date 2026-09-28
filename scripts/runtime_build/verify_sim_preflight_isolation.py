# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail closed on the rendered disposable graph clone's network boundary.

This complements, not replaces, the typed runtime node allowlist and source
provenance verifiers. Input may contain credentials; errors never echo values.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PROJECT = "omnibase-infra-sim-preflight"
NETWORK = "omnibase-infra-dogfood-network"
SERVICES = frozenset(
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
        "keycloak",
        "cloud-migration-files",
        "cloud-migration",
        "onex-api",
    }
)
RUNTIMES = frozenset({"omninode-runtime", "runtime-effects"})
BOOTSTRAP_CAPABILITIES = {"CHOWN", "DAC_OVERRIDE", "FOWNER", "SETGID", "SETUID"}
RELAY_HEALTHCHECK = {
    "test": [
        "CMD",
        "python",
        "-c",
        "import os; assert os.getuid() == 1000; assert b'import signal; signal.pause()' in open('/proc/1/cmdline', 'rb').read().split(bytes([0]))",
    ],
    "interval": "30s",
    "timeout": "3s",
    "start_period": "5s",
    "retries": 3,
}


def verify_runtime_files(main_path: Path, effects_path: Path) -> dict[str, bool | int]:
    """Validate the typed selection without resolving container-only key paths."""
    import yaml

    from omnibase_infra.runtime.models.model_graph_ledger_node_allowlist import (
        ModelGraphLedgerNodeAllowlist,
    )

    for profile, path in (("main", main_path), ("effects", effects_path)):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("runtime config must be an object")
        ModelGraphLedgerNodeAllowlist.model_validate(
            raw.get("graph_ledger_node_allowlist")
        )
        if (
            raw.get("event_bus", {}).get("type") != "kafka"
            or raw.get("local_ingress", {}).get("enabled") is not False
            or raw.get("pattern_b_broker", {}).get("enabled") is not False
            or raw.get("contract_registry", {}).get("enabled") is not False
        ):
            raise ValueError("runtime transport or auxiliary consumers not restricted")
        if profile == "effects" and (
            not raw.get("execution_graph_read")
            or not raw.get("execution_graph_read_gateway")
        ):
            raise ValueError("effects graph read and trusted gateway config required")
        if profile == "main" and raw.get("execution_graph_read"):
            raise ValueError("main must not activate graph effect")
    return {"runtime_configs_restricted": True}


def verify(config: dict[str, Any]) -> dict[str, bool | int]:
    """Validate exact disposable resources; return only nonsecret counts."""
    service_names = set(config.get("services", {}))
    if config.get("name") != PROJECT or service_names not in (
        SERVICES,
        SERVICES | {"relay-transport"},
    ):
        raise ValueError("isolated project or service set mismatch")
    networks = config.get("networks", {})
    if set(networks) != {NETWORK}:
        raise ValueError("isolated network set mismatch")
    network = networks[NETWORK]
    if (
        network.get("name") != PROJECT + "-network"
        or network.get("driver") != "bridge"
        or network.get("enable_ipv6")
        or network.get("internal") is not True
        or network.get("external")
        or network.get("driver_opts")
    ):
        raise ValueError("network must be private and internal")
    for volume in config.get("volumes", {}).values():
        if (
            volume.get("external")
            or volume.get("driver_opts")
            or not str(volume.get("name", "")).startswith(
                (PROJECT + "-", "omninode-sim-preflight-")
            )
        ):
            raise ValueError("volume must belong to disposable project")
    names: set[str] = set()
    for name, service in config["services"].items():
        expected = (
            "omninode-sim-preflight-runtime"
            if name == "omninode-runtime"
            else "omninode-sim-preflight-runtime-effects"
            if name == "runtime-effects"
            else f"{PROJECT}-{name}"
        )
        if service.get("container_name") != expected or expected in names:
            raise ValueError("container identity mismatch")
        names.add(expected)
        if (
            set(service.get("networks", {})) != {NETWORK}
            or service.get("network_mode")
            or service.get("extra_hosts")
            or service.get("privileged")
            or (
                set(service.get("cap_add", []))
                != (BOOTSTRAP_CAPABILITIES if name in RUNTIMES else set())
            )
            or service.get("devices")
            or service.get("pid")
            or service.get("ipc")
            or service.get("volumes_from")
            or service.get("use_api_socket")
        ):
            raise ValueError("service has an isolation escape hatch")
        if service.get("dns") != ["127.0.0.1"]:
            raise ValueError("external DNS forwarding must be disabled")
        if name == "relay-transport" and (
            service.get("environment")
            or service.get("env_file")
            or service.get("secrets")
            or service.get("configs")
            or service.get("volumes")
            or service.get("ports")
            or service.get("read_only") is not True
            or service.get("user") != "1000:1000"
            or service.get("cap_drop") != ["ALL"]
            or service.get("security_opt") != ["no-new-privileges:true"]
            or service.get("entrypoint") != ["python"]
            or service.get("command") != ["-c", "import signal; signal.pause()"]
            or service.get("healthcheck") != RELAY_HEALTHCHECK
        ):
            raise ValueError("relay transport must be inert and credential-free")
        for port in service.get("ports", []):
            if port.get("host_ip") != "127.0.0.1":
                raise ValueError("published ports must be loopback only")
        targets: set[str] = set()
        for mount in service.get("volumes", []):
            source = str(mount.get("source", ""))
            target = str(mount.get("target", ""))
            if "docker.sock" in source or "docker.sock" in target:
                raise ValueError("Docker socket mounting is forbidden")
            if mount.get("type") == "bind" and mount.get("read_only") is not True:
                raise ValueError("host bind mounts must be read only")
            targets.add(target)
        if name in RUNTIMES:
            environment = service.get("environment", {})
            if (
                any(
                    urlsplit(str(environment.get(key, ""))).hostname != "postgres"
                    for key in ("OMNIBASE_INFRA_DB_URL", "OMNIDASH_ANALYTICS_DB_URL")
                )
                or environment.get("VALKEY_HOST") != "valkey"
            ):
                raise ValueError("runtime databases and cache must be disposable")
            if (
                environment.get("BIFROST_VERIFY_ENDPOINTS") != "false"
                or environment.get("OTEL_SDK_DISABLED") != "true"
                or any(
                    value
                    for key, value in environment.items()
                    if key.lower().endswith("_proxy")
                    or key.startswith("OTEL_EXPORTER_")
                )
                or any(
                    environment.get(key)
                    for key in (
                        "GEMINI_API_KEY",
                        "GOOGLE_API_KEY",
                        "OPENROUTER_API_KEY",
                        "LLM_GLM_API_KEY",
                        "LOCAL_LLM_SHARED_SECRET",
                        "GITHUB_TOKEN",
                        "GH_TOKEN",
                        "LINEAR_API_KEY",
                        "INFISICAL_CLIENT_SECRET",
                    )
                )
            ):
                raise ValueError(
                    "runtime has external credentials or egress configuration"
                )
            if (
                service.get("cap_drop") != ["ALL"]
                or "no-new-privileges:true" not in service.get("security_opt", [])
                or "/app/contracts/runtime/runtime_config.yaml" not in targets
                or environment.get("ONEX_BOX_ID") != "sim-preflight"
                or environment.get("ONEX_RUNTIME_LANE") != "sim-202"
                or environment.get("KAFKA_BOOTSTRAP_SERVERS") != "redpanda:9092"
            ):
                raise ValueError("runtime isolation configuration missing or invalid")
    return {"isolated": True, "service_count": len(service_names)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-config", type=Path)
    parser.add_argument("--effects-config", type=Path)
    parser.add_argument("--private-dir", type=Path)
    parser.add_argument("--check-ports", action="store_true")
    args = parser.parse_args()
    result: dict[str, bool | int]
    try:
        if args.main_config or args.effects_config:
            if not args.main_config or not args.effects_config:
                raise ValueError("both runtime configs required")
            result = verify_runtime_files(args.main_config, args.effects_config)
        else:
            from scripts.runtime_build.verify_sim_preflight_mounts import verify_mounts

            config = json.load(sys.stdin)
            result = verify(config)
            if args.private_dir is None:
                raise ValueError("private directory is required for launch validation")
            result.update(verify_mounts(config, args.private_dir))
            if args.check_ports:
                for service in config["services"].values():
                    for port in service.get("ports", []):
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                            probe.bind(("127.0.0.1", int(port["published"])))
    except (ValueError, TypeError, KeyError, AttributeError, OSError):
        print("isolated sim-preflight validation failed", file=sys.stderr)
        return 65
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
