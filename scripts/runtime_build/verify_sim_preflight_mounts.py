# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Validate the exact host-bind and named-volume boundary for sim-preflight."""

from __future__ import annotations

import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

_PRIVATE_FILES = frozenset(
    {
        "main-runtime-config.yaml",
        "effects-runtime-config.yaml",
        "gateway-keys.json",
        "terminal-private.pem",
        "gateway-private.pem",
        "terminal-public.pem",
        "omninode-realm.json",
        "workflow-contracts.yaml",
    }
)

_VOLUME_NAMES = {
    "dogfood_postgres_data": "omnibase-infra-sim-preflight-postgres-data",
    "dogfood_redpanda_data": "omnibase-infra-sim-preflight-redpanda-data",
    "dogfood_valkey_data": "omnibase-infra-sim-preflight-valkey-data",
    "dogfood_runtime_logs": "omninode-sim-preflight-runtime-logs",
    "dogfood_runtime_data": "omninode-sim-preflight-runtime-data",
    "dogfood_effects_logs": "omninode-sim-preflight-effects-logs",
    "dogfood_effects_data": "omninode-sim-preflight-effects-data",
    "sim_preflight_cloud_migrations": "omnibase-infra-sim-preflight-cloud-migrations",
}


def _expected_mounts(
    root: Path, private_dir: Path
) -> dict[str, frozenset[tuple[str, str, str, bool]]]:
    def binds(source: Path, target: str) -> tuple[str, str, str, bool]:
        return ("bind", str(source), target, True)

    def volume(source: str, target: str) -> tuple[str, str, str, bool]:
        return ("volume", source, target, False)

    forward = root / "docker" / "migrations" / "forward"
    return {
        "postgres": frozenset(
            {
                volume("dogfood_postgres_data", "/var/lib/postgresql/data"),
                binds(forward, "/docker-entrypoint-initdb.d"),
            }
        ),
        "redpanda": frozenset(
            {volume("dogfood_redpanda_data", "/var/lib/redpanda/data")}
        ),
        "redpanda-partition-cap": frozenset(),
        "valkey": frozenset({volume("dogfood_valkey_data", "/data")}),
        "forward-migration": frozenset(
            {
                binds(
                    root / "scripts" / "run-forward-migrations.sh",
                    "/run-forward-migrations.sh",
                ),
                binds(forward, "/migrations/forward"),
            }
        ),
        "migration-gate": frozenset(
            {
                binds(
                    root / "scripts" / "check_migrations_complete.sh",
                    "/check_migrations_complete.sh",
                )
            }
        ),
        "intelligence-migration": frozenset(
            {
                binds(
                    root / "scripts" / "run-intelligence-migrations.sh",
                    "/run-intelligence-migrations.sh",
                ),
                binds(
                    root / "docker" / "migrations" / "intelligence",
                    "/migrations/intelligence",
                ),
            }
        ),
        "omninode-runtime": frozenset(
            {
                volume("dogfood_runtime_logs", "/app/logs"),
                volume("dogfood_runtime_data", "/app/data"),
                binds(
                    private_dir / "main-runtime-config.yaml",
                    "/app/contracts/runtime/runtime_config.yaml",
                ),
            }
        ),
        "runtime-effects": frozenset(
            {
                volume("dogfood_effects_logs", "/app/logs"),
                volume("dogfood_effects_data", "/app/data"),
                binds(
                    private_dir / "effects-runtime-config.yaml",
                    "/app/contracts/runtime/runtime_config.yaml",
                ),
                binds(
                    private_dir / "gateway-keys.json",
                    "/app/config/execution-graph/gateway-keys.json",
                ),
                binds(
                    private_dir / "terminal-private.pem",
                    "/app/config/execution-graph/terminal-private.pem",
                ),
            }
        ),
        "keycloak": frozenset(
            {
                binds(
                    private_dir / "omninode-realm.json",
                    "/opt/keycloak/data/import/omninode-realm.json",
                )
            }
        ),
        "cloud-migration-files": frozenset(
            {volume("sim_preflight_cloud_migrations", "/work")}
        ),
        "cloud-migration": frozenset(
            {
                volume("sim_preflight_cloud_migrations", "/work"),
                binds(
                    root
                    / "docker"
                    / "migrations"
                    / "cloud"
                    / "run-cloud-migrations.sh",
                    "/run-cloud-migrations.sh",
                ),
            }
        ),
        "onex-api": frozenset(
            {
                binds(
                    private_dir / "workflow-contracts.yaml",
                    "/app/workflow-contracts.yaml",
                ),
                binds(
                    private_dir / "gateway-private.pem",
                    "/app/config/execution-graph/gateway-private.pem",
                ),
                binds(
                    private_dir / "terminal-public.pem",
                    "/app/config/execution-graph/terminal-public.pem",
                ),
            }
        ),
    }


def _check_private_dir(private_dir: Path) -> Path:
    if not private_dir.is_absolute() or private_dir.is_symlink():
        raise ValueError("private mount directory must be absolute and not a symlink")
    if not private_dir.is_dir() or stat.S_IMODE(private_dir.stat().st_mode) != 0o700:
        raise ValueError("private mount directory must be a mode-0700 directory")
    for filename in _PRIVATE_FILES:
        path = private_dir / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError("private mount source must be a regular file")
        if stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise ValueError("private mount source must have mode 0600")
    return private_dir


def _check_volumes(config: dict[str, Any]) -> None:
    volumes = config.get("volumes")
    if not isinstance(volumes, dict) or set(volumes) != set(_VOLUME_NAMES):
        raise ValueError("named-volume keys do not match the disposable allowlist")
    for key, expected_name in _VOLUME_NAMES.items():
        spec = volumes[key]
        if not isinstance(spec, dict):
            raise ValueError("named-volume specification is invalid")
        if (
            spec.get("name") != expected_name
            or spec.get("external")
            or spec.get("driver") not in (None, "local")
            or spec.get("driver_opts")
        ):
            raise ValueError("named volume is not local disposable storage")


def verify_mounts(config: dict[str, Any], private_dir: Path) -> dict[str, int]:
    """Require exact approved bind/volume pairs and safe private source files.

    The caller passes the rendered Compose JSON and the private credential
    directory. Only counts are returned so callers can log a safe summary.
    """
    if not isinstance(config, dict):
        raise ValueError("compose config must be an object")
    private_dir = _check_private_dir(private_dir)
    root = Path(__file__).resolve().parents[2]
    expected = _expected_mounts(root, private_dir)
    services = config.get("services")
    if isinstance(services, dict) and "relay-transport" in services:
        expected["relay-transport"] = frozenset()
    if not isinstance(services, dict) or set(services) != set(expected):
        raise ValueError("service set does not match the mount allowlist")

    bind_count = 0
    for service_name, expected_mounts in expected.items():
        service = services[service_name]
        if not isinstance(service, dict):
            raise ValueError("service mount declaration is invalid")
        mounts = service.get("volumes", [])
        if not isinstance(mounts, list):
            raise ValueError("service volumes must be a list")
        actual: set[tuple[str, str, str, bool]] = set()
        for mount in mounts:
            if not isinstance(mount, dict):
                raise ValueError("service mount declaration is invalid")
            mount_type = mount.get("type")
            source = mount.get("source")
            target = mount.get("target")
            read_only = mount.get("read_only") is True
            if not isinstance(source, str) or not isinstance(target, str):
                raise ValueError("service mount source and target are required")
            if mount_type not in ("bind", "volume"):
                raise ValueError(
                    "only approved bind and named-volume mounts are allowed"
                )
            if mount_type == "bind":
                bind_count += 1
                if not read_only:
                    raise ValueError("host bind mounts must be read-only")
                source_path = Path(source)
                if (
                    not source_path.is_absolute()
                    or source_path.is_symlink()
                    or not source_path.exists()
                ):
                    raise ValueError(
                        "host bind source must be an existing absolute path"
                    )
                resolved_source = str(source_path.resolve(strict=True))
            else:
                if read_only:
                    raise ValueError("disposable named volumes must remain writable")
                resolved_source = source
            item = (mount_type, resolved_source, target, read_only)
            if item in actual:
                raise ValueError("duplicate service mount")
            actual.add(item)
        if actual != expected_mounts:
            raise ValueError(
                "service mounts differ from the exact disposable allowlist"
            )

    _check_volumes(config)
    return {"bind_mount_count": bind_count, "named_volume_count": len(_VOLUME_NAMES)}
