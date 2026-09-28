# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for exact mount boundaries on the disposable sim-preflight stack."""

from __future__ import annotations

import copy
import importlib.util
import json
import runpy
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = (
    _REPO_ROOT / "scripts" / "runtime_build" / "verify_sim_preflight_mounts.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "verify_sim_preflight_mounts", _SCRIPT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

_PRIVATE_FILES = (
    "main-runtime-config.yaml",
    "effects-runtime-config.yaml",
    "gateway-keys.json",
    "terminal-private.pem",
    "gateway-private.pem",
    "terminal-public.pem",
    "omninode-realm.json",
    "workflow-contracts.yaml",
)
_NAMED_VOLUMES = {
    "dogfood_postgres_data": "omnibase-infra-sim-preflight-postgres-data",
    "dogfood_redpanda_data": "omnibase-infra-sim-preflight-redpanda-data",
    "dogfood_valkey_data": "omnibase-infra-sim-preflight-valkey-data",
    "dogfood_runtime_logs": "omninode-sim-preflight-runtime-logs",
    "dogfood_runtime_data": "omninode-sim-preflight-runtime-data",
    "dogfood_effects_logs": "omninode-sim-preflight-effects-logs",
    "dogfood_effects_data": "omninode-sim-preflight-effects-data",
    "sim_preflight_cloud_migrations": "omnibase-infra-sim-preflight-cloud-migrations",
}


def _bind(source: Path, target: str) -> dict[str, Any]:
    return {
        "type": "bind",
        "source": str(source),
        "target": target,
        "read_only": True,
    }


def _volume(source: str, target: str) -> dict[str, Any]:
    return {"type": "volume", "source": source, "target": target}


def _fixture(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    private_dir = tmp_path / "private"
    private_dir.mkdir(mode=0o700)
    private_dir.chmod(0o700)
    for filename in _PRIVATE_FILES:
        path = private_dir / filename
        path.write_text("fixture\n", encoding="utf-8")
        path.chmod(0o600)

    root = _REPO_ROOT
    services: dict[str, dict[str, Any]] = {
        "postgres": {
            "volumes": [
                _volume("dogfood_postgres_data", "/var/lib/postgresql/data"),
                _bind(
                    root / "docker/migrations/forward", "/docker-entrypoint-initdb.d"
                ),
            ]
        },
        "redpanda": {
            "volumes": [_volume("dogfood_redpanda_data", "/var/lib/redpanda/data")]
        },
        "redpanda-partition-cap": {},
        "valkey": {"volumes": [_volume("dogfood_valkey_data", "/data")]},
        "forward-migration": {
            "volumes": [
                _bind(
                    root / "scripts/run-forward-migrations.sh",
                    "/run-forward-migrations.sh",
                ),
                _bind(root / "docker/migrations/forward", "/migrations/forward"),
            ]
        },
        "migration-gate": {
            "volumes": [
                _bind(
                    root / "scripts/check_migrations_complete.sh",
                    "/check_migrations_complete.sh",
                )
            ]
        },
        "intelligence-migration": {
            "volumes": [
                _bind(
                    root / "scripts/run-intelligence-migrations.sh",
                    "/run-intelligence-migrations.sh",
                ),
                _bind(
                    root / "docker/migrations/intelligence", "/migrations/intelligence"
                ),
            ]
        },
        "omninode-runtime": {
            "volumes": [
                _volume("dogfood_runtime_logs", "/app/logs"),
                _volume("dogfood_runtime_data", "/app/data"),
                _bind(
                    private_dir / "main-runtime-config.yaml",
                    "/app/contracts/runtime/runtime_config.yaml",
                ),
            ]
        },
        "runtime-effects": {
            "volumes": [
                _volume("dogfood_effects_logs", "/app/logs"),
                _volume("dogfood_effects_data", "/app/data"),
                _bind(
                    private_dir / "effects-runtime-config.yaml",
                    "/app/contracts/runtime/runtime_config.yaml",
                ),
                _bind(
                    private_dir / "gateway-keys.json",
                    "/app/config/execution-graph/gateway-keys.json",
                ),
                _bind(
                    private_dir / "terminal-private.pem",
                    "/app/config/execution-graph/terminal-private.pem",
                ),
            ]
        },
        "keycloak": {
            "volumes": [
                _bind(
                    private_dir / "omninode-realm.json",
                    "/opt/keycloak/data/import/omninode-realm.json",
                )
            ]
        },
        "cloud-migration-files": {
            "volumes": [_volume("sim_preflight_cloud_migrations", "/work")]
        },
        "cloud-migration": {
            "volumes": [
                _volume("sim_preflight_cloud_migrations", "/work"),
                _bind(
                    root / "docker/migrations/cloud/run-cloud-migrations.sh",
                    "/run-cloud-migrations.sh",
                ),
            ]
        },
        "onex-api": {
            "volumes": [
                _bind(
                    private_dir / "workflow-contracts.yaml",
                    "/app/workflow-contracts.yaml",
                ),
                _bind(
                    private_dir / "gateway-private.pem",
                    "/app/config/execution-graph/gateway-private.pem",
                ),
                _bind(
                    private_dir / "terminal-public.pem",
                    "/app/config/execution-graph/terminal-public.pem",
                ),
            ]
        },
    }
    volumes = {
        key: {"name": name, "driver": "local"} for key, name in _NAMED_VOLUMES.items()
    }
    return {"services": services, "volumes": volumes}, private_dir


def test_accepts_exact_readonly_private_and_migration_mounts(tmp_path: Path) -> None:
    config, private_dir = _fixture(tmp_path)

    result = _MODULE.verify_mounts(config, private_dir)

    assert result == {"bind_mount_count": 15, "named_volume_count": 8}


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker CLI is unavailable")
def test_accepts_private_path_rewritten_real_isolated_compose_render(
    tmp_path: Path,
) -> None:
    auth_test = (
        _REPO_ROOT / "tests" / "scripts" / "test_sim_preflight_auth_overlay_omn19726.py"
    )
    render = runpy.run_path(str(auth_test))["_render"]
    rendered = render(isolated=True)
    assert rendered.returncode == 0, "synthetic Compose config render failed"
    config = json.loads(rendered.stdout)
    _, private_dir = _fixture(tmp_path)
    private_source_by_target = {
        "/app/contracts/runtime/runtime_config.yaml": "effects-runtime-config.yaml",
        "/app/config/execution-graph/gateway-keys.json": "gateway-keys.json",
        "/app/config/execution-graph/terminal-private.pem": "terminal-private.pem",
        "/opt/keycloak/data/import/omninode-realm.json": "omninode-realm.json",
        "/app/workflow-contracts.yaml": "workflow-contracts.yaml",
        "/app/config/execution-graph/gateway-private.pem": "gateway-private.pem",
        "/app/config/execution-graph/terminal-public.pem": "terminal-public.pem",
    }
    for service in config["services"].values():
        for mount in service.get("volumes", []):
            filename = private_source_by_target.get(mount.get("target"))
            if filename is not None:
                mount["source"] = str(private_dir / filename)
    main_runtime = config["services"]["omninode-runtime"]
    for mount in main_runtime["volumes"]:
        if mount.get("target") == "/app/contracts/runtime/runtime_config.yaml":
            mount["source"] = str(private_dir / "main-runtime-config.yaml")

    result = _MODULE.verify_mounts(config, private_dir)

    assert result == {"bind_mount_count": 15, "named_volume_count": 8}


@pytest.mark.parametrize(
    "mutation",
    [
        "unexpected_host_bind",
        "wrong_migration_source",
        "wrong_private_target",
        "writable_bind",
        "extra_named_volume",
        "wrong_volume_name",
        "external_volume",
        "remote_volume_driver",
        "volume_driver_options",
    ],
)
def test_rejects_unapproved_mount_or_volume_shape(
    tmp_path: Path, mutation: str
) -> None:
    config, private_dir = _fixture(tmp_path)
    altered = copy.deepcopy(config)
    runtime_mounts = altered["services"]["runtime-effects"]["volumes"]
    if mutation == "unexpected_host_bind":
        runtime_mounts.append(
            _bind(Path("/private/credential-directory"), "/run/extra")
        )
    elif mutation == "wrong_migration_source":
        altered["services"]["forward-migration"]["volumes"][0]["source"] = (
            "/outside-source.sh"
        )
    elif mutation == "wrong_private_target":
        runtime_mounts[-1]["target"] = "/outside/terminal-private.pem"
    elif mutation == "writable_bind":
        runtime_mounts[-1]["read_only"] = False
    elif mutation == "extra_named_volume":
        altered["volumes"]["unexpected"] = {
            "name": "omnibase-infra-sim-preflight-unexpected",
            "driver": "local",
        }
    elif mutation == "wrong_volume_name":
        altered["volumes"]["dogfood_postgres_data"]["name"] = "shared-postgres-data"
    elif mutation == "external_volume":
        altered["volumes"]["dogfood_postgres_data"]["external"] = True
    elif mutation == "remote_volume_driver":
        altered["volumes"]["dogfood_postgres_data"]["driver"] = "nfs"
    elif mutation == "volume_driver_options":
        altered["volumes"]["dogfood_postgres_data"]["driver_opts"] = {"type": "nfs"}

    with pytest.raises(ValueError):
        _MODULE.verify_mounts(altered, private_dir)


@pytest.mark.parametrize(
    "mutation", ["directory_mode", "file_mode", "symlink_file", "symlink_directory"]
)
def test_rejects_unsafe_private_mount_sources(tmp_path: Path, mutation: str) -> None:
    config, private_dir = _fixture(tmp_path)
    target_file = private_dir / "terminal-private.pem"
    if mutation == "directory_mode":
        private_dir.chmod(0o755)
    elif mutation == "file_mode":
        target_file.chmod(0o644)
    elif mutation == "symlink_file":
        target_file.unlink()
        target_file.symlink_to(tmp_path / "not-private.pem")
    else:
        linked_dir = tmp_path / "private-link"
        linked_dir.symlink_to(private_dir, target_is_directory=True)
        private_dir = linked_dir

    with pytest.raises(ValueError):
        _MODULE.verify_mounts(config, private_dir)
