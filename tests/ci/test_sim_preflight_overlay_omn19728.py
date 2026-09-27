# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Static isolation assertions for the disposable OMN-19728 sim overlay."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

pytestmark = pytest.mark.ci

ROOT = Path(__file__).resolve().parents[2]
OVERLAY = ROOT / "docker" / "docker-compose.sim-preflight.yml"
RENDERER = ROOT / "scripts" / "runtime_build" / "render_sim_preflight_compose.sh"
SUMMARY = ROOT / "scripts" / "runtime_build" / "summarize_sim_preflight_compose.py"
SIM_202_OVERLAY = ROOT / "docker" / "docker-compose.sim-202.yml"

_EXPECTED_SUMMARY = {
    "all_bind_mounts_read_only": True,
    "bind_mount_count": 8,
    "credential_fields_blank": True,
    "container_count": 10,
    "db_hosts_are_postgres": True,
    "network_count": 1,
    "port_count": 7,
    "project_is_expected": True,
    "service_count": 10,
    "volume_count": 7,
}
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

_RENDER_INPUTS = {
    "DOGFOOD_REDPANDA_ADVERTISED_HOST": "127.0.0.1",
    "DOGFOOD_RUNTIME_EFFECTS_BIFROST_VERIFY_ENDPOINTS": "false",
    "DOGFOOD_RUNTIME_EFFECTS_CAPABILITIES": "[]",
    "DOGFOOD_RUNTIME_EFFECTS_OMNIMEMORY_ENABLED": "false",
    "DOGFOOD_RUNTIME_EFFECTS_OMNIMEMORY_MEMGRAPH_HOST": "memgraph",
    "DOGFOOD_RUNTIME_EFFECTS_PORT": "8085",
    "DOGFOOD_RUNTIME_EFFECTS_SECRET_RESOLVER_CONFIG_JSON": "{}",
    "DOGFOOD_RUNTIME_EFFECTS_SECRET_RESOLVER_CONFIG_PATH": "/var/empty/config",
    "DOGFOOD_RUNTIME_MAIN_BIFROST_VERIFY_ENDPOINTS": "false",
    "DOGFOOD_RUNTIME_MAIN_CAPABILITIES": "[]",
    "DOGFOOD_RUNTIME_MAIN_OMNIMEMORY_ENABLED": "false",
    "DOGFOOD_RUNTIME_MAIN_OMNIMEMORY_MEMGRAPH_HOST": "memgraph",
    "DOGFOOD_RUNTIME_MAIN_PORT": "8085",
    "DOGFOOD_RUNTIME_MAIN_PUBLISH_INTROSPECTION": "false",
    "DOGFOOD_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_JSON": "{}",
    "DOGFOOD_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_PATH": "/var/empty/config",
    "DOGFOOD_TOPIC_PROVISIONER_MAX_PARTITIONS": "1",
    "LLM_CLOUD_ENDPOINT_HOST_ALLOWLIST": "example.invalid",
    "OMNICLAUDE_SKILLS_DIR": "/var/empty",
    "OMNIMEMORY_MEMGRAPH_PORT": "7687",
    "OMNINODE_RUNTIME_PASSWORD": "render-only-password",
    "ONEX_ACTIVE_RUNTIME_PACKAGES": "omnibase_infra",
    "POSTGRES_PASSWORD": "render-only-password",
    "SIM_202_RUNTIME_IMAGE": "example.invalid/sim@sha256:" + "a" * 64,
    "TENANT_PROJECTION_WRITER_PASSWORD": "render-only-password",
    "VALKEY_PASSWORD": "render-only-password",
}


def test_sim_preflight_overlay_has_no_shared_identifiers_or_ambient_credentials() -> (
    None
):
    """The rendered-chain overlay must remain a separate, credential-blank lane."""
    raw = OVERLAY.read_text(encoding="utf-8")
    assert "name: omnibase-infra-sim-preflight" in raw
    assert "name: omnibase-infra-sim-202" not in raw
    assert "-p omnibase-infra-sim-202` is forbidden" in raw
    assert "omnibase-infra-dogfood-network" in raw
    assert "name: omnibase-infra-sim-preflight-network" in raw
    for name in (
        'GEMINI_API_KEY: ""',
        'OPENROUTER_API_KEY: ""',
        'LLM_GLM_API_KEY: ""',
        'LOCAL_LLM_SHARED_SECRET: ""',
        'GITHUB_TOKEN: ""',
        'LINEAR_API_KEY: ""',
    ):
        assert name in raw
    assert 'profiles: !override ["sim-preflight-faults-disabled"]' in raw
    assert "127.0.0.1:65092:19092" in raw
    assert "127.0.0.1:65444:9644" in raw
    assert "127.0.0.1:65036:5432" in raw


def test_renderer_pins_only_the_disposable_project_and_has_no_lifecycle_mode() -> None:
    raw = RENDERER.read_text(encoding="utf-8")
    assert 'readonly PROJECT="omnibase-infra-sim-preflight"' in raw
    assert 'COMPOSE_PROJECT_NAME="${PROJECT}"' in raw
    assert 'docker compose -p "${PROJECT}"' in raw
    assert "config --format json" in raw
    assert "summarize_sim_preflight_compose.py" in raw
    assert " up " not in raw
    assert " down " not in raw


@pytest.mark.parametrize(
    ("environment", "args", "expected"),
    [
        ({"COMPOSE_PROJECT_NAME": "omnibase-infra-sim-202"}, (), "refusing"),
        ({}, ("up",), "usage:"),
    ],
)
def test_renderer_rejects_project_override_and_lifecycle_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    environment: dict[str, str],
    args: tuple[str, ...],
    expected: str,
) -> None:
    """The wrapper refuses before a Docker executable can be reached."""
    fake_docker = tmp_path / "docker"
    fake_docker.write_text("#!/usr/bin/env sh\nexit 99\n", encoding="utf-8")
    fake_docker.chmod(fake_docker.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.delenv("COMPOSE_PROJECT_NAME", raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    result = subprocess.run(
        [str(RENDERER), *args], capture_output=True, check=False, text=True
    )

    assert result.returncode == 64
    assert expected in result.stderr


def test_renderer_invokes_only_the_pinned_compose_chain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A fake Docker binary records structural argv without a real render."""
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(
        '#!/usr/bin/env sh\nprintf \'%s\\n\' "$@" > "$RENDER_RECORD"\nprintf \'%s\' "$RENDER_CONFIG"\n',
        encoding="utf-8",
    )
    fake_docker.chmod(fake_docker.stat().st_mode | stat.S_IXUSR)
    record = tmp_path / "argv"
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("RENDER_RECORD", str(record))
    monkeypatch.setenv("RENDER_CONFIG", json.dumps(_valid_summary_config()))
    monkeypatch.delenv("COMPOSE_PROJECT_NAME", raising=False)

    result = subprocess.run(
        [str(RENDERER)], capture_output=True, check=False, text=True
    )

    assert result.returncode == 0
    assert json.loads(result.stdout) == _EXPECTED_SUMMARY
    assert record.read_text(encoding="utf-8").splitlines() == [
        "compose",
        "-p",
        "omnibase-infra-sim-preflight",
        "--env-file",
        "/dev/null",
        "-f",
        str(ROOT / "docker" / "docker-compose.dogfood.yml"),
        "-f",
        str(ROOT / "docker" / "docker-compose.sim-202.yml"),
        "-f",
        str(ROOT / "docker" / "docker-compose.sim-preflight.yml"),
        "--profile",
        "dogfood",
        "config",
        "--format",
        "json",
    ]


def test_renderer_suppresses_compose_failure_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env sh\necho command-sentinel >&2\nexit 1\n", encoding="utf-8"
    )
    fake_docker.chmod(fake_docker.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.delenv("COMPOSE_PROJECT_NAME", raising=False)

    result = subprocess.run(
        [str(RENDERER)], capture_output=True, check=False, text=True
    )

    assert result.returncode == 1
    assert result.stderr == "sim-preflight compose config failed (details suppressed)\n"
    assert "command-sentinel" not in result.stderr


def test_renderer_suppresses_summary_when_compose_fails_after_valid_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env sh\nprintf '%s' \"$RENDER_CONFIG\"\necho command-sentinel >&2\nexit 1\n",
        encoding="utf-8",
    )
    fake_docker.chmod(fake_docker.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("RENDER_CONFIG", json.dumps(_valid_summary_config()))
    monkeypatch.delenv("COMPOSE_PROJECT_NAME", raising=False)

    result = subprocess.run(
        [str(RENDERER)], capture_output=True, check=False, text=True
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "sim-preflight compose config failed (details suppressed)\n"


def test_strict_summary_excludes_adversarial_non_environment_values() -> None:
    sentinel = "ambient-sentinel"
    raw = {
        "services": {
            "runtime": {
                "container_name": "safe-container",
                "command": [sentinel],
                "environment": {"TOKEN": sentinel},
                "labels": {"unsafe": sentinel},
                "volumes": [{"source": sentinel, "target": "/safe"}],
                "ports": [{"host_ip": "127.0.0.1", "published": 65085}],
            }
        },
        "networks": {"n": {"name": "safe-network", "driver_opts": {"x": sentinel}}},
        "volumes": {"v": {"name": "safe-volume", "labels": {"x": sentinel}}},
    }
    result = subprocess.run(
        ["uv", "run", "python", str(SUMMARY)],
        cwd=ROOT,
        input=json.dumps(raw),
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode != 0
    assert sentinel not in result.stdout
    assert sentinel not in result.stderr


def _valid_summary_config() -> dict[str, Any]:
    """Minimal fixed-identity Compose model for wrapper argv testing only."""
    containers = {
        "postgres": "omnibase-infra-sim-preflight-postgres",
        "redpanda": "omnibase-infra-sim-preflight-redpanda",
        "redpanda-partition-cap": "omnibase-infra-sim-preflight-redpanda-partition-cap",
        "valkey": "omnibase-infra-sim-preflight-valkey",
        "forward-migration": "omnibase-infra-sim-preflight-forward-migration",
        "migration-gate": "omnibase-infra-sim-preflight-migration-gate",
        "intelligence-migration": "omnibase-infra-sim-preflight-intelligence-migration",
        "omninode-runtime": "omninode-sim-preflight-runtime",
        "runtime-effects": "omninode-sim-preflight-runtime-effects",
        "projection-api": "omnimarket-sim-preflight-projection-api",
    }
    published = {
        "postgres": ["65036"],
        "redpanda": ["65092", "65444"],
        "valkey": ["65379"],
        "omninode-runtime": ["65085"],
        "runtime-effects": ["65086"],
        "projection-api": ["65002"],
    }
    services: dict[str, dict[str, Any]] = {}
    for service, container in containers.items():
        service_config: dict[str, Any] = {
            "container_name": container,
            "ports": [
                {"host_ip": "127.0.0.1", "published": port}
                for port in published.get(service, [])
            ],
        }
        if service in {"omninode-runtime", "runtime-effects", "projection-api"}:
            service_config["environment"] = {
                **dict.fromkeys(_CREDENTIAL_KEYS, ""),
                "OMNIBASE_INFRA_DB_URL": "postgresql://x@postgres:5432/x",
                "OMNIDASH_ANALYTICS_DB_URL": "postgresql://x@postgres:5432/x",
            }
            service_config["volumes"] = [
                {"type": "bind", "read_only": True}
                for _ in range(3 if service != "projection-api" else 2)
            ]
        services[service] = service_config
    volume_names = (
        "omnibase-infra-sim-preflight-postgres-data",
        "omnibase-infra-sim-preflight-redpanda-data",
        "omnibase-infra-sim-preflight-valkey-data",
        "omninode-sim-preflight-runtime-logs",
        "omninode-sim-preflight-runtime-data",
        "omninode-sim-preflight-effects-logs",
        "omninode-sim-preflight-effects-data",
    )
    return {
        "name": "omnibase-infra-sim-preflight",
        "services": services,
        "networks": {"network": {"name": "omnibase-infra-sim-preflight-network"}},
        "volumes": {name: {"name": name} for name in volume_names},
    }


def _docker_compose_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return (
        subprocess.run(
            ["docker", "compose", "version"], check=False, capture_output=True
        ).returncode
        == 0
    )


def _synthetic_environment() -> dict[str, str]:
    return {
        "HOME": os.environ.get("HOME", ""),
        "PATH": os.environ.get("PATH", ""),
        "GEMINI_API_KEY": "ambient-sentinel",
        "GOOGLE_API_KEY": "ambient-sentinel",
        "OPENROUTER_API_KEY": "ambient-sentinel",
        "LLM_GLM_API_KEY": "ambient-sentinel",
        "LOCAL_LLM_SHARED_SECRET": "ambient-sentinel",
        "GITHUB_TOKEN": "ambient-sentinel",
        "GH_TOKEN": "ambient-sentinel",
        "LINEAR_API_KEY": "ambient-sentinel",
        **_RENDER_INPUTS,
    }


def _render_compose(overlay: Path) -> tuple[dict[str, Any], str]:
    """Test-only, non-mutating synthetic parser; operational render uses wrapper."""
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-p",
            "render-only-project",
            "--env-file",
            "/dev/null",
            "-f",
            "docker/docker-compose.dogfood.yml",
            "-f",
            str(overlay.relative_to(ROOT)),
            "--profile",
            "dogfood",
            "config",
            "--format",
            "json",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        env=_synthetic_environment(),
        text=True,
    )
    return cast("dict[str, Any]", json.loads(result.stdout)), result.stdout


@pytest.mark.skipif(not _docker_compose_available(), reason="compose renderer absent")
def test_renderer_output_does_not_interpolate_ambient_sentinels() -> None:
    result = subprocess.run(
        [str(RENDERER)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        env=_synthetic_environment(),
        text=True,
    )
    assert "ambient-sentinel" not in result.stdout
    assert "ambient-sentinel" not in result.stderr


@pytest.mark.skipif(not _docker_compose_available(), reason="compose renderer absent")
def test_synthetic_render_has_isolated_resources_and_no_ambient_credentials() -> None:
    """Parse the effective multi-file config; render-only values are never printed."""
    rendered, _ = _render_compose(OVERLAY)
    sim_202, _ = _render_compose(SIM_202_OVERLAY)
    services = cast("dict[str, dict[str, Any]]", rendered["services"])
    sim_202_services = cast("dict[str, dict[str, Any]]", sim_202["services"])
    assert not {"dogfood-delegation-fault-429", "dogfood-delegation-fault-503"} & set(
        services
    )
    containers = [service.get("container_name") for service in services.values()]
    assert len(containers) == len(set(containers))
    assert not set(containers).intersection(
        service.get("container_name") for service in sim_202_services.values()
    )
    for resource in ("networks", "volumes"):
        ours = {
            entry["name"]
            for entry in cast("dict[str, dict[str, str]]", rendered[resource]).values()
        }
        theirs = {
            entry["name"]
            for entry in cast("dict[str, dict[str, str]]", sim_202[resource]).values()
        }
        assert not ours.intersection(theirs)
    ours_ports = {
        (port["host_ip"], port["published"])
        for service in services.values()
        for port in cast("list[dict[str, Any]]", service.get("ports", []))
    }
    theirs_ports = {
        (port["host_ip"], port["published"])
        for service in sim_202_services.values()
        for port in cast("list[dict[str, Any]]", service.get("ports", []))
    }
    assert len(ours_ports) == sum(
        len(cast("list[dict[str, Any]]", service.get("ports", [])))
        for service in services.values()
    )
    assert not ours_ports.intersection(theirs_ports)
    for service in services.values():
        for port in cast("list[dict[str, Any]]", service.get("ports", [])):
            assert port["host_ip"] == "127.0.0.1"
    for service_name in ("omninode-runtime", "runtime-effects", "projection-api"):
        environment = cast("dict[str, str]", services[service_name]["environment"])
        assert "@postgres:5432/" in environment["OMNIBASE_INFRA_DB_URL"]
        assert "@postgres:5432/" in environment["OMNIDASH_ANALYTICS_DB_URL"]
        for mount in cast("list[dict[str, Any]]", services[service_name]["volumes"]):
            if mount["type"] == "bind":
                assert mount["read_only"] is True
        for key in (
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "OPENROUTER_API_KEY",
            "LLM_GLM_API_KEY",
            "LOCAL_LLM_SHARED_SECRET",
            "GITHUB_TOKEN",
            "GH_TOKEN",
            "LINEAR_API_KEY",
        ):
            assert environment[key] == ""
