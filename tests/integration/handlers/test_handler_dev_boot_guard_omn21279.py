# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-to-handler integration for the dev boot guard effect (OMN-21279).

Loads node_dev_boot_guard_effect's contract from disk, resolves each routed
operation to its declared handler, and drives HandlerDevBootGuard.handle()
through a fake host execution seam that answers Docker the way a dev lane
does. No Docker daemon is touched.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, get_args

import pytest
import yaml

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.handlers import handler_dev_boot_guard as guard_mod
from omnibase_infra.models.model_dev_boot_report_config import ModelDevBootReportConfig
from omnibase_infra.models.model_dev_boot_request import ModelDevBootRequest

CONTRACT = (
    Path(guard_mod.__file__).resolve().parents[1]
    / "nodes"
    / "node_dev_boot_guard_effect"
    / "contract.yaml"
)
PROJECT = "omnibase-infra"
REVISION = "a" * 40
BOOT_ERROR = (
    "[entrypoint] BOOT FAILED 5/5: preflight exited 1 "
    "BIFROST_BASE_CONTRACT_PATH is not set"
)


class _Lane(Enum):
    DEV = "dev"


@contextlib.contextmanager
def _probe_file_args(lane: object, service: str) -> Iterator[list[str]]:
    yield ["-f", "compose.yml", "-f", f"no-build-{service}.yml"]


def _container(service: str, *, healthy: bool = True) -> dict[str, Any]:
    return {
        "Id": f"id-{service}",
        "Image": f"sha256:{service}",
        "Config": {
            "Labels": {
                "com.docker.compose.project": PROJECT,
                "com.docker.compose.service": service,
                "com.docker.compose.config-hash": f"hash-{service}",
                "org.opencontainers.image.revision": REVISION,
            },
            "Healthcheck": {
                "StartPeriod": 10_000_000_000,
                "Retries": 1,
                "Interval": 5_000_000_000,
                "Timeout": 5_000_000_000,
            },
        },
        "State": {
            "Status": "running",
            "Health": {"Status": "healthy" if healthy else "starting"},
        },
    }


@dataclass
class _Host:
    """Docker as a dev lane answers it, plus every command the guard ran."""

    containers: list[dict[str, Any]] = field(
        default_factory=lambda: [_container(s) for s in guard_mod.DEV_BOOT_SERVICES]
    )
    logs: str = ""
    preflight_rc: int = 0
    preflight_err: str = ""
    calls: list[list[str]] = field(default_factory=list)
    written: dict[Path, object] = field(default_factory=dict)
    tagged: list[tuple[str, str]] = field(default_factory=list)

    def run(
        self, argv: list[str], timeout: int, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        del timeout, env
        self.calls.append(argv)
        # Nothing is executed: the guard's argv is answered from fixed state.
        verb = argv[1]
        if verb == "ps":
            ids = "\n".join(c["Id"] for c in self.containers)
            return subprocess.CompletedProcess(argv, 0, ids, "")
        if verb == "inspect":
            return subprocess.CompletedProcess(argv, 0, json.dumps(self.containers), "")
        if verb == "logs":
            return subprocess.CompletedProcess(argv, 0, self.logs, "")
        if "config" in argv:
            services = {s: {"image": f"img-{s}"} for s in guard_mod.DEV_BOOT_SERVICES}
            return subprocess.CompletedProcess(
                argv, 0, json.dumps({"services": services}), ""
            )
        if "run" in argv:
            return subprocess.CompletedProcess(
                argv, self.preflight_rc, "", self.preflight_err
            )
        if "up" in argv:
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise AssertionError(f"unexpected host command: {argv}")

    def execution(self) -> ModuleType:
        module = ModuleType("fake_deploy_execution")
        module.__dict__.update(
            EnumRuntimeLane=_Lane,
            lane_config_for=lambda lane: SimpleNamespace(
                compose_project=PROJECT, disabled_services=()
            ),
            _run=self.run,
            _compose_file_args=lambda lane: ["-f", "compose.yml"],
            candidate_probe_file_args=_probe_file_args,
            _compose_env=dict,
            IMAGE_PREFLIGHT_SCRIPT="pass",
            CONFIG_HASH_TIMEOUT_SECONDS=30,
            VERIFY_RECREATE_TIMEOUT_SECONDS=30,
            _tag_image=lambda image, tag: self.tagged.append((image, tag)),
            _atomic_write_private_json=self.written.__setitem__,
        )
        return module


def _no_phase_updates(update: object) -> None:
    del update


def _guard(host: _Host, tmp_path: Path) -> guard_mod.HandlerDevBootGuard:
    return guard_mod.HandlerDevBootGuard(
        tmp_path,
        config_hashes=lambda services: {s: f"hash-{s}" for s in services},
        verify_recovery=_no_phase_updates,
        execution=host.execution(),
        reporting=ModelDevBootReportConfig(
            registry_root=tmp_path, plugin_dir=None, ledger_via_bus=False
        ),
    )


def _routes() -> dict[str, type]:
    contract = yaml.safe_load(CONTRACT.read_text())
    routes: dict[str, type] = {}
    for entry in contract["handler_routing"]["handlers"]:
        module = importlib.import_module(entry["handler"]["module"])
        routes[entry["operation"]] = getattr(module, entry["handler"]["name"])
    return routes


def test_contract_routes_every_request_operation_to_the_guard() -> None:
    operations = set(get_args(ModelDevBootRequest.model_fields["operation"].annotation))
    routes = _routes()
    assert set(routes) == operations
    assert set(routes.values()) == {guard_mod.HandlerDevBootGuard}


def test_guard_classifies_as_an_infra_effect(tmp_path: Path) -> None:
    guard = _guard(_Host(), tmp_path)
    assert guard.handler_type is EnumHandlerType.INFRA_HANDLER
    assert guard.handler_category is EnumHandlerTypeCategory.EFFECT


def test_wait_passes_when_every_runtime_service_is_healthy(tmp_path: Path) -> None:
    guard = _guard(_Host(), tmp_path)
    result = _routes()["wait"].handle(guard, ModelDevBootRequest(operation="wait"))
    assert (result.operation, result.status) == ("wait", "pass")


def test_wait_fails_fast_on_an_exhausted_boot(tmp_path: Path) -> None:
    host = _Host(
        containers=[_container(s, healthy=False) for s in guard_mod.DEV_BOOT_SERVICES],
        logs=BOOT_ERROR,
    )
    with pytest.raises(guard_mod.DevBootFailedError, match="BOOT FAILED 5/5"):
        _guard(host, tmp_path).handle(ModelDevBootRequest(operation="wait"))


def test_preflight_refuses_a_candidate_missing_a_required_setting(
    tmp_path: Path,
) -> None:
    host = _Host(
        preflight_rc=1,
        preflight_err="Traceback (most recent call last):\n"
        "BIFROST_BASE_CONTRACT_PATH is not set in the candidate image",
    )
    with pytest.raises(
        guard_mod.DevBootFailedError, match="BIFROST_BASE_CONTRACT_PATH"
    ):
        _guard(host, tmp_path).handle(ModelDevBootRequest(operation="preflight"))
    assert host.calls
    assert all("--no-deps" in argv and "never" in argv for argv in host.calls)
    # OMN-21279: the probe runs the built image, never a rebuild.
    assert all("no-build-" in " ".join(argv) for argv in host.calls)


def test_prepare_retains_the_healthy_family_checked_by_injected_hashes(
    tmp_path: Path,
) -> None:
    host = _Host()
    guard = _guard(host, tmp_path)
    result = guard.handle(ModelDevBootRequest(operation="prepare"))
    assert result.status == "pass"
    assert guard.point.current is not None
    assert guard.point.current.revision == REVISION
    assert set(guard.point.current.images.values()) == {
        f"sha256:{s}" for s in guard_mod.DEV_BOOT_SERVICES
    }
    assert guard.snapshot_path in host.written
    assert len(host.tagged) == len(guard_mod.DEV_BOOT_SERVICES)


def test_prepare_refuses_a_family_whose_settings_drifted(tmp_path: Path) -> None:
    host = _Host()
    host.containers[0]["Config"]["Labels"]["com.docker.compose.config-hash"] = "stale"
    with pytest.raises(guard_mod.DevBootFailedError, match="no last healthy"):
        _guard(host, tmp_path).handle(ModelDevBootRequest(operation="prepare"))
    assert host.written == {}


def test_recovery_phase_updates_reach_the_injected_verifier(tmp_path: Path) -> None:
    class _Phase(StrEnum):
        VERIFY = "verify"

    seen: list[object] = []

    def verify(update: object) -> None:
        assert callable(update)
        update(_Phase.VERIFY, _Phase.VERIFY)
        seen.append(update)

    host = _Host()
    guard = guard_mod.HandlerDevBootGuard(
        tmp_path,
        config_hashes=lambda services: {s: f"hash-{s}" for s in services},
        verify_recovery=verify,
        execution=host.execution(),
        reporting=ModelDevBootReportConfig(
            registry_root=tmp_path, plugin_dir=None, ledger_via_bus=False
        ),
    )
    guard.handle(ModelDevBootRequest(operation="prepare"))
    guard._restore()
    assert len(seen) == 1
    assert any("--force-recreate" in argv for argv in host.calls)
