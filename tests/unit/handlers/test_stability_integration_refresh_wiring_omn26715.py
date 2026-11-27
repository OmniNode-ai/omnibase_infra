# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The refresh handler's host is registered and its recreate gets the lane declaration (OMN-26715)."""

from __future__ import annotations

import asyncio
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_infra.handlers import handler_stability_integration_refresh as module
from omnibase_infra.handlers.handler_stability_integration_refresh import (
    _RECREATE_SH,
    DockerStabilityIntegrationHost,
    HandlerStabilityIntegrationRefresh,
)
from omnibase_infra.nodes.node_stability_integration_refresh_effect.runtime_stability_integration_refresh import (
    bootstrap_container,
)

pytestmark = pytest.mark.unit


def _host(tmp_path: Path) -> DockerStabilityIntegrationHost:
    return DockerStabilityIntegrationHost(
        registry_root=tmp_path / "registry",
        infra_checkout=tmp_path / "infra",
        work_dir=tmp_path / "work",
        operator_env_file=tmp_path / "operator.env",
        health_timeout_seconds=5.0,
        lane_declaration_dir=tmp_path / "config" / "lab",
        lane_overlay_dir=tmp_path / "config" / "lab" / "lane-overlays",
    )


def test_an_unwired_container_cannot_build_the_handler() -> None:
    """Positive control: with nothing registered the handler's host lookup fails."""
    with pytest.raises(Exception, match="DockerStabilityIntegrationHost"):
        HandlerStabilityIntegrationRefresh(ModelONEXContainer())


def test_the_composition_root_registers_the_host_the_handler_resolves(
    tmp_path: Path,
) -> None:
    host = _host(tmp_path)
    container = asyncio.run(bootstrap_container(host))
    assert container.get_service(DockerStabilityIntegrationHost) is host
    # The handler resolves its host through the same registry; it builds only when registered.
    assert isinstance(
        container.get_service(HandlerStabilityIntegrationRefresh),
        HandlerStabilityIntegrationRefresh,
    )


def test_the_recreate_script_exports_the_lane_declaration_dirs(
    tmp_path: Path,
) -> None:
    infra = tmp_path / "infra"
    (infra / "docker").mkdir(parents=True)
    (infra / "docker" / "runtime-policy.env").write_text(
        "ONEX_LANE_OVERLAY_DIR=/stale/overlay\n", encoding="utf-8"
    )
    bindir = tmp_path / "bin"
    bindir.mkdir()
    shim = bindir / "docker"
    shim.write_text(
        '#!/bin/sh\nprintf "%s|%s|%s" "$ONEX_LANE_OVERLAY_DIR" '
        '"$ONEX_LANE_DECLARATION_DIR" "$*"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)
    done = subprocess.run(
        [
            "bash",
            "-c",
            _RECREATE_SH,
            "recreate",
            str(infra),
            str(tmp_path / "missing.env"),
            "/lane/bifrost-dir",
            "/lane/declaration",
            "-p",
            "lane-project",
            "up",
        ],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}"},
    )
    assert (
        done.stdout == "/lane/bifrost-dir|/lane/declaration|compose -p lane-project up"
    )


def test_the_host_passes_its_injected_lane_dirs_to_the_recreate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: Sequence[str], *, cwd: Path | None = None) -> str:
        calls.append(list(argv))
        if argv[:2] == ["docker", "ps"]:
            return "container-1"
        if argv[:2] == ["docker", "inspect"]:
            return (
                '{"com.docker.compose.project.config_files": "/c/a.yml",'
                ' "com.docker.compose.project.working_dir": "/c"}'
            )
        return ""

    monkeypatch.setattr(module, "_run", fake_run)
    host = _host(tmp_path)
    host._recreate_on("proj", (("svc", "sha256:new"),))
    script = next(argv for argv in calls if argv[:2] == ["bash", "-c"])
    positional = script[3:]
    assert positional[0] == "recreate"
    assert positional[3] == str(tmp_path / "config" / "lab" / "lane-overlays")
    assert positional[4] == str(tmp_path / "config" / "lab")
