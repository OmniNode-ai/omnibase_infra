# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Embedded config resolves registered handlers before the runtime is built."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_core.enums.enum_event_bus_type import EnumEventBusType
from omnibase_infra.handlers.handler_workspace_config_root_resolve import (
    HandlerWorkspaceConfigRootResolve,
)
from omnibase_infra.handlers.handler_workspace_runtime_config_materializer import (
    HandlerWorkspaceRuntimeConfigMaterializer,
)
from omnibase_infra.models.workspace import ModelWorkspaceConfigRootResolution
from omnibase_infra.runtime.service_kernel import resolve_embedded_runtime_config

pytestmark = pytest.mark.unit


def _write_config(config_root: Path) -> None:
    runtime = config_root / "config" / "onex" / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "runtime_config.yaml").write_text('event_bus:\n  type: "kafka"\n')


@pytest.mark.asyncio
async def test_workspace_resolution_uses_registered_handlers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ONEX_CONTRACTS_DIR", raising=False)
    # A different ambient root must not replace the registered resolver's answer.
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(tmp_path / "absent"))
    config_root = tmp_path / "owner"
    _write_config(config_root)
    resolver = Mock(spec=HandlerWorkspaceConfigRootResolve)
    resolver.handle.return_value = ModelWorkspaceConfigRootResolution(
        config_root=config_root, source="environment"
    )
    materializer = Mock(spec=HandlerWorkspaceRuntimeConfigMaterializer)
    materializer.read.return_value = None
    container = ModelONEXContainer()
    assert container.service_registry is not None
    await container.service_registry.register_instance(
        HandlerWorkspaceConfigRootResolve, resolver
    )
    await container.service_registry.register_instance(
        HandlerWorkspaceRuntimeConfigMaterializer, materializer
    )

    config, provenance = resolve_embedded_runtime_config(
        workspace_root=tmp_path, container=container
    )

    assert config.event_bus.type is EnumEventBusType.KAFKA
    assert str(config_root / "config" / "onex" / "runtime") in provenance
    resolver.handle.assert_called_once()
    assert resolver.handle.call_args.args[0].workspace_root == tmp_path
    materializer.read.assert_called_once_with(tmp_path)


def test_standalone_workspace_bootstraps_registered_handlers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ONEX_CONTRACTS_DIR", raising=False)
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(tmp_path))
    _write_config(tmp_path)

    config, provenance = resolve_embedded_runtime_config(workspace_root=tmp_path)

    assert config.event_bus.type is EnumEventBusType.KAFKA
    assert str(tmp_path / "config" / "onex" / "runtime") in provenance


@pytest.mark.asyncio
async def test_standalone_workspace_bootstraps_inside_running_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_standalone_workspace_bootstraps_registered_handlers(tmp_path, monkeypatch)
