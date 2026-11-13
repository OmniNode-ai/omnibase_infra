# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24032: a node's overlay is read from the overlay roots before the workspace config root.

The one overlay root resolver lives in omnimarket and renders ``ONEX_SKILL_OVERLAY_ROOTS`` for a unit
or a lane; this handler reads that selector like every other overlay reader and builds no list of its
own, so none of these tests imports omnimarket. Each falsifier would fail at the merge base, where
``onex run-node`` read only the workspace config root and refused a node whose overlay lives in the
overlays repository:

* an overlay under the first provided root resolves to a binding even with no workspace root;
* when two roots both hold the overlay, the first root's file is the binding's overlay path;
* with no roots and no workspace overlay the old refusal names the workspace overlay path;
* a malformed ``event_bus`` in a roots overlay is refused naming that file;
* the retired inside-the-registry path is not looked at; and
* with the selector unset the old refusal stands.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from omnibase_core.enums.enum_event_bus_type import EnumEventBusType
from omnibase_core.models.event_bus.model_node_broker_resolution_error import (
    ModelNodeBrokerResolutionError,
)
from omnibase_infra.handlers.handler_node_broker_resolve import (
    HandlerNodeBrokerResolve,
)
from omnibase_infra.models.workspace import ModelNodeBrokerResolveRequest

pytestmark = pytest.mark.unit

_NODE = "node_broker_acl_overlay_render_compute"
_ENV_NAMES = ("OMNIBASE_PATH", "OMNI_HOME", "ONEX_SKILL_OVERLAY_ROOTS")


def _overlay_at(root: Path, body: object) -> Path:
    path = root / _NODE / "overlay.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def _handler(
    monkeypatch: pytest.MonkeyPatch, roots: tuple[str, ...]
) -> HandlerNodeBrokerResolve:
    """The real handler with the selector set to ``roots``, as a unit or a lane declares it."""
    monkeypatch.setenv("ONEX_SKILL_OVERLAY_ROOTS", os.pathsep.join(roots))
    return HandlerNodeBrokerResolve()


def test_overlay_under_the_first_root_resolves_with_no_workspace_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "store-root"
    overlay = _overlay_at(root, {"event_bus": {"type": "inmemory"}})

    binding = _handler(monkeypatch, (str(root),)).handle(
        ModelNodeBrokerResolveRequest(overlay_key=_NODE, workspace_root=None)
    )

    assert binding.bus_type == EnumEventBusType.INMEMORY
    assert binding.overlay_path == overlay


def test_the_first_root_holding_the_overlay_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first-root"
    second = tmp_path / "second-root"
    overlay = _overlay_at(first, {"event_bus": {"type": "inmemory"}})
    _overlay_at(second, {"event_bus": {"type": "inmemory"}})

    binding = _handler(monkeypatch, (str(first), str(second))).handle(
        ModelNodeBrokerResolveRequest(overlay_key=_NODE, workspace_root=None)
    )

    assert binding.overlay_path == overlay


def test_no_roots_and_no_workspace_overlay_is_refused_naming_the_workspace_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    config_root = tmp_path / "config-owner"
    declaration = config_root / "config" / "lab" / "ci_bus_lanes.yaml"
    declaration.parent.mkdir(parents=True)
    declaration.write_text(
        yaml.safe_dump({"lanes": {"plain": {"broker": "lane-broker.test:19092"}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(config_root))
    monkeypatch.setenv("OMNIBASE_PATH", str(config_root))
    expected = config_root / "config" / "onex" / "overlays" / _NODE / "overlay.yaml"

    with pytest.raises(ModelNodeBrokerResolutionError) as raised:
        _handler(monkeypatch, ()).handle(
            ModelNodeBrokerResolveRequest(overlay_key=_NODE, workspace_root=config_root)
        )

    assert raised.value.overlay_path == str(expected)
    assert str(expected) in raised.value.message


def test_malformed_event_bus_in_a_roots_overlay_is_refused_naming_that_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "store-root"
    overlay = _overlay_at(root, {"event_bus": {"type": "kafka"}})

    with pytest.raises(ModelNodeBrokerResolutionError) as raised:
        _handler(monkeypatch, (str(root),)).handle(
            ModelNodeBrokerResolveRequest(overlay_key=_NODE, workspace_root=None)
        )

    assert raised.value.overlay_path == str(overlay)


def test_the_retired_inside_the_registry_path_is_not_looked_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    registry = tmp_path / "registry"
    declaration = registry / "config" / "lab" / "ci_bus_lanes.yaml"
    declaration.parent.mkdir(parents=True)
    declaration.write_text(
        yaml.safe_dump({"lanes": {"plain": {"broker": "lane-broker.test:19092"}}}),
        encoding="utf-8",
    )
    _overlay_at(registry / "overlays" / "overlays", {"event_bus": {"type": "inmemory"}})
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(registry))
    monkeypatch.setenv("OMNIBASE_PATH", str(registry))
    expected = registry / Path("config", "onex", "overlays") / _NODE / "overlay.yaml"

    with pytest.raises(ModelNodeBrokerResolutionError) as raised:
        _handler(monkeypatch, ()).handle(
            ModelNodeBrokerResolveRequest(overlay_key=_NODE, workspace_root=registry)
        )

    assert "the file does not exist" in raised.value.message
    assert raised.value.overlay_path == str(expected)
    assert "overlays/overlays" not in raised.value.overlay_path


def test_a_root_without_the_overlay_falls_through_to_the_workspace_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declared root that holds no file for the node is skipped; the workspace overlay is the last source."""
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    empty_root = tmp_path / "empty-root"
    empty_root.mkdir()
    config_root = tmp_path / "config-owner"
    declaration = config_root / "config" / "lab" / "ci_bus_lanes.yaml"
    declaration.parent.mkdir(parents=True)
    declaration.write_text(
        yaml.safe_dump({"lanes": {"plain": {"broker": "lane-broker.test:19092"}}}),
        encoding="utf-8",
    )
    workspace_overlay = _overlay_at(
        config_root / "config" / "onex" / "overlays",
        {"event_bus": {"type": "inmemory"}},
    )
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(config_root))

    binding = _handler(monkeypatch, (str(empty_root),)).handle(
        ModelNodeBrokerResolveRequest(overlay_key=_NODE, workspace_root=config_root)
    )

    assert binding.overlay_path == workspace_overlay
