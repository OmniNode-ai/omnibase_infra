# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24032: the resolver honours ONEX_SKILL_OVERLAY_ROOTS over the workspace config root.

``onex run-node`` resolves a node's broker binding through the installed entry point. When the
``ONEX_SKILL_OVERLAY_ROOTS`` environment variable names a directory tree, the resolver looks for
``<node>/overlay.yaml`` under each root before falling back to the workspace config root's
``config/onex/overlays/<node>/overlay.yaml``. These tests take no patch of the entry-point lookup:
they go through the installed distribution metadata, then through
``ModelNodeBrokerBinding.for_node``, the call ``onex run-node`` makes, over real overlay files in a
scratch overlay root and a scratch workspace config root.

Golden chain: an in-memory overlay under the scratch overlay root resolves through the installed
registration even though the workspace config root holds no overlay for the node. Control: with the
overlay roots unset the same call is refused naming the merge-base overlay path. Error chain: a
malformed overlay under the root is refused naming that file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_core.enums.enum_event_bus_type import EnumEventBusType
from omnibase_core.models.event_bus.model_node_broker_binding import (
    ModelNodeBrokerBinding,
)
from omnibase_core.models.event_bus.model_node_broker_resolution_error import (
    ModelNodeBrokerResolutionError,
)

_NODE = "node_overlay_roots_probe_effect"


@pytest.fixture
def config_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A workspace config root with a lane declaration, selected the way a deployment selects it."""
    for name in (
        "KAFKA_BOOTSTRAP_SERVERS",
        "KAFKA_SASL_USERNAME",
        "KAFKA_SASL_PASSWORD",
        "OMNIBASE_PATH",
        "OMNI_HOME",
    ):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "config-owner"
    declaration = root / "config" / "lab" / "ci_bus_lanes.yaml"
    declaration.parent.mkdir(parents=True)
    declaration.write_text(
        yaml.safe_dump(
            {
                "lanes": {
                    "plain": {
                        "broker": "installed-lane-broker.test:19092",
                        "security_protocol": "PLAINTEXT",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(root))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("OMNIBASE_PATH", str(tmp_path / "workspace"))
    return root


def _declare_overlay_root(root: Path, body: object) -> Path:
    """Write ``<node>/overlay.yaml`` under a scratch overlay root and return the file path."""
    path = root / _NODE / "overlay.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def test_overlay_root_inmemory_overlay_resolves_over_a_config_root_with_no_overlay(
    config_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlay_root = tmp_path / "overlay-root"
    overlay = _declare_overlay_root(overlay_root, {"event_bus": {"type": "inmemory"}})
    monkeypatch.setenv("ONEX_SKILL_OVERLAY_ROOTS", str(overlay_root))

    binding = ModelNodeBrokerBinding.for_node(_NODE)

    assert binding.bus_type == EnumEventBusType.INMEMORY
    assert binding.overlay_path == overlay


def test_without_overlay_roots_the_call_is_refused_naming_the_merge_base_overlay_path(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ONEX_SKILL_OVERLAY_ROOTS", raising=False)

    with pytest.raises(ModelNodeBrokerResolutionError) as refused:
        ModelNodeBrokerBinding.for_node(_NODE)

    assert f"config/onex/overlays/{_NODE}/overlay.yaml" in str(refused.value)


def test_malformed_overlay_under_the_root_is_refused_naming_that_file(
    config_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlay_root = tmp_path / "overlay-root"
    overlay = _declare_overlay_root(overlay_root, {"event_bus": {"type": "kafka"}})
    monkeypatch.setenv("ONEX_SKILL_OVERLAY_ROOTS", str(overlay_root))

    with pytest.raises(ModelNodeBrokerResolutionError) as refused:
        ModelNodeBrokerBinding.for_node(_NODE)

    assert str(overlay) in str(refused.value)
