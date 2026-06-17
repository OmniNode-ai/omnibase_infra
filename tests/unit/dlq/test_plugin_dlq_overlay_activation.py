# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for PluginDlq overlay-based activation (OMN-12634).

Verifies that PluginDlq.should_activate() resolves DLQ_ENABLED and DLQ_DB_URL
from config.overlay_config instead of environment variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest


@dataclass
class _MinimalPluginConfig:
    """Minimal stand-in for ModelDomainPluginConfig — only fields PluginDlq reads."""

    correlation_id: Any
    overlay_config: dict[str, str] | None = None


def _make_config(overlay: dict[str, str] | None = None) -> _MinimalPluginConfig:
    return _MinimalPluginConfig(correlation_id=uuid4(), overlay_config=overlay)


class TestPluginDlqOverlayActivation:
    """PluginDlq.should_activate() must read from overlay_config, not env vars."""

    def test_activates_when_overlay_enabled_and_db_url_set(self) -> None:
        from omnibase_infra.dlq.plugin_dlq import PluginDlq

        plugin = PluginDlq()
        config = _make_config(
            {"DLQ_ENABLED": "true", "DLQ_DB_URL": "postgresql://localhost/dlq"}
        )
        assert plugin.should_activate(config) is True  # type: ignore[arg-type]

    def test_does_not_activate_when_dlq_disabled_in_overlay(self) -> None:
        from omnibase_infra.dlq.plugin_dlq import PluginDlq

        plugin = PluginDlq()
        config = _make_config(
            {"DLQ_ENABLED": "false", "DLQ_DB_URL": "postgresql://localhost/dlq"}
        )
        assert plugin.should_activate(config) is False  # type: ignore[arg-type]

    def test_does_not_activate_when_dlq_key_absent_from_overlay(self) -> None:
        from omnibase_infra.dlq.plugin_dlq import PluginDlq

        plugin = PluginDlq()
        # overlay present but DLQ_ENABLED key missing → treat as disabled
        config = _make_config({"DLQ_DB_URL": "postgresql://localhost/dlq"})
        assert plugin.should_activate(config) is False  # type: ignore[arg-type]

    def test_raises_when_enabled_but_db_url_missing_from_overlay(self) -> None:
        from omnibase_infra.dlq.plugin_dlq import PluginDlq

        plugin = PluginDlq()
        config = _make_config({"DLQ_ENABLED": "true"})
        # fail-fast: overlay says enabled but DLQ_DB_URL not provided
        with pytest.raises(ValueError, match="DLQ_DB_URL"):
            plugin.should_activate(config)  # type: ignore[arg-type]

    def test_does_not_activate_when_overlay_config_is_none(self) -> None:
        from omnibase_infra.dlq.plugin_dlq import PluginDlq

        plugin = PluginDlq()
        config = _make_config(overlay=None)
        assert plugin.should_activate(config) is False  # type: ignore[arg-type]

    @pytest.mark.parametrize("truthy", ["true", "1", "yes", "True", "YES"])
    def test_activates_for_all_truthy_values(self, truthy: str) -> None:
        from omnibase_infra.dlq.plugin_dlq import PluginDlq

        plugin = PluginDlq()
        config = _make_config(
            {"DLQ_ENABLED": truthy, "DLQ_DB_URL": "postgresql://localhost/dlq"}
        )
        assert plugin.should_activate(config) is True  # type: ignore[arg-type]

    def test_dsn_stored_after_activation(self) -> None:
        from omnibase_infra.dlq.plugin_dlq import PluginDlq

        dsn = "postgresql://user:pass@host/dlq"
        plugin = PluginDlq()
        config = _make_config({"DLQ_ENABLED": "true", "DLQ_DB_URL": dsn})
        plugin.should_activate(config)  # type: ignore[arg-type]
        assert plugin._dsn == dsn

    def test_env_vars_are_not_read_when_overlay_provided(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env vars must NOT be consulted when overlay_config is present."""
        from omnibase_infra.dlq.plugin_dlq import PluginDlq

        # Set env vars to the opposite of overlay so any env-var read would fail
        monkeypatch.setenv("OMNIBASE_INFRA_DLQ_ENABLED", "true")
        monkeypatch.setenv("OMNIBASE_INFRA_DB_URL", "postgresql://env/wrong")

        plugin = PluginDlq()
        # overlay says disabled → must return False regardless of env
        config = _make_config(
            {"DLQ_ENABLED": "false", "DLQ_DB_URL": "postgresql://env/wrong"}
        )
        assert plugin.should_activate(config) is False  # type: ignore[arg-type]


class TestModelDomainPluginConfigOverlayField:
    """ModelDomainPluginConfig must carry overlay_config without breaking existing usage."""

    def test_overlay_config_defaults_to_none(self) -> None:
        from uuid import uuid4

        from omnibase_infra.event_bus import EventBusInmemory
        from omnibase_infra.runtime.models import ModelDomainPluginConfig

        mock_container = MagicMock()
        mock_bus = MagicMock(spec=EventBusInmemory)
        config = ModelDomainPluginConfig(
            container=mock_container,
            event_bus=mock_bus,
            correlation_id=uuid4(),
            input_topic="in",
            output_topic="out",
            consumer_group="g",
        )
        assert config.overlay_config is None

    def test_overlay_config_can_be_set(self) -> None:
        from uuid import uuid4

        from omnibase_infra.event_bus import EventBusInmemory
        from omnibase_infra.runtime.models import ModelDomainPluginConfig

        mock_container = MagicMock()
        mock_bus = MagicMock(spec=EventBusInmemory)
        overlay = {"DLQ_ENABLED": "true", "DLQ_DB_URL": "postgresql://localhost/dlq"}
        config = ModelDomainPluginConfig(
            container=mock_container,
            event_bus=mock_bus,
            correlation_id=uuid4(),
            input_topic="in",
            output_topic="out",
            consumer_group="g",
            overlay_config=overlay,
        )
        assert config.overlay_config == overlay
