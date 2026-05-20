# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for overlay_from_env_dict()."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.runtime.overlay.overlay_file_loader import OverlayFileLoader
from omnibase_infra.runtime.overlay.overlay_from_env import overlay_from_env_dict
from omnibase_infra.runtime.overlay.overlay_writer import OverlayWriter


@pytest.mark.unit
class TestOverlayFromEnv:
    def test_classifies_known_transport_keys(self) -> None:
        env_dict = {
            "POSTGRES_HOST": "192.168.86.201",
            "POSTGRES_PORT": "5436",
            "KAFKA_BOOTSTRAP_SERVERS": "192.168.86.201:19092",  # kafka-fallback-ok
        }
        overlay = overlay_from_env_dict(env_dict, environment="dev")
        assert overlay.transports["db"]["POSTGRES_HOST"] == "192.168.86.201"
        assert overlay.transports["db"]["POSTGRES_PORT"] == "5436"
        assert (
            overlay.transports["kafka"]["KAFKA_BOOTSTRAP_SERVERS"]
            == "192.168.86.201:19092"  # kafka-fallback-ok
        )

    def test_unclassified_keys_with_allow_flag(self) -> None:
        env_dict = {"CUSTOM_THING": "value"}
        overlay = overlay_from_env_dict(
            env_dict, environment="dev", allow_unclassified=True
        )
        assert overlay.services.get("unclassified", {}).get("CUSTOM_THING") == "value"

    def test_unclassified_keys_without_flag_warns(self) -> None:
        env_dict = {"CUSTOM_THING": "value"}
        overlay, warnings = overlay_from_env_dict(
            env_dict,
            environment="dev",
            allow_unclassified=False,
            return_warnings=True,
        )
        assert any("CUSTOM_THING" in w for w in warnings)
        assert "unclassified" not in overlay.services

    def test_unclassified_key_omitted_without_allow_flag(self) -> None:
        env_dict = {"CUSTOM_THING": "value"}
        overlay = overlay_from_env_dict(
            env_dict, environment="dev", allow_unclassified=False
        )
        assert not overlay.transports
        assert "unclassified" not in overlay.services

    def test_round_trips_through_writer_loader(self, tmp_path: Path) -> None:
        env_dict = {"POSTGRES_HOST": "localhost"}
        overlay = overlay_from_env_dict(env_dict, environment="dev")
        target = tmp_path / "overlay.yaml"
        OverlayWriter().write(overlay, target)
        loaded = OverlayFileLoader().load(target)
        assert loaded.transports["db"]["POSTGRES_HOST"] == "localhost"

    def test_return_warnings_false_returns_overlay_only(self) -> None:
        env_dict = {"POSTGRES_HOST": "h"}
        result = overlay_from_env_dict(env_dict, environment="dev")
        assert not isinstance(result, tuple)

    def test_return_warnings_true_returns_tuple(self) -> None:
        env_dict = {"POSTGRES_HOST": "h", "MYSTERY_KEY": "x"}
        result = overlay_from_env_dict(
            env_dict, environment="dev", return_warnings=True
        )
        assert isinstance(result, tuple)
        overlay, warnings = result
        assert overlay.transports["db"]["POSTGRES_HOST"] == "h"
        assert any("MYSTERY_KEY" in w for w in warnings)

    def test_mixed_transport_keys_classified_correctly(self) -> None:
        env_dict = {
            "VALKEY_HOST": "192.168.86.201",
            "VALKEY_PORT": "16379",
            "KAFKA_GROUP_ID": "my-group",
            "INFISICAL_ADDR": "https://192.168.86.201:8880",
        }
        overlay = overlay_from_env_dict(env_dict, environment="dev")
        assert "valkey" in overlay.transports
        assert "kafka" in overlay.transports
        assert "infisical" in overlay.transports
        assert overlay.transports["valkey"]["VALKEY_HOST"] == "192.168.86.201"
