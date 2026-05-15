# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for load_overlay_config() boot helper."""

from __future__ import annotations

import pytest

from omnibase_infra.runtime.overlay.boot_overlay import load_overlay_config
from omnibase_infra.runtime.overlay.errors import OverlayNotFoundError

_VALID_OVERLAY = (
    "overlay_version: '1.0.0'\n"
    "environment: dev\n"
    "scope: env\n"
    "transports:\n"
    "  database:\n"
    "    POSTGRES_HOST: overlay-host\n"
)


@pytest.mark.unit
class TestBootOverlay:
    def test_overlay_present_resolves_and_returns_manifest(self, tmp_path):
        overlay_path = tmp_path / "overlay.yaml"
        overlay_path.write_text(_VALID_OVERLAY)
        contracts_dir = tmp_path / "contracts"
        contracts_dir.mkdir()

        result = load_overlay_config(
            overlay_path=overlay_path,
            contracts_dir=contracts_dir,
            require_overlay=False,
        )

        assert result is not None
        assert result.manifest.config_source == "overlay"

    def test_overlay_missing_env_present_returns_none_with_warning(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("POSTGRES_HOST", "legacy-host")

        result = load_overlay_config(
            overlay_path=tmp_path / "missing.yaml",
            contracts_dir=tmp_path,
            require_overlay=False,
        )

        assert result is None

    def test_overlay_missing_no_env_raises_with_onboarding_message(
        self, tmp_path, monkeypatch
    ):
        for key in (
            "POSTGRES_HOST",
            "KAFKA_BOOTSTRAP_SERVERS",
            "VALKEY_HOST",
            "POSTGRES_PASSWORD",
        ):
            monkeypatch.delenv(key, raising=False)

        with pytest.raises(OverlayNotFoundError, match="onboarding"):
            load_overlay_config(
                overlay_path=tmp_path / "missing.yaml",
                contracts_dir=tmp_path,
                require_overlay=False,
            )

    def test_require_overlay_true_fails_even_with_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("POSTGRES_HOST", "legacy")

        with pytest.raises(OverlayNotFoundError):
            load_overlay_config(
                overlay_path=tmp_path / "missing.yaml",
                contracts_dir=tmp_path,
                require_overlay=True,
            )

    def test_overlay_present_resolved_keys_accessible(self, tmp_path):
        overlay_path = tmp_path / "overlay.yaml"
        overlay_path.write_text(_VALID_OVERLAY)

        result = load_overlay_config(
            overlay_path=overlay_path,
            contracts_dir=tmp_path,
            require_overlay=False,
        )

        assert result is not None
        assert result.resolved.get("POSTGRES_HOST") == "overlay-host"
