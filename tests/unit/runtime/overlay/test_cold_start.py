# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for cold-start detection paths in load_overlay_config()."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.runtime.overlay.boot_overlay import load_overlay_config
from omnibase_infra.runtime.overlay.errors import OverlayNotFoundError

_SENTINEL_KEYS = (
    "POSTGRES_HOST",
    "KAFKA_BOOTSTRAP_SERVERS",
    "VALKEY_HOST",
    "POSTGRES_PASSWORD",
)


@pytest.mark.unit
class TestColdStart:
    def test_no_overlay_no_env_raises_cold_start(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for key in _SENTINEL_KEYS:
            monkeypatch.delenv(key, raising=False)
        with pytest.raises(OverlayNotFoundError, match="onboarding"):
            load_overlay_config(
                overlay_path=tmp_path / "missing.yaml",
                contracts_dir=tmp_path,
                require_overlay=False,
            )

    def test_no_overlay_env_present_returns_none_compat(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("POSTGRES_HOST", "legacy-host")
        result = load_overlay_config(
            overlay_path=tmp_path / "missing.yaml",
            contracts_dir=tmp_path,
            require_overlay=False,
        )
        assert result is None

    def test_require_overlay_true_overrides_compat(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("POSTGRES_HOST", "legacy")
        with pytest.raises(OverlayNotFoundError):
            load_overlay_config(
                overlay_path=tmp_path / "missing.yaml",
                contracts_dir=tmp_path,
                require_overlay=True,
            )

    def test_overlay_present_succeeds(self, tmp_path: Path) -> None:
        p = tmp_path / "overlay.yaml"
        p.write_text("overlay_version: '1.0.0'\nenvironment: dev\nscope: env\n")
        result = load_overlay_config(
            overlay_path=p,
            contracts_dir=tmp_path,
            require_overlay=False,
        )
        assert result is not None
        assert result.manifest.config_source == "overlay"
