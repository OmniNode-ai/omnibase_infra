# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test: overlay loading against real contract files from this repo.

Task 11 per the OMN-11069 overlay config implementation plan.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from omnibase_core.models.overlay.model_overlay_file import ModelOverlayFile
from omnibase_infra.runtime.overlay.boot_overlay import load_overlay_config
from omnibase_infra.runtime.overlay.errors import OverlayNotFoundError
from omnibase_infra.runtime.overlay.overlay_writer import OverlayWriter

# Real contracts directory from this repo — used to exercise overlay loading
# against actual contract declarations, not synthetic fixtures.
_CONTRACTS_DIR = (
    Path(__file__).parent.parent.parent.parent.parent / "src" / "omnibase_infra"
)


@pytest.mark.integration
class TestOverlayRuntimeIntegration:
    def test_boot_overlay_with_real_contracts(self, tmp_path: Path) -> None:
        """Exercise overlay loading against real contract files from this repo."""
        # 1. Generate overlay with values for transports that real contracts declare
        overlay = ModelOverlayFile.model_validate(
            {
                "overlay_version": "1.0.0",
                "environment": "dev",
                "scope": "env",
                "transports": {
                    "db": {
                        "POSTGRES_HOST": "192.168.86.201",
                        "POSTGRES_PORT": "5436",
                        "POSTGRES_USER": "postgres",
                    },
                    "kafka": {
                        "KAFKA_BOOTSTRAP_SERVERS": "192.168.86.201:19092",  # kafka-fallback-ok
                    },
                    "valkey": {
                        "VALKEY_HOST": "192.168.86.201",
                        "VALKEY_PORT": "16379",
                    },
                },
            }
        )

        # 2. Write to tmp_path/overlay.yaml
        overlay_path = tmp_path / "overlay.yaml"
        OverlayWriter().write(overlay, overlay_path)

        # 3. Call load_overlay_config with real contracts_dir
        assert _CONTRACTS_DIR.exists(), (
            f"Expected contracts dir at {_CONTRACTS_DIR} — run from the repo root."
        )

        result = load_overlay_config(
            overlay_path=overlay_path,
            contracts_dir=_CONTRACTS_DIR,
            require_overlay=False,
        )

        # 4. Assert manifest produced
        assert result is not None, "Expected resolved result, not None"
        assert result.manifest.config_source == "overlay"
        assert result.manifest.overlay_file_hash.startswith("sha256:")

        # 5. Overlay transports present in resolved keys
        assert "POSTGRES_HOST" in result.resolved
        assert "KAFKA_BOOTSTRAP_SERVERS" in result.resolved

        # 6. The manifest records resolved_transports from the overlay
        assert len(result.manifest.resolved_transports) > 0

    def test_cold_start_with_real_contracts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No overlay + no env → OverlayNotFoundError."""
        # Remove legacy sentinel env vars so cold-start path is taken
        for key in (
            "POSTGRES_HOST",
            "KAFKA_BOOTSTRAP_SERVERS",
            "VALKEY_HOST",
            "POSTGRES_PASSWORD",
        ):
            monkeypatch.delenv(key, raising=False)

        assert _CONTRACTS_DIR.exists(), f"Expected contracts dir at {_CONTRACTS_DIR}"

        with pytest.raises(OverlayNotFoundError, match="onboarding"):
            load_overlay_config(
                overlay_path=tmp_path / "missing.yaml",
                contracts_dir=_CONTRACTS_DIR,
                require_overlay=False,
            )

    def test_compat_mode_with_real_contracts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No overlay + env present → returns None (compat)."""
        # Set a legacy sentinel key so compat path is taken
        monkeypatch.setenv("POSTGRES_HOST", "legacy-host-compat")

        assert _CONTRACTS_DIR.exists(), f"Expected contracts dir at {_CONTRACTS_DIR}"

        result = load_overlay_config(
            overlay_path=tmp_path / "missing.yaml",
            contracts_dir=_CONTRACTS_DIR,
            require_overlay=False,
        )
        assert result is None, (
            "Expected None in compat mode (no overlay, legacy env present)"
        )
