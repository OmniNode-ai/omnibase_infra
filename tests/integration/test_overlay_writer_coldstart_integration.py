# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for OverlayWriter + cold-start pipeline.

Exercises: write → load → resolve → inject → verify env values land correctly.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from omnibase_core.models.overlay.model_overlay_file import ModelOverlayFile
from omnibase_infra.runtime.overlay.boot_overlay import load_overlay_config
from omnibase_infra.runtime.overlay.errors import OverlayNotFoundError
from omnibase_infra.runtime.overlay.overlay_file_loader import OverlayFileLoader
from omnibase_infra.runtime.overlay.overlay_writer import OverlayWriter


@pytest.mark.integration
class TestOverlayWriterColdStartIntegration:
    def test_write_load_content_hash_stable(self, tmp_path: Path) -> None:
        overlay = ModelOverlayFile.model_validate(
            {
                "overlay_version": "1.0.0",
                "environment": "dev",
                "scope": "env",
                "transports": {
                    "database": {
                        "POSTGRES_HOST": "db.test.local",
                        "POSTGRES_PORT": "5432",
                    },
                    "kafka": {
                        "KAFKA_BOOTSTRAP_SERVERS": "kafka.test.local:9092"
                    },  # kafka-fallback-ok
                },
            }
        )
        target = tmp_path / "overlay.yaml"
        OverlayWriter().write(overlay, target)

        loaded = OverlayFileLoader().load(target)
        assert loaded.content_hash() == overlay.content_hash()
        assert loaded.transports["database"]["POSTGRES_HOST"] == "db.test.local"
        assert (
            loaded.transports["kafka"]["KAFKA_BOOTSTRAP_SERVERS"]
            == "kafka.test.local:9092"
        )

    def test_write_sets_restricted_permissions(self, tmp_path: Path) -> None:
        overlay = ModelOverlayFile.model_validate(
            {"overlay_version": "1.0.0", "environment": "dev", "scope": "env"}
        )
        target = tmp_path / "secure.yaml"
        OverlayWriter().write(overlay, target)

        assert stat.S_IMODE(target.stat().st_mode) == 0o600

    def test_cold_start_overlay_present_resolves(self, tmp_path: Path) -> None:
        overlay = ModelOverlayFile.model_validate(
            {
                "overlay_version": "1.0.0",
                "environment": "integration-test",
                "scope": "env",
                "transports": {"database": {"INTEGRATION_TEST_HOST": "db.test"}},
            }
        )
        overlay_path = tmp_path / "overlay.yaml"
        OverlayWriter().write(overlay, overlay_path)

        result = load_overlay_config(
            overlay_path=overlay_path,
            contracts_dir=tmp_path,
            require_overlay=False,
        )
        assert result is not None
        assert result.manifest.config_source == "overlay"
        assert result.manifest.overlay_file_hash.startswith("sha256:")

    def test_cold_start_missing_overlay_no_env_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for key in (
            "POSTGRES_HOST",
            "KAFKA_BOOTSTRAP_SERVERS",
            "VALKEY_HOST",
            "POSTGRES_PASSWORD",
        ):
            monkeypatch.delenv(key, raising=False)

        with pytest.raises(OverlayNotFoundError, match="onboarding"):
            load_overlay_config(
                overlay_path=tmp_path / "nonexistent.yaml",
                contracts_dir=tmp_path,
                require_overlay=False,
            )

    def test_require_overlay_enforced_over_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("POSTGRES_HOST", "legacy-host")

        with pytest.raises(OverlayNotFoundError):
            load_overlay_config(
                overlay_path=tmp_path / "nonexistent.yaml",
                contracts_dir=tmp_path,
                require_overlay=True,
            )

    def test_double_write_is_idempotent(self, tmp_path: Path) -> None:
        overlay = ModelOverlayFile.model_validate(
            {
                "overlay_version": "1.0.0",
                "environment": "dev",
                "scope": "env",
                "transports": {"database": {"POSTGRES_HOST": "localhost"}},
            }
        )
        target = tmp_path / "overlay.yaml"
        writer = OverlayWriter()
        writer.write(overlay, target)
        hash1 = OverlayFileLoader().load(target).content_hash()
        writer.write(overlay, target)
        hash2 = OverlayFileLoader().load(target).content_hash()
        assert hash1 == hash2
