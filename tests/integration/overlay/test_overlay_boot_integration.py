# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


class TestOverlayBootIntegration:
    """Integration tests for boot_overlay — validates that boot_overlay.py
    is the SOLE authority for os.environ mutation during overlay loading."""

    def test_load_overlay_config_injects_env(
        self,
        sample_overlay_yaml: Path,
        contracts_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from omnibase_infra.runtime.overlay.boot_overlay import load_overlay_config

        monkeypatch.setenv("ONEX_OVERLAY_PATH", str(sample_overlay_yaml))
        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
        monkeypatch.delenv("POSTGRES_HOST", raising=False)

        result = load_overlay_config(contracts_dir=contracts_dir)
        assert result is not None
        # boot_overlay is where env mutation happens
        assert os.environ["KAFKA_BOOTSTRAP_SERVERS"] == "localhost:9092"
        assert os.environ["POSTGRES_HOST"] == "localhost"

    def test_load_overlay_config_writes_manifest(
        self,
        sample_overlay_yaml: Path,
        contracts_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from omnibase_infra.runtime.overlay.boot_overlay import load_overlay_config

        manifest_path = tmp_path / "overlay_resolution_manifest.json"
        monkeypatch.setenv("ONEX_OVERLAY_PATH", str(sample_overlay_yaml))
        monkeypatch.setenv("ONEX_OVERLAY_MANIFEST_PATH", str(manifest_path))
        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)

        load_overlay_config(contracts_dir=contracts_dir)
        assert manifest_path.exists()
        import json

        manifest = json.loads(manifest_path.read_text())
        assert "stable_identity_hash" in manifest
        assert "resolved_pairs_hash" in manifest

    def test_missing_overlay_with_env_vars_returns_none(
        self, contracts_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Migration mode: no overlay file but env vars present → returns None."""
        from omnibase_infra.runtime.overlay.boot_overlay import load_overlay_config

        monkeypatch.setenv("ONEX_OVERLAY_PATH", "/nonexistent/overlay.yaml")
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "existing:9092")
        # Ensure ONEX_REQUIRE_OVERLAY does not leak from the ambient env or a
        # prior test; if true it would override migration mode and raise.
        monkeypatch.delenv("ONEX_REQUIRE_OVERLAY", raising=False)
        result = load_overlay_config(contracts_dir=contracts_dir)
        assert result is None

    def test_missing_overlay_no_env_raises(
        self, contracts_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Greenfield: no overlay, no env vars → error with onboarding instructions."""
        from omnibase_infra.runtime.overlay.boot_overlay import load_overlay_config
        from omnibase_infra.runtime.overlay.errors import OverlayNotFoundError

        monkeypatch.setenv("ONEX_OVERLAY_PATH", str(tmp_path / "missing.yaml"))
        # Clear all transport indicator vars so _has_transport_env_vars() returns False
        for var in (
            "KAFKA_BOOTSTRAP_SERVERS",
            "POSTGRES_HOST",
            "POSTGRES_PASSWORD",
            "VALKEY_URL",
            "INFISICAL_ADDR",
        ):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(OverlayNotFoundError):
            load_overlay_config(contracts_dir=contracts_dir)

    def test_require_overlay_overrides_migration_mode(
        self, contracts_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ONEX_REQUIRE_OVERLAY=true raises even when env vars exist."""
        from omnibase_infra.runtime.overlay.boot_overlay import load_overlay_config
        from omnibase_infra.runtime.overlay.errors import OverlayNotFoundError

        monkeypatch.setenv("ONEX_OVERLAY_PATH", "/nonexistent/overlay.yaml")
        monkeypatch.setenv("ONEX_REQUIRE_OVERLAY", "true")
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "exists:9092")
        with pytest.raises(OverlayNotFoundError, match="ONEX_REQUIRE_OVERLAY=true"):
            load_overlay_config(contracts_dir=contracts_dir)
