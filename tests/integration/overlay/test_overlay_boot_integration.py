# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


class TestOverlayBootIntegration:
    """Integration tests for overlay load, resolve, and explicit env injection."""

    def test_load_overlay_config_resolves_env_pairs(
        self,
        sample_overlay_yaml: Path,
        contracts_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from omnibase_infra.runtime.overlay.boot_overlay import load_overlay_config

        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
        monkeypatch.delenv("POSTGRES_HOST", raising=False)

        result = load_overlay_config(
            overlay_path=sample_overlay_yaml,
            contracts_dir=contracts_dir,
        )
        assert result is not None
        assert result.resolved["KAFKA_BOOTSTRAP_SERVERS"] == "localhost:9092"
        assert result.resolved["POSTGRES_HOST"] == "localhost"
        assert "KAFKA_BOOTSTRAP_SERVERS" not in os.environ

        injection = result.apply_to_environment()
        assert "KAFKA_BOOTSTRAP_SERVERS" in injection.injected_keys
        assert os.environ["KAFKA_BOOTSTRAP_SERVERS"] == "localhost:9092"
        assert os.environ["POSTGRES_HOST"] == "localhost"

    def test_load_overlay_config_returns_manifest(
        self,
        sample_overlay_yaml: Path,
        contracts_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from omnibase_infra.runtime.overlay.boot_overlay import load_overlay_config

        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)

        result = load_overlay_config(
            overlay_path=sample_overlay_yaml,
            contracts_dir=contracts_dir,
        )
        assert result is not None
        assert result.manifest.config_source == "overlay"
        assert result.manifest.overlay_file_hash
        assert result.manifest.resolved_config_hash
        assert result.manifest.stable_identity_hash()

    def test_missing_overlay_with_env_vars_returns_none(
        self, contracts_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Migration mode: no overlay file but env vars present -> returns None."""
        from omnibase_infra.runtime.overlay.boot_overlay import load_overlay_config

        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "existing:9092")
        result = load_overlay_config(
            overlay_path=tmp_path / "missing.yaml",
            contracts_dir=contracts_dir,
        )
        assert result is None

    def test_missing_overlay_no_env_raises(
        self, contracts_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Greenfield: no overlay, no env vars -> error with onboarding instructions."""
        from omnibase_infra.runtime.overlay.boot_overlay import load_overlay_config
        from omnibase_infra.runtime.overlay.errors import OverlayNotFoundError

        for var in (
            "KAFKA_BOOTSTRAP_SERVERS",
            "POSTGRES_HOST",
            "POSTGRES_PASSWORD",
            "VALKEY_HOST",
        ):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(OverlayNotFoundError, match="fresh install"):
            load_overlay_config(
                overlay_path=tmp_path / "missing.yaml",
                contracts_dir=contracts_dir,
            )

    def test_require_overlay_overrides_migration_mode(
        self, contracts_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """require_overlay=True raises even when legacy env vars exist."""
        from omnibase_infra.runtime.overlay.boot_overlay import load_overlay_config
        from omnibase_infra.runtime.overlay.errors import OverlayNotFoundError

        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "exists:9092")
        with pytest.raises(OverlayNotFoundError, match="required but not found"):
            load_overlay_config(
                overlay_path=tmp_path / "missing.yaml",
                contracts_dir=contracts_dir,
                require_overlay=True,
            )
