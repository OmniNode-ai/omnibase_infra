# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration


class TestOverlayConfigResolverIntegration:
    """Integration tests for pure overlay resolution."""

    def test_resolve_returns_pairs_without_env_mutation(
        self,
        sample_overlay_yaml: Path,
        contracts_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from omnibase_infra.runtime.overlay.overlay_config_resolver import (
            OverlayConfigResolver,
        )
        from omnibase_infra.runtime.overlay.overlay_file_loader import OverlayFileLoader

        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
        monkeypatch.delenv("POSTGRES_HOST", raising=False)

        overlay = OverlayFileLoader().load(sample_overlay_yaml)
        result = OverlayConfigResolver().resolve(overlay, contracts_dir)

        assert result.resolved["KAFKA_BOOTSTRAP_SERVERS"] == "localhost:9092"
        assert result.resolved["POSTGRES_HOST"] == "localhost"
        assert "KAFKA_BOOTSTRAP_SERVERS" not in os.environ
        assert "POSTGRES_HOST" not in os.environ

    def test_apply_to_environment_identifies_skipped_existing_keys(
        self,
        sample_overlay_yaml: Path,
        contracts_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from omnibase_infra.runtime.overlay.overlay_config_resolver import (
            OverlayConfigResolver,
        )
        from omnibase_infra.runtime.overlay.overlay_file_loader import OverlayFileLoader

        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "already-set:9092")
        overlay = OverlayFileLoader().load(sample_overlay_yaml)
        result = OverlayConfigResolver().resolve(overlay, contracts_dir)
        injection = result.apply_to_environment()

        assert "KAFKA_BOOTSTRAP_SERVERS" in injection.skipped_existing_keys
        assert "KAFKA_BOOTSTRAP_SERVERS" not in injection.injected_keys
        assert os.environ["KAFKA_BOOTSTRAP_SERVERS"] == "already-set:9092"

    def test_resolve_incomplete_overlay_returns_available_pairs(
        self, overlay_dir: Path, contracts_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omnibase_infra.runtime.overlay.overlay_config_resolver import (
            OverlayConfigResolver,
        )
        from omnibase_infra.runtime.overlay.overlay_file_loader import OverlayFileLoader

        monkeypatch.delenv("POSTGRES_HOST", raising=False)
        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)

        overlay_dir.mkdir(parents=True, exist_ok=True)
        incomplete = overlay_dir / "incomplete.yaml"
        incomplete.write_text(
            yaml.safe_dump(
                {
                    "overlay_version": "1.0.0",
                    "environment": "test",
                    "scope": "env",
                    "transports": {"kafka": {"KAFKA_BOOTSTRAP_SERVERS": "host:9092"}},
                },
                sort_keys=True,
            )
        )
        incomplete.chmod(0o600)

        overlay = OverlayFileLoader().load(incomplete)
        result = OverlayConfigResolver().resolve(overlay, contracts_dir)
        assert result.resolved == {"KAFKA_BOOTSTRAP_SERVERS": "host:9092"}
        assert result.missing == ()

    def test_resolve_includes_all_overlay_keys(
        self,
        sample_overlay_yaml: Path,
        contracts_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from omnibase_infra.runtime.overlay.overlay_config_resolver import (
            OverlayConfigResolver,
        )
        from omnibase_infra.runtime.overlay.overlay_file_loader import OverlayFileLoader

        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
        monkeypatch.delenv("POSTGRES_HOST", raising=False)

        overlay = OverlayFileLoader().load(sample_overlay_yaml)
        result = OverlayConfigResolver().resolve(overlay, contracts_dir)

        assert "POSTGRES_PORT" in result.resolved
        assert "POSTGRES_PASSWORD" in result.resolved

    def test_resolve_is_deterministic_for_identical_inputs(
        self,
        sample_overlay_yaml: Path,
        contracts_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Replay determinism: same inputs -> same resolved values and hash."""
        from omnibase_infra.runtime.overlay.overlay_config_resolver import (
            OverlayConfigResolver,
        )
        from omnibase_infra.runtime.overlay.overlay_file_loader import OverlayFileLoader

        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
        monkeypatch.delenv("POSTGRES_HOST", raising=False)

        overlay = OverlayFileLoader().load(sample_overlay_yaml)
        resolver = OverlayConfigResolver()

        result1 = resolver.resolve(overlay, contracts_dir)
        result2 = resolver.resolve(overlay, contracts_dir)
        assert result1.resolved == result2.resolved
        assert (
            result1.manifest.resolved_config_hash
            == result2.manifest.resolved_config_hash
        )
