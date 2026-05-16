# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path

import pytest
import yaml


class TestOverlayConfigResolverIntegration:
    """Integration tests for OverlayConfigResolver — validates that resolution
    is PURE (no os.environ mutation), filters against contract requirements,
    and enforces RequiredConfigMissingError."""

    def test_resolve_returns_resolved_pairs_without_env_mutation(
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

        loader = OverlayFileLoader()
        overlay = loader.load(sample_overlay_yaml)
        resolver = OverlayConfigResolver(contracts_dir=contracts_dir)
        result = resolver.resolve(overlay)

        # Resolver returns resolved pairs
        assert "KAFKA_BOOTSTRAP_SERVERS" in result.resolved_pairs
        assert "POSTGRES_HOST" in result.resolved_pairs
        assert result.resolved_pairs["KAFKA_BOOTSTRAP_SERVERS"] == "localhost:9092"

        # CRITICAL: resolver does NOT mutate os.environ
        assert "KAFKA_BOOTSTRAP_SERVERS" not in os.environ
        assert "POSTGRES_HOST" not in os.environ

    def test_resolve_identifies_skipped_existing_keys(
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
        loader = OverlayFileLoader()
        overlay = loader.load(sample_overlay_yaml)
        resolver = OverlayConfigResolver(contracts_dir=contracts_dir)
        result = resolver.resolve(overlay)

        assert "KAFKA_BOOTSTRAP_SERVERS" in result.skipped_existing_keys
        assert "KAFKA_BOOTSTRAP_SERVERS" not in result.resolved_pairs

    def test_resolve_raises_when_required_keys_missing(
        self, overlay_dir: Path, contracts_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omnibase_infra.runtime.overlay.errors import RequiredConfigMissingError
        from omnibase_infra.runtime.overlay.overlay_config_resolver import (
            OverlayConfigResolver,
        )
        from omnibase_infra.runtime.overlay.overlay_file_loader import OverlayFileLoader

        monkeypatch.delenv("POSTGRES_HOST", raising=False)
        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)

        # Overlay that only provides KAFKA, not POSTGRES_HOST (which contract requires)
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

        loader = OverlayFileLoader()
        overlay = loader.load(incomplete)
        resolver = OverlayConfigResolver(contracts_dir=contracts_dir)
        with pytest.raises(RequiredConfigMissingError, match="POSTGRES_HOST"):
            resolver.resolve(overlay)

    def test_resolve_tracks_unused_overlay_keys(
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

        loader = OverlayFileLoader()
        overlay = loader.load(sample_overlay_yaml)
        resolver = OverlayConfigResolver(contracts_dir=contracts_dir)
        result = resolver.resolve(overlay)

        # POSTGRES_PORT and POSTGRES_PASSWORD are in overlay but not in contract requirements
        assert (
            "POSTGRES_PORT" in result.unused_overlay_keys
            or "POSTGRES_PASSWORD" in result.unused_overlay_keys
        )

    def test_resolve_is_deterministic_for_identical_inputs(
        self,
        sample_overlay_yaml: Path,
        contracts_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Replay determinism: same inputs → same resolved_pairs regardless of call order."""
        from omnibase_infra.runtime.overlay.overlay_config_resolver import (
            OverlayConfigResolver,
        )
        from omnibase_infra.runtime.overlay.overlay_file_loader import OverlayFileLoader

        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
        monkeypatch.delenv("POSTGRES_HOST", raising=False)

        loader = OverlayFileLoader()
        overlay = loader.load(sample_overlay_yaml)
        resolver = OverlayConfigResolver(contracts_dir=contracts_dir)

        result1 = resolver.resolve(overlay)
        result2 = resolver.resolve(overlay)
        assert result1.resolved_pairs == result2.resolved_pairs
        assert result1.resolved_pairs_hash == result2.resolved_pairs_hash
