# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

import json
import os
from pathlib import Path

import pytest


class TestOverlayEndToEnd:
    """End-to-end: generate overlay via writer → load → resolve → inject → verify manifest.
    Includes replay determinism proof."""

    def test_full_pipeline_generate_load_resolve_inject(
        self, tmp_path: Path, contracts_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omnibase_infra.runtime.overlay.boot_overlay import load_overlay_config
        from omnibase_infra.runtime.overlay.overlay_writer import OverlayWriter

        env_dict = {
            "KAFKA_BOOTSTRAP_SERVERS": "redpanda:9092",
            "POSTGRES_HOST": "pg.internal",
            "POSTGRES_PORT": "5436",
            "POSTGRES_PASSWORD": "secret123",
        }
        writer = OverlayWriter()
        overlay_path = tmp_path / "generated_overlay.yaml"
        writer.write(
            env_dict=env_dict,
            output_path=overlay_path,
            environment="production",
            scope="env",
        )
        assert overlay_path.exists()
        assert (overlay_path.stat().st_mode & 0o777) == 0o600

        monkeypatch.setenv("ONEX_OVERLAY_PATH", str(overlay_path))
        manifest_path = tmp_path / "manifest.json"
        monkeypatch.setenv("ONEX_OVERLAY_MANIFEST_PATH", str(manifest_path))
        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
        monkeypatch.delenv("POSTGRES_HOST", raising=False)

        result = load_overlay_config(contracts_dir=contracts_dir)
        assert result is not None
        assert os.environ["KAFKA_BOOTSTRAP_SERVERS"] == "redpanda:9092"
        assert os.environ["POSTGRES_HOST"] == "pg.internal"

        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["stable_identity_hash"]
        assert manifest["environment"] == "production"
        assert manifest["resolved_pairs_hash"]

    def test_round_trip_content_hash_stable(self, tmp_path: Path) -> None:
        """Writer output → Loader input produces identical content_hash."""
        from omnibase_infra.runtime.overlay.overlay_file_loader import OverlayFileLoader
        from omnibase_infra.runtime.overlay.overlay_writer import OverlayWriter

        env_dict = {"KAFKA_BOOTSTRAP_SERVERS": "host:9092", "POSTGRES_HOST": "db"}
        writer = OverlayWriter()
        path = tmp_path / "overlay.yaml"
        writer.write(
            env_dict=env_dict, output_path=path, environment="test", scope="env"
        )

        loader = OverlayFileLoader(require_restricted_permissions=False)
        model = loader.load(path)
        assert model.content_hash()

        path2 = tmp_path / "overlay2.yaml"
        writer.write(
            env_dict=env_dict, output_path=path2, environment="test", scope="env"
        )
        model2 = loader.load(path2)
        assert model.content_hash() == model2.content_hash()

    def test_insertion_order_independent_hash(self, tmp_path: Path) -> None:
        """Determinism: reversed dict insertion order → identical content_hash."""
        from omnibase_infra.runtime.overlay.overlay_file_loader import OverlayFileLoader
        from omnibase_infra.runtime.overlay.overlay_writer import OverlayWriter

        env_a = {
            "KAFKA_BOOTSTRAP_SERVERS": "host:9092",
            "POSTGRES_HOST": "db",
            "POSTGRES_PORT": "5436",
        }
        env_b = {
            "POSTGRES_PORT": "5436",
            "POSTGRES_HOST": "db",
            "KAFKA_BOOTSTRAP_SERVERS": "host:9092",
        }

        writer = OverlayWriter()
        path_a = tmp_path / "a.yaml"
        path_b = tmp_path / "b.yaml"
        writer.write(
            env_dict=env_a, output_path=path_a, environment="test", scope="env"
        )
        writer.write(
            env_dict=env_b, output_path=path_b, environment="test", scope="env"
        )

        # Byte-identical output regardless of insertion order
        assert path_a.read_bytes() == path_b.read_bytes()

        loader = OverlayFileLoader(require_restricted_permissions=False)
        assert loader.load(path_a).content_hash() == loader.load(path_b).content_hash()

    def test_replay_determinism_same_inputs_same_manifest(
        self,
        sample_overlay_yaml: Path,
        contracts_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Replay proof: same overlay + same contracts → same manifest hash across runs."""
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
        assert result1.unused_overlay_keys == result2.unused_overlay_keys
