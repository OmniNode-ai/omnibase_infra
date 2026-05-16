# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""E2E proof of life: generate overlay → write → load → resolve → inject → verify.

Task 10 per the OMN-11069 overlay config implementation plan.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from omnibase_core.models.overlay.model_overlay_file import ModelOverlayFile
from omnibase_infra.runtime.overlay.overlay_config_resolver import OverlayConfigResolver
from omnibase_infra.runtime.overlay.overlay_file_loader import OverlayFileLoader
from omnibase_infra.runtime.overlay.overlay_writer import OverlayWriter


@pytest.mark.integration
class TestOverlayEndToEnd:
    def test_full_pipeline_no_env_file(self, tmp_path: Path) -> None:
        """Generate overlay → write → load → resolve → inject → verify."""
        # 1. Build overlay (simulating onboarding output)
        overlay = ModelOverlayFile.model_validate(
            {
                "overlay_version": "1.0.0",
                "environment": "test",
                "scope": "env",
                "transports": {
                    "db": {
                        "POSTGRES_HOST": "test-host",
                        "POSTGRES_PORT": "5432",
                    },
                    "kafka": {
                        "KAFKA_BOOTSTRAP_SERVERS": "kafka:9092",
                    },
                },
            }
        )

        # 2. Write to disk via OverlayWriter
        overlay_path = tmp_path / "overlay.yaml"
        OverlayWriter().write(overlay, overlay_path)
        assert overlay_path.exists(), "OverlayWriter must create the file"

        # 3. Load from disk via OverlayFileLoader
        loaded = OverlayFileLoader().load(overlay_path)
        # content_hash() may differ if overlay_version serializes differently
        # (ModelSemVer dict vs string). Assert logical equivalence instead.
        assert loaded.transports == overlay.transports, (
            "Round-trip must preserve transport values"
        )
        assert loaded.environment == overlay.environment

        # 4. Build synthetic requirements the overlay SATISFIES
        # The stub resolver accepts any requirements object; we pass None to use
        # the no-requirements path (all overlay keys are resolved).
        # When the real resolver lands, pass ModelConfigRequirements here.
        requirements = None

        # 5. Resolve via OverlayConfigResolver
        result = OverlayConfigResolver().resolve(loaded, requirements)

        # 6. Assert resolved values match
        assert result.resolved.get("POSTGRES_HOST") == "test-host"
        assert result.resolved.get("POSTGRES_PORT") == "5432"
        assert result.resolved.get("KAFKA_BOOTSTRAP_SERVERS") == "kafka:9092"

        # 7. Assert manifest hashes populated
        assert result.manifest.overlay_file_hash.startswith("sha256:")
        assert result.manifest.config_source == "overlay"
        assert len(result.missing) == 0

        # 8. Apply to environment, verify injected
        env_keys = list(result.resolved.keys())
        env_backup: dict[str, str | None] = {k: os.environ.get(k) for k in env_keys}
        try:
            inj = result.apply_to_environment()
            injected_or_present = set(inj.injected_keys) | set(
                inj.skipped_existing_keys
            )
            assert "POSTGRES_HOST" in injected_or_present
            assert "KAFKA_BOOTSTRAP_SERVERS" in injected_or_present
            # Keys injected by us must be in os.environ
            for key in inj.injected_keys:
                assert key in os.environ, f"Injected key {key} not in os.environ"
        finally:
            # 9. Clean up env vars
            for k, v in env_backup.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_deterministic_resolution(self, tmp_path: Path) -> None:
        """Same overlay + same requirements → identical stable_identity_hash."""
        overlay = ModelOverlayFile.model_validate(
            {
                "overlay_version": "1.0.0",
                "environment": "test",
                "scope": "env",
                "transports": {
                    "db": {"POSTGRES_HOST": "deterministic-host"},
                },
            }
        )
        overlay_path = tmp_path / "overlay.yaml"
        OverlayWriter().write(overlay, overlay_path)
        loaded = OverlayFileLoader().load(overlay_path)

        r1 = OverlayConfigResolver().resolve(loaded, None)
        r2 = OverlayConfigResolver().resolve(loaded, None)

        assert (
            r1.manifest.stable_identity_hash() == r2.manifest.stable_identity_hash()
        ), "stable_identity_hash must be deterministic across invocations"
        assert r1.manifest.resolved_config_hash == r2.manifest.resolved_config_hash, (
            "resolved_config_hash must be deterministic"
        )
