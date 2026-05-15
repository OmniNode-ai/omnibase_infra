# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for OverlayFileLoader.

These tests verify the full parse-and-validate path end-to-end using real
YAML fixtures on disk, without mocking the model or YAML parser.
"""

from __future__ import annotations

import pytest

pytest.importorskip("omnibase_core.models.overlay.model_overlay_file")

from pathlib import Path

from omnibase_infra.runtime.overlay.errors import (
    OverlayNotFoundError,
    OverlayPermissionError,
    OverlaySchemaInvalidError,
)
from omnibase_infra.runtime.overlay.overlay_file_loader import (
    OverlayFileLoader,
)

_FULL_OVERLAY = """\
overlay_version: '1.0.0'
environment: dev
scope: env
transports:
  database:
    POSTGRES_HOST: db.test.local
    POSTGRES_PORT: '5436'
  kafka:
    KAFKA_BOOTSTRAP_SERVERS: kafka.test.local:9092
secrets:
  INFISICAL_ADDR: 'http://infisical.test.local:8880'
services:
  omniclaude:
    ENABLE_POSTGRES: 'true'
llm:
  coder:
    url: 'http://llm.test.local:8000'
"""


@pytest.mark.integration
class TestOverlayFileLoaderIntegration:
    def test_full_overlay_round_trip(self, tmp_path: Path) -> None:
        p = tmp_path / "overlay.yaml"
        p.write_text(_FULL_OVERLAY)
        result = OverlayFileLoader().load(p)

        assert str(result.overlay_version) == "1.0.0"
        assert result.environment == "dev"
        assert result.transports["database"]["POSTGRES_HOST"] == "db.test.local"
        assert (
            result.transports["kafka"]["KAFKA_BOOTSTRAP_SERVERS"]
            == "kafka.test.local:9092"
        )
        assert result.secrets["INFISICAL_ADDR"] == "http://infisical.test.local:8880"
        assert result.services["omniclaude"]["ENABLE_POSTGRES"] == "true"
        assert result.llm["coder"]["url"] == "http://llm.test.local:8000"

    def test_content_hash_stable_on_reload(self, tmp_path: Path) -> None:
        p = tmp_path / "overlay.yaml"
        p.write_text(_FULL_OVERLAY)
        loader = OverlayFileLoader()
        r1 = loader.load(p)
        r2 = loader.load(p)
        assert r1.content_hash() == r2.content_hash()

    def test_missing_overlay_gives_onboarding_instructions(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(OverlayNotFoundError, match="onboarding"):
            OverlayFileLoader().load(tmp_path / "does_not_exist.yaml")

    def test_invalid_yaml_content_raises_schema_error(self, tmp_path: Path) -> None:
        p = tmp_path / "overlay.yaml"
        p.write_text("not_a_mapping: [1, 2\n  broken yaml {{")
        with pytest.raises(OverlaySchemaInvalidError):
            OverlayFileLoader().load(p)

    def test_permission_enforcement_on_600(self, tmp_path: Path) -> None:
        p = tmp_path / "overlay.yaml"
        p.write_text("overlay_version: '1.0.0'\nenvironment: dev\nscope: env\n")
        p.chmod(0o600)
        result = OverlayFileLoader(require_restricted_permissions=True).load(p)
        assert result is not None

    def test_permission_enforcement_rejects_644(self, tmp_path: Path) -> None:
        p = tmp_path / "overlay.yaml"
        p.write_text("overlay_version: '1.0.0'\nenvironment: dev\nscope: env\n")
        p.chmod(0o644)
        with pytest.raises(OverlayPermissionError):
            OverlayFileLoader(require_restricted_permissions=True).load(p)
