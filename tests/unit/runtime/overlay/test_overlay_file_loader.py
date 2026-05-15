# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for OverlayFileLoader."""

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

_VALID_OVERLAY = (
    "overlay_version: '1.0.0'\n"
    "environment: dev\n"
    "scope: env\n"
    "transports:\n"
    "  database:\n"
    "    POSTGRES_HOST: localhost\n"
)


@pytest.mark.unit
class TestOverlayFileLoader:
    def test_loads_valid_overlay(self, tmp_path: Path) -> None:
        p = tmp_path / "overlay.yaml"
        p.write_text(_VALID_OVERLAY)
        result = OverlayFileLoader().load(p)
        assert result.environment == "dev"
        assert result.transports["database"]["POSTGRES_HOST"] == "localhost"

    def test_not_found_raises_with_onboarding_message(self, tmp_path: Path) -> None:
        with pytest.raises(OverlayNotFoundError, match="onboarding"):
            OverlayFileLoader().load(tmp_path / "nonexistent.yaml")

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "overlay.yaml"
        p.write_text("{{invalid yaml")
        with pytest.raises(OverlaySchemaInvalidError):
            OverlayFileLoader().load(p)

    def test_missing_required_fields_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "overlay.yaml"
        p.write_text("overlay_version: '1.0.0'\n")
        with pytest.raises(OverlaySchemaInvalidError):
            OverlayFileLoader().load(p)

    def test_unsupported_version_rejected_by_model(self, tmp_path: Path) -> None:
        p = tmp_path / "overlay.yaml"
        p.write_text("overlay_version: '99.0.0'\nenvironment: dev\nscope: env\n")
        with pytest.raises(OverlaySchemaInvalidError):
            OverlayFileLoader().load(p)

    def test_warn_open_permissions_logs_only(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        p = tmp_path / "overlay.yaml"
        p.write_text(_VALID_OVERLAY)
        p.chmod(0o644)
        import logging

        with caplog.at_level(logging.WARNING):
            result = OverlayFileLoader(require_restricted_permissions=False).load(p)
        assert result is not None
        assert any(
            "permission" in r.message.lower() or "chmod" in r.message
            for r in caplog.records
        )

    def test_require_restricted_permissions_raises_on_open(
        self, tmp_path: Path
    ) -> None:
        p = tmp_path / "overlay.yaml"
        p.write_text(_VALID_OVERLAY)
        p.chmod(0o644)
        with pytest.raises(OverlayPermissionError, match="chmod 600"):
            OverlayFileLoader(require_restricted_permissions=True).load(p)

    def test_restricted_permissions_passes_on_600(self, tmp_path: Path) -> None:
        p = tmp_path / "overlay.yaml"
        p.write_text(_VALID_OVERLAY)
        p.chmod(0o600)
        result = OverlayFileLoader(require_restricted_permissions=True).load(p)
        assert result is not None
