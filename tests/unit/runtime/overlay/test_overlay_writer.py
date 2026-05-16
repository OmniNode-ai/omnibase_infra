# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for OverlayWriter — atomic write, permissions, sorted YAML, secret warning."""

from __future__ import annotations

import logging
import stat
from pathlib import Path

import pytest

from omnibase_core.models.overlay.model_overlay_file import ModelOverlayFile
from omnibase_infra.runtime.overlay.overlay_file_loader import OverlayFileLoader
from omnibase_infra.runtime.overlay.overlay_writer import OverlayWriter


def _make_overlay(**kwargs: object) -> ModelOverlayFile:
    defaults: dict[str, object] = {
        "overlay_version": "1.0.0",
        "environment": "dev",
        "scope": "env",
    }
    defaults.update(kwargs)
    return ModelOverlayFile.model_validate(defaults)


@pytest.mark.unit
class TestOverlayWriter:
    def test_round_trips_through_loader(self, tmp_path: Path) -> None:
        overlay = _make_overlay(transports={"database": {"POSTGRES_HOST": "localhost"}})
        target = tmp_path / "overlay.yaml"
        OverlayWriter().write(overlay, target)
        loaded = OverlayFileLoader().load(target)
        assert loaded.transports["database"]["POSTGRES_HOST"] == "localhost"
        assert loaded.content_hash() == overlay.content_hash()

    def test_atomic_write_no_corruption(self, tmp_path: Path) -> None:
        overlay = _make_overlay()
        target = tmp_path / "overlay.yaml"
        writer = OverlayWriter()
        writer.write(overlay, target)
        writer.write(overlay, target)
        loaded = OverlayFileLoader().load(target)
        assert loaded.environment == "dev"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "dir" / "overlay.yaml"
        OverlayWriter().write(_make_overlay(), target)
        assert target.exists()

    def test_sets_chmod_600(self, tmp_path: Path) -> None:
        target = tmp_path / "overlay.yaml"
        OverlayWriter().write(_make_overlay(), target)
        mode = target.stat().st_mode
        assert not (mode & stat.S_IROTH)
        assert not (mode & stat.S_IWOTH)
        assert not (mode & stat.S_IRGRP)
        assert not (mode & stat.S_IWGRP)

    def test_yaml_output_sorted_keys(self, tmp_path: Path) -> None:
        overlay = _make_overlay(
            transports={"kafka": {"Z_KEY": "z"}, "database": {"A_KEY": "a"}}
        )
        target = tmp_path / "overlay.yaml"
        OverlayWriter().write(overlay, target)
        content = target.read_text()
        db_pos = content.index("database")
        kafka_pos = content.index("kafka")
        assert db_pos < kafka_pos

    def test_secret_warning_logged_for_secrets_section(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        overlay = _make_overlay(secrets={"INFISICAL_CLIENT_SECRET": "actual_secret"})
        target = tmp_path / "overlay.yaml"
        with caplog.at_level(logging.WARNING):
            OverlayWriter().write(overlay, target)
        assert any(
            "secret" in r.message.lower() or "SECRET" in r.message
            for r in caplog.records
        )

    def test_secret_warning_logged_for_transport_password_key(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        overlay = _make_overlay(
            transports={"database": {"POSTGRES_PASSWORD": "s3cr3t"}}
        )
        target = tmp_path / "overlay.yaml"
        with caplog.at_level(logging.WARNING):
            OverlayWriter().write(overlay, target)
        assert any(
            "secret" in r.message.lower() or "PASSWORD" in r.message
            for r in caplog.records
        )

    def test_no_warning_for_non_secret_keys(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        overlay = _make_overlay(
            transports={
                "database": {"POSTGRES_HOST": "localhost", "POSTGRES_PORT": "5436"}
            }
        )
        target = tmp_path / "overlay.yaml"
        with caplog.at_level(logging.WARNING):
            OverlayWriter().write(overlay, target)
        assert not any("secret" in r.message.lower() for r in caplog.records)
