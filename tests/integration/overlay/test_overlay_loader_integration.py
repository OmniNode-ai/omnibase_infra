# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
import yaml

from omnibase_core.models.overlay.model_overlay_file import ModelOverlayFile


class TestOverlayFileLoaderIntegration:
    """Integration tests for OverlayFileLoader — validates YAML parsing,
    schema validation, and permission enforcement with real files."""

    def test_load_valid_overlay_returns_model(self, sample_overlay_yaml: Path) -> None:
        from omnibase_infra.runtime.overlay.overlay_file_loader import OverlayFileLoader

        loader = OverlayFileLoader()
        result = loader.load(sample_overlay_yaml)
        assert isinstance(result, ModelOverlayFile)
        assert result.environment == "test"

    def test_load_rejects_world_readable(self, sample_overlay_yaml: Path) -> None:
        from omnibase_infra.runtime.overlay.errors import OverlayPermissionError
        from omnibase_infra.runtime.overlay.overlay_file_loader import OverlayFileLoader

        sample_overlay_yaml.chmod(0o644)
        loader = OverlayFileLoader(require_restricted_permissions=True)
        with pytest.raises(OverlayPermissionError):
            loader.load(sample_overlay_yaml)

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        from omnibase_infra.runtime.overlay.errors import OverlayNotFoundError
        from omnibase_infra.runtime.overlay.overlay_file_loader import OverlayFileLoader

        loader = OverlayFileLoader()
        with pytest.raises(OverlayNotFoundError):
            loader.load(tmp_path / "nonexistent.yaml")

    def test_load_invalid_schema_raises(self, overlay_dir: Path) -> None:
        from omnibase_infra.runtime.overlay.errors import OverlaySchemaInvalidError
        from omnibase_infra.runtime.overlay.overlay_file_loader import OverlayFileLoader

        overlay_dir.mkdir(parents=True, exist_ok=True)
        bad = overlay_dir / "bad.yaml"
        bad.write_text(yaml.safe_dump({"garbage": True}))
        bad.chmod(0o600)
        loader = OverlayFileLoader()
        with pytest.raises(OverlaySchemaInvalidError):
            loader.load(bad)
