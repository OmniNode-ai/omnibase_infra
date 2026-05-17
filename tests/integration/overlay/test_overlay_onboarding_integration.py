# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


class TestOverlayOnboardingIntegration:
    def test_overlay_from_env_dict_produces_loadable_overlay(
        self, tmp_path: Path
    ) -> None:
        from omnibase_infra.runtime.overlay.overlay_file_loader import OverlayFileLoader
        from omnibase_infra.runtime.overlay.overlay_from_env import (
            overlay_from_env_dict,
        )

        env = {"KAFKA_BOOTSTRAP_SERVERS": "host:9092", "POSTGRES_HOST": "db"}
        path = overlay_from_env_dict(env, output_path=tmp_path / "overlay.yaml")

        loader = OverlayFileLoader(require_restricted_permissions=False)
        model = loader.load(path)
        assert model.environment == "local"
        pairs = model.all_env_pairs()
        assert pairs["KAFKA_BOOTSTRAP_SERVERS"] == "host:9092"

    def test_overlay_from_env_dict_never_targets_default_in_tests(
        self, tmp_path: Path
    ) -> None:
        """Enforce: tests always pass explicit output_path."""
        from omnibase_infra.runtime.overlay.overlay_from_env import (
            overlay_from_env_dict,
        )

        # This test proves the function works with explicit paths
        path = overlay_from_env_dict(
            {"KAFKA_BOOTSTRAP_SERVERS": "x:9092"},
            output_path=tmp_path / "explicit.yaml",
        )
        assert path == tmp_path / "explicit.yaml"
        assert path.exists()
