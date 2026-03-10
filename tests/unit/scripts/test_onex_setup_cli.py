# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for onex-setup.py CLI.

Tests cover:
    - Preset flag skips interactive prompt
    - --dry-run does not write topology.yaml
    - --no-interactive writes standard topology to OMNIBASE_DIR
    - Cloud selection stores mode=CLOUD in topology (not converted to DISABLED)
    - resolve_compose_file resolution order (CLI arg, env var, upward search)

Ticket: OMN-3496
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Load the CLI module from the scripts/ directory without installing it.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CLI_PATH = _REPO_ROOT / "scripts" / "onex-setup.py"

_spec = importlib.util.spec_from_file_location("onex_setup", _CLI_PATH)
assert _spec is not None and _spec.loader is not None, f"Cannot load {_CLI_PATH}"
_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cli)  # type: ignore[union-attr]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_main(argv: list[str], env: dict[str, str] | None = None) -> int:
    """Run main() with the given argv, optionally overriding environment.

    Mocks _run_orchestrator to skip actual Docker / Infisical calls.

    Returns:
        Exit code returned by main().
    """
    saved_argv = sys.argv[:]
    saved_env: dict[str, str | None] = {}

    try:
        sys.argv = ["onex-setup"] + argv

        if env is not None:
            for key, value in env.items():
                saved_env[key] = os.environ.get(key)
                os.environ[key] = value

        # Patch _run_orchestrator to always return True without doing I/O.
        async def _fake_orchestrator(
            topology: Any, compose_file_path: str, dry_run: bool
        ) -> bool:
            return True

        with patch.object(_cli, "_run_orchestrator", side_effect=_fake_orchestrator):
            return _cli.main()

    finally:
        sys.argv = saved_argv
        for key, old_value in saved_env.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOnexSetupCLI:
    """Tests for the onex-setup.py interactive CLI."""

    def test_preset_flag_skips_interactive_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--preset minimal must not invoke input() and must succeed."""
        monkeypatch.setenv("OMNIBASE_DIR", str(tmp_path))
        monkeypatch.setenv("ONEX_COMPOSE_FILE", "docker/docker-compose.infra.yml")

        with patch("builtins.input") as mock_input:
            exit_code = _run_main(["--preset", "minimal", "--no-interactive"])

        mock_input.assert_not_called()
        assert exit_code == 0

    def test_dry_run_does_not_write_topology_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--dry-run must not write topology.yaml to OMNIBASE_DIR."""
        monkeypatch.setenv("OMNIBASE_DIR", str(tmp_path))

        exit_code = _run_main(["--preset", "minimal", "--dry-run"])

        topology_file = tmp_path / "topology.yaml"
        assert not topology_file.exists(), "--dry-run must not write topology.yaml"
        assert exit_code == 0

    def test_no_interactive_writes_standard_topology_to_omnibase_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--no-interactive writes standard 4-service topology to OMNIBASE_DIR."""
        monkeypatch.setenv("OMNIBASE_DIR", str(tmp_path))
        monkeypatch.setenv("ONEX_COMPOSE_FILE", "docker/docker-compose.infra.yml")

        exit_code = _run_main(["--no-interactive"])

        topology_file = tmp_path / "topology.yaml"
        assert topology_file.exists(), "topology.yaml must be written"
        content = topology_file.read_text()
        # Standard preset includes infisical (4th service)
        assert "infisical" in content
        assert exit_code == 0

    def test_cloud_selection_stores_cloud_mode_in_topology_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cloud mode must be stored as 'CLOUD' in topology.yaml (I8 — not converted).

        The CLI must NOT silently convert CLOUD to DISABLED. The topology file
        must preserve the user's intent even though provisioning is not yet
        implemented for cloud services.
        """
        from omnibase_core.enums.enum_deployment_mode import EnumDeploymentMode
        from omnibase_core.models.core.model_deployment_topology import (
            ModelDeploymentTopology,
        )
        from omnibase_core.models.core.model_deployment_topology_service import (
            ModelDeploymentTopologyService,
        )

        # Build a topology with one cloud service.
        topology = ModelDeploymentTopology(
            schema_version="1.0",
            services={
                "postgres": ModelDeploymentTopologyService(
                    mode=EnumDeploymentMode.CLOUD,
                    local=None,
                ),
            },
            presets={},
            active_preset=None,
        )

        topo_path = tmp_path / "topology.yaml"
        topology.to_yaml(topo_path)

        # Verify the YAML preserves CLOUD mode (not DISABLED or LOCAL).
        content = topo_path.read_text()
        assert "CLOUD" in content, "topology.yaml must store mode=CLOUD, not convert it"
        assert "DISABLED" not in content

        # Verify round-trip: loading back preserves CLOUD mode.
        loaded = ModelDeploymentTopology.from_yaml(topo_path)
        assert loaded.services["postgres"].mode == EnumDeploymentMode.CLOUD

    def test_compose_file_resolved_from_env_var_when_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """resolve_compose_file must return ONEX_COMPOSE_FILE env var when set."""
        expected = "/custom/path/docker-compose.infra.yml"
        monkeypatch.setenv("ONEX_COMPOSE_FILE", expected)

        result = _cli.resolve_compose_file(None)

        assert result == expected

    def test_compose_file_cli_arg_takes_precedence_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """resolve_compose_file must prefer CLI arg over ONEX_COMPOSE_FILE env."""
        monkeypatch.setenv("ONEX_COMPOSE_FILE", "/env/path/compose.yml")
        cli_arg = "/cli/path/compose.yml"

        result = _cli.resolve_compose_file(cli_arg)

        assert result == cli_arg

    def test_compose_file_raises_when_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """resolve_compose_file must raise RuntimeError when no compose file found."""
        monkeypatch.delenv("ONEX_COMPOSE_FILE", raising=False)
        # Use an empty tmp directory that has no docker-compose.infra.yml
        monkeypatch.chdir(tmp_path)

        with pytest.raises(
            RuntimeError, match=r"Cannot locate docker-compose\.infra\.yml"
        ):
            _cli.resolve_compose_file(None)

    def test_omnibase_dir_respects_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_omnibase_dir() must return OMNIBASE_DIR env var when set."""
        monkeypatch.setenv("OMNIBASE_DIR", str(tmp_path))

        result = _cli._omnibase_dir()

        assert result == tmp_path

    def test_preset_minimal_has_three_services(self) -> None:
        """minimal preset must produce exactly 3 services."""
        from omnibase_core.models.core.model_deployment_topology import (
            ModelDeploymentTopology,
        )

        topo = ModelDeploymentTopology.default_minimal()
        assert topo.active_preset == "minimal"
        assert len(topo.services) == 3
        assert "postgres" in topo.services
        assert "redpanda" in topo.services
        assert "valkey" in topo.services

    def test_preset_standard_includes_infisical(self) -> None:
        """standard preset must include infisical as the 4th service."""
        from omnibase_core.models.core.model_deployment_topology import (
            ModelDeploymentTopology,
        )

        topo = ModelDeploymentTopology.default_standard()
        assert topo.active_preset == "standard"
        assert "infisical" in topo.services
        assert len(topo.services) == 4

    def test_topology_file_flag_loads_existing_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--topology-file must load from the provided path."""
        from omnibase_core.models.core.model_deployment_topology import (
            ModelDeploymentTopology,
        )

        topo = ModelDeploymentTopology.default_minimal()
        topo_path = tmp_path / "my-topology.yaml"
        topo.to_yaml(topo_path)

        monkeypatch.setenv("OMNIBASE_DIR", str(tmp_path))
        monkeypatch.setenv("ONEX_COMPOSE_FILE", "docker/docker-compose.infra.yml")

        exit_code = _run_main(["--topology-file", str(topo_path), "--no-interactive"])

        assert exit_code == 0
