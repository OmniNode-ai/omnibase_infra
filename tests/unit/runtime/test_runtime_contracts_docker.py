# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Test that runtime contracts are accessible in Docker-like conditions.

Simulates the Docker scenario where there is no source tree and the
contracts must be found via ONEX_RUNTIME_CONTRACTS_DIR or the package-
relative path added in OMN-6440.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from omnibase_core.contracts.runtime_contracts import get_runtime_contracts_dir


@pytest.mark.unit
class TestRuntimeContractsDockerAccessibility:
    """Verify runtime contracts resolve correctly when source tree is absent."""

    def test_contracts_found_via_env_var(self, tmp_path: Path) -> None:
        """When ONEX_RUNTIME_CONTRACTS_DIR is set, contracts resolve from it."""
        contracts_dir = tmp_path / "runtime"
        contracts_dir.mkdir()
        (contracts_dir / "runtime_orchestrator.yaml").touch()

        with patch.dict(os.environ, {"ONEX_RUNTIME_CONTRACTS_DIR": str(contracts_dir)}):
            result = get_runtime_contracts_dir()
            assert result == contracts_dir

    def test_contracts_found_without_env_var(self) -> None:
        """Contracts are found without env var (package-relative or repo-relative)."""
        # Clear env var to force non-env resolution
        with patch.dict(os.environ, {"ONEX_RUNTIME_CONTRACTS_DIR": ""}):
            result = get_runtime_contracts_dir()
            yamls = list(result.glob("*.yaml"))
            assert len(yamls) == 5, (
                f"Expected 5 runtime contract YAMLs, found {len(yamls)}: "
                f"{[y.name for y in yamls]}"
            )

    def test_docker_compose_env_includes_runtime_contracts_dir(self) -> None:
        """The x-runtime-env anchor must declare ONEX_RUNTIME_CONTRACTS_DIR."""
        import yaml

        compose_path = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "docker"
            / "docker-compose.infra.yml"
        )
        with open(compose_path) as f:
            data = yaml.safe_load(f)

        runtime_env = data.get("x-runtime-env", {})
        assert "ONEX_RUNTIME_CONTRACTS_DIR" in runtime_env, (
            "x-runtime-env must declare ONEX_RUNTIME_CONTRACTS_DIR "
            "so runtime containers can find omnibase_core contract YAMLs"
        )
