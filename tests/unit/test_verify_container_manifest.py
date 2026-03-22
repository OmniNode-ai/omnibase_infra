# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for verify_container_manifest.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


def _make_docker_ps_line(name: str, state: str, status: str) -> str:
    return json.dumps({"Names": name, "State": state, "Status": status})


@pytest.mark.unit
class TestContainerManifestVerification:
    """Test container manifest verification logic."""

    def test_all_containers_running(self) -> None:
        """All expected containers running -> exit 0."""
        from omnibase_infra.scripts.verify_container_manifest import verify_containers

        docker_output = "\n".join(
            [
                _make_docker_ps_line(
                    "omnibase-infra-postgres", "running", "Up 2 hours (healthy)"
                ),
                _make_docker_ps_line(
                    "omnibase-infra-redpanda", "running", "Up 2 hours (healthy)"
                ),
            ]
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=docker_output, returncode=0)
            result = verify_containers(
                expected=[
                    "omnibase-infra-postgres",
                    "omnibase-infra-redpanda",
                ],
                restart_once=False,
            )
        assert result.exit_code == 0
        assert len(result.failures) == 0

    def test_container_stuck_in_created(self) -> None:
        """Container in Created state -> exit 1."""
        from omnibase_infra.scripts.verify_container_manifest import verify_containers

        docker_output = "\n".join(
            [
                _make_docker_ps_line(
                    "omnibase-infra-postgres", "running", "Up 2 hours"
                ),
                _make_docker_ps_line(
                    "omnibase-infra-runtime-worker-1", "created", "Created"
                ),
            ]
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=docker_output, returncode=0)
            result = verify_containers(
                expected=[
                    "omnibase-infra-postgres",
                    "omnibase-infra-runtime-worker-1",
                ],
                restart_once=False,
            )
        assert result.exit_code == 1
        assert any("created" in f.lower() for f in result.failures)

    def test_container_missing(self) -> None:
        """Expected container not in docker ps -a -> exit 1."""
        from omnibase_infra.scripts.verify_container_manifest import verify_containers

        docker_output = _make_docker_ps_line(
            "omnibase-infra-postgres", "running", "Up 2 hours"
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=docker_output, returncode=0)
            result = verify_containers(
                expected=["omnibase-infra-postgres", "omninode-runtime"],
                restart_once=False,
            )
        assert result.exit_code == 1
        assert any("not found" in f.lower() for f in result.failures)

    def test_restart_once_recovers(self) -> None:
        """Container exited, restart succeeds -> exit 0."""
        from omnibase_infra.scripts.verify_container_manifest import verify_containers

        # First call: docker ps -a shows exited
        docker_before = _make_docker_ps_line("omninode-runtime", "exited", "Exited (1)")
        # After restart: docker ps -a shows running
        docker_after = _make_docker_ps_line(
            "omninode-runtime", "running", "Up 5 seconds"
        )
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(stdout=docker_before, returncode=0),  # initial ps
                MagicMock(returncode=0),  # docker restart
                MagicMock(stdout=docker_after, returncode=0),  # recheck ps
            ]
            result = verify_containers(
                expected=["omninode-runtime"],
                restart_once=True,
            )
        assert result.exit_code == 0
        assert len(result.recovered) == 1
