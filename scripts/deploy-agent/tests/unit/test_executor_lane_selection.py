# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-lane compose/project/health-target selection for the deploy executor (OMN-12572).

The executor previously hardcoded the dev lane (``docker-compose.infra.yml`` /
``omnibase-infra`` / ports 8085+8086). It must now parameterize the compose
file(s), compose project, and runtime health targets by ``runtime_lane`` so the
agent can deploy ``stability-test`` (18085/18086, project
``omnibase-infra-stability-test``) and ``prod`` (28085/28086, project
``omnibase-infra-prod``). The dev lane is unchanged.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest
from deploy_agent.events import EnumRuntimeLane, Phase, PhaseStatus, Scope
from deploy_agent.executor import (
    COMPOSE_FILE,
    COMPOSE_PROJECT,
    RUNTIME_HEALTH_TARGETS,
    DeployExecutor,
    lane_config_for,
)

pytestmark = pytest.mark.unit


def _noop_phase_update(phase: Phase, status: PhaseStatus) -> None:
    return None


def _ok() -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


def _is_projection_table_check(cmd: list[str]) -> bool:
    return "omnidash_analytics" in cmd and any(
        f"SELECT to_regclass('public.{table}') IS NOT NULL" in cmd
        for table in ("delegation_events", "node_service_registry")
    )


class TestLaneConfig:
    def test_dev_lane_matches_legacy_module_constants(self) -> None:
        cfg = lane_config_for(EnumRuntimeLane.DEV)
        assert cfg.compose_project == COMPOSE_PROJECT == "omnibase-infra"
        assert cfg.postgres_container == "omnibase-infra-postgres"
        assert cfg.compose_files == (COMPOSE_FILE,)
        assert cfg.runtime_health_targets == RUNTIME_HEALTH_TARGETS
        assert cfg.runtime_health_targets == (
            ("omninode-runtime", 8085),
            ("runtime-effects", 8086),
        )

    def test_stability_test_lane_config(self) -> None:
        cfg = lane_config_for(EnumRuntimeLane.STABILITY_TEST)
        assert cfg.compose_project == "omnibase-infra-stability-test"
        assert cfg.postgres_container == "omnibase-infra-stability-test-postgres"
        # overlay must be layered on top of the base infra compose file
        assert cfg.compose_files[0] == COMPOSE_FILE
        assert any("docker-compose.stability-test.yml" in f for f in cfg.compose_files)
        assert cfg.runtime_health_targets == (
            ("omninode-runtime", 18085),
            ("runtime-effects", 18086),
        )

    def test_prod_lane_config(self) -> None:
        cfg = lane_config_for(EnumRuntimeLane.PROD)
        assert cfg.compose_project == "omnibase-infra-prod"
        assert cfg.postgres_container == "omnibase-infra-prod-postgres"
        assert cfg.compose_files[0] == COMPOSE_FILE
        assert any("docker-compose.prod.yml" in f for f in cfg.compose_files)
        assert cfg.runtime_health_targets == (
            ("omninode-runtime", 28085),
            ("runtime-effects", 28086),
        )


class TestComposeUpLaneSelection:
    def test_dev_lane_compose_up_uses_base_file_and_project(self) -> None:
        executor = DeployExecutor()
        captured: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            captured.append(cmd)
            return _ok()

        with (
            patch("deploy_agent.executor._run", side_effect=fake_run),
            patch.object(executor, "_ensure_runtime_migrations_ready"),
            patch(
                "deploy_agent.executor.verify_containers_up", return_value=(True, [])
            ),
        ):
            executor._compose_up(
                Phase.RUNTIME,
                Scope.RUNTIME,
                [],
                _noop_phase_update,
                lane=EnumRuntimeLane.DEV,
            )

        cmd = captured[0]
        assert cmd[:3] == ["docker", "compose", "-f"]
        assert cmd[3] == COMPOSE_FILE
        assert "-p" in cmd
        assert cmd[cmd.index("-p") + 1] == "omnibase-infra"
        # dev does not layer an overlay
        assert cmd.count("-f") == 1

    def test_stability_lane_compose_up_layers_overlay_and_project(self) -> None:
        executor = DeployExecutor()
        captured: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            captured.append(cmd)
            return _ok()

        with (
            patch("deploy_agent.executor._run", side_effect=fake_run),
            patch.object(executor, "_ensure_runtime_migrations_ready"),
            patch(
                "deploy_agent.executor.verify_containers_up", return_value=(True, [])
            ),
        ):
            executor._compose_up(
                Phase.RUNTIME,
                Scope.RUNTIME,
                [],
                _noop_phase_update,
                lane=EnumRuntimeLane.STABILITY_TEST,
            )

        cmd = captured[0]
        assert cmd.count("-f") == 2, f"stability lane must layer the overlay: {cmd}"
        assert any("docker-compose.stability-test.yml" in tok for tok in cmd)
        assert "omnibase-infra-stability-test" in cmd

    def test_prod_lane_compose_up_layers_overlay_and_project(self) -> None:
        executor = DeployExecutor()
        captured: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            captured.append(cmd)
            return _ok()

        with (
            patch("deploy_agent.executor._run", side_effect=fake_run),
            patch.object(executor, "_ensure_runtime_migrations_ready"),
            patch(
                "deploy_agent.executor.verify_containers_up", return_value=(True, [])
            ),
        ):
            executor._compose_up(
                Phase.RUNTIME,
                Scope.RUNTIME,
                [],
                _noop_phase_update,
                lane=EnumRuntimeLane.PROD,
            )

        cmd = captured[0]
        assert cmd.count("-f") == 2, f"prod lane must layer the overlay: {cmd}"
        assert any("docker-compose.prod.yml" in tok for tok in cmd)
        assert "omnibase-infra-prod" in cmd


class TestVerifyLaneHealthTargets:
    @staticmethod
    def _health_payload() -> str:
        import json

        return json.dumps(
            {
                "status": "healthy",
                "details": {"is_running": True, "config_prefetch_status": "ok"},
            }
        )

    def test_stability_lane_verify_probes_18085_18086(self) -> None:
        executor = DeployExecutor()

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            if cmd[:2] == ["docker", "ps"]:
                return _ok()
            if _is_projection_table_check(cmd):
                assert "omnibase-infra-stability-test-postgres" in cmd
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="t\n", stderr=""
                )
            if (
                "http://localhost:18085/health" in cmd
                or "http://localhost:18086/health" in cmd
            ):
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout=self._health_payload(), stderr=""
                )
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr="unexpected"
            )

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            checks = executor.verify(
                on_phase_update=_noop_phase_update, lane=EnumRuntimeLane.STABILITY_TEST
            )

        endpoints = [c.endpoint for c in checks]
        assert "http://localhost:18085/health" in endpoints
        assert "http://localhost:18086/health" in endpoints
        assert not any(":8085/health" in e for e in endpoints)

    def test_prod_lane_verify_probes_28085_28086(self) -> None:
        executor = DeployExecutor()

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            if cmd[:2] == ["docker", "ps"]:
                return _ok()
            if _is_projection_table_check(cmd):
                assert "omnibase-infra-prod-postgres" in cmd
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="t\n", stderr=""
                )
            if (
                "http://localhost:28085/health" in cmd
                or "http://localhost:28086/health" in cmd
            ):
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout=self._health_payload(), stderr=""
                )
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr="unexpected"
            )

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            checks = executor.verify(
                on_phase_update=_noop_phase_update, lane=EnumRuntimeLane.PROD
            )

        endpoints = [c.endpoint for c in checks]
        assert "http://localhost:28085/health" in endpoints
        assert "http://localhost:28086/health" in endpoints
        assert not any(":8085/health" in e for e in endpoints)

    def test_default_lane_verify_probes_dev_ports(self) -> None:
        """No explicit lane -> dev lane (legacy behavior preserved)."""
        executor = DeployExecutor()

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            if cmd[:2] == ["docker", "ps"]:
                return _ok()
            if _is_projection_table_check(cmd):
                assert "omnibase-infra-postgres" in cmd
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="t\n", stderr=""
                )
            if (
                "http://localhost:8085/health" in cmd
                or "http://localhost:8086/health" in cmd
            ):
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout=self._health_payload(), stderr=""
                )
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr="unexpected"
            )

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            checks = executor.verify(on_phase_update=_noop_phase_update)

        endpoints = [c.endpoint for c in checks]
        assert "http://localhost:8085/health" in endpoints
        assert "http://localhost:8086/health" in endpoints
