# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the compose_gen phase added in OMN-8430.

compose_gen must regenerate docker-compose.infra.yml from the catalog CLI on
every deploy so that catalog changes (e.g. new LLM_* env vars) are reflected
in the running containers immediately after the next redeploy — not silently
skipped because the static compose file is never updated.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest
from deploy_agent.events import Phase, PhaseStatus, Scope
from deploy_agent.executor import SCOPE_BUNDLES, DeployExecutor


def _noop_phase_update(phase: Phase, status: PhaseStatus) -> None:
    pass


def _make_result(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


class TestComposeGen:
    """compose_gen must invoke catalog CLI and write to COMPOSE_FILE."""

    def test_compose_gen_calls_catalog_cli(self) -> None:
        """compose_gen must invoke uv run python -m omnibase_infra.docker.catalog.cli generate."""
        executor = DeployExecutor()
        captured_cmds: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs
        ) -> subprocess.CompletedProcess:
            captured_cmds.append(cmd)
            return _make_result()

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            executor.compose_gen(["core", "runtime"], _noop_phase_update)

        assert captured_cmds, "Expected compose_gen to call _run at least once"
        cmd = captured_cmds[0]

        assert "python" in cmd, f"Expected 'python' in cmd, got: {cmd}"
        assert "-m" in cmd, f"Expected '-m' in cmd, got: {cmd}"
        assert "omnibase_infra.docker.catalog.cli" in cmd, (
            f"Expected catalog CLI module in cmd, got: {cmd}"
        )
        assert "generate" in cmd, f"Expected 'generate' subcommand in cmd, got: {cmd}"

    def test_compose_gen_passes_bundles(self) -> None:
        """compose_gen must pass the provided bundle names to the catalog CLI."""
        executor = DeployExecutor()
        captured_cmds: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs
        ) -> subprocess.CompletedProcess:
            captured_cmds.append(cmd)
            return _make_result()

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            executor.compose_gen(["core", "runtime"], _noop_phase_update)

        cmd = captured_cmds[0]
        assert "core" in cmd, f"Expected 'core' bundle in cmd, got: {cmd}"
        assert "runtime" in cmd, f"Expected 'runtime' bundle in cmd, got: {cmd}"

    def test_compose_gen_writes_to_compose_file(self) -> None:
        """compose_gen must pass --output pointing to COMPOSE_FILE."""
        from deploy_agent.executor import COMPOSE_FILE

        executor = DeployExecutor()
        captured_cmds: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs
        ) -> subprocess.CompletedProcess:
            captured_cmds.append(cmd)
            return _make_result()

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            executor.compose_gen(["core", "runtime"], _noop_phase_update)

        cmd = captured_cmds[0]
        assert "--output" in cmd, f"Expected --output flag in cmd, got: {cmd}"
        output_idx = cmd.index("--output") + 1
        assert cmd[output_idx] == COMPOSE_FILE, (
            f"Expected --output {COMPOSE_FILE!r}, got {cmd[output_idx]!r}"
        )

    def test_compose_gen_succeeds_on_catalog_cli_failure(self) -> None:
        """compose_gen must not raise even when the catalog CLI exits non-zero (non-fatal)."""
        executor = DeployExecutor()

        def fake_run(
            cmd: list[str], timeout: int, **kwargs
        ) -> subprocess.CompletedProcess:
            return _make_result(returncode=1, stderr="catalog CLI failed")

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            # Must not raise
            executor.compose_gen(["core", "runtime"], _noop_phase_update)

    def test_compose_gen_emits_in_progress_then_success(self) -> None:
        """compose_gen must call on_phase_update with IN_PROGRESS then SUCCESS."""
        executor = DeployExecutor()
        phase_updates: list[tuple[Phase, PhaseStatus]] = []

        def track_updates(phase: Phase, status: PhaseStatus) -> None:
            phase_updates.append((phase, status))

        def fake_run(
            cmd: list[str], timeout: int, **kwargs
        ) -> subprocess.CompletedProcess:
            return _make_result()

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            executor.compose_gen(["core", "runtime"], track_updates)

        assert (
            Phase.COMPOSE_GEN,
            PhaseStatus.IN_PROGRESS,
        ) in phase_updates, "Expected IN_PROGRESS update for COMPOSE_GEN"
        assert (
            Phase.COMPOSE_GEN,
            PhaseStatus.SUCCESS,
        ) in phase_updates, "Expected SUCCESS update for COMPOSE_GEN"

    def test_scope_bundles_covers_all_scopes(self) -> None:
        """SCOPE_BUNDLES must have an entry for every Scope value."""
        for scope in Scope:
            assert scope in SCOPE_BUNDLES, (
                f"SCOPE_BUNDLES is missing entry for Scope.{scope}"
            )

    @pytest.mark.parametrize(
        ("scope", "expected_bundles"),
        [
            (Scope.CORE, ["core"]),
            (Scope.RUNTIME, ["core", "runtime"]),
            (Scope.FULL, ["core", "runtime"]),
        ],
    )
    def test_scope_bundles_values(
        self, scope: Scope, expected_bundles: list[str]
    ) -> None:
        """SCOPE_BUNDLES must map each scope to the correct catalog bundle list."""
        assert SCOPE_BUNDLES[scope] == expected_bundles, (
            f"Scope.{scope} should map to {expected_bundles}, got {SCOPE_BUNDLES[scope]}"
        )
