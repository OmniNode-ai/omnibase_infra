# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the compose_gen phase added in OMN-8430, corrected in OMN-17291.

compose_gen renders the catalog on every deploy and validates the result, so a
catalog change that cannot render stops the deploy rather than reaching a lane.

OMN-17291 moved that render's ``--output`` off the TRACKED
``docker/docker-compose.infra.yml`` and onto the gitignored build artifact
``docker/docker-compose.generated.yml`` — the catalog CLI's own declared default
output. The tracked file was being overwritten in place on every deploy, which
left the lab deploy-source clone permanently dirty and, because the render is a
DIFFERENT stack (12 extra services, 31 required ``${VAR:?}`` names against the
tracked file's 50), broke compose validation for every subsequent
``scripts/deploy-runtime.sh`` run on the dev and stability lanes.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from deploy_agent.events import EnumRuntimeLane, Phase, PhaseStatus, Scope
from deploy_agent.executor import SCOPE_BUNDLES, DeployExecutor, _compose_env


def _noop_phase_update(phase: Phase, status: PhaseStatus) -> None:
    pass


def _make_result(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


@pytest.mark.unit
def test_compose_env_loads_contract_rendered_runtime_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deploy-agent compose commands must receive contract-rendered policy env."""
    policy_env = tmp_path / "runtime-policy.env"
    policy_env.write_text(
        "ONEX_ACTIVE_RUNTIME_PACKAGES=omnibase_infra,omnimarket\n"
        "DEV_RUNTIME_MAIN_CAPABILITIES=market.skill-proof,runtime.main\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("deploy_agent.executor.RUNTIME_POLICY_ENV_FILE", policy_env)
    monkeypatch.delenv("ONEX_ACTIVE_RUNTIME_PACKAGES", raising=False)
    monkeypatch.delenv("DEV_RUNTIME_MAIN_CAPABILITIES", raising=False)

    env = _compose_env()

    assert env["ONEX_ACTIVE_RUNTIME_PACKAGES"] == "omnibase_infra,omnimarket"
    assert env["DEV_RUNTIME_MAIN_CAPABILITIES"] == "market.skill-proof,runtime.main"


@pytest.mark.unit
def test_compose_env_unquotes_runtime_policy_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deploy-agent must not pass shell quotes through to Compose process env."""
    policy_env = tmp_path / "runtime-policy.env"
    policy_env.write_text(
        "STABILITY_TEST_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_JSON="
        '\'{"enable_convention_fallback":false,"mappings":[{"logical_name":"llm.openrouter.api_key","source":{"source_type":"env","source_path":"OPENROUTER_API_KEY"}}]}\'\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("deploy_agent.executor.RUNTIME_POLICY_ENV_FILE", policy_env)
    monkeypatch.delenv(
        "STABILITY_TEST_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_JSON",
        raising=False,
    )

    env = _compose_env()
    resolver_config = json.loads(
        env["STABILITY_TEST_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_JSON"]
    )

    assert resolver_config["enable_convention_fallback"] is False
    assert resolver_config["mappings"][0]["logical_name"] == "llm.openrouter.api_key"


@pytest.mark.unit
class TestComposeGen:
    """compose_gen invokes the catalog CLI and writes the BUILD ARTIFACT."""

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

    def test_compose_gen_writes_to_the_build_artifact_not_the_tracked_file(
        self,
    ) -> None:
        """--output is the gitignored build artifact, never the tracked base.

        OMN-17291. The tracked compose base is written by git and by nothing
        else; a generator pointed at it gives that file two writers whose
        contents differ, and the loser is whichever one ran first.
        """
        from deploy_agent.executor import COMPOSE_FILE, COMPOSE_GEN_OUTPUT_FILE

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
        assert cmd[output_idx] == COMPOSE_GEN_OUTPUT_FILE, (
            f"Expected --output {COMPOSE_GEN_OUTPUT_FILE!r}, got {cmd[output_idx]!r}"
        )
        assert cmd[output_idx] != COMPOSE_FILE
        assert cmd[output_idx].endswith("docker/docker-compose.generated.yml")

    def test_compose_gen_validates_the_render_before_the_lane_stack(self) -> None:
        """The render is validated on its own, then the lane's real stack.

        Validating only the lane stack would leave a broken catalog render
        undetected now that the render is no longer the file the lane deploys.
        """
        from deploy_agent.executor import COMPOSE_FILE, COMPOSE_GEN_OUTPUT_FILE

        executor = DeployExecutor()
        captured_cmds: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs
        ) -> subprocess.CompletedProcess:
            captured_cmds.append(cmd)
            return _make_result()

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            executor.compose_gen(["core", "runtime"], _noop_phase_update)

        config_cmds = [c for c in captured_cmds if "config" in c and "--quiet" in c]
        assert len(config_cmds) == 2, (
            f"Expected a render validation and a lane validation, got: {config_cmds}"
        )
        render_cmd, lane_cmd = config_cmds
        assert COMPOSE_GEN_OUTPUT_FILE in render_cmd
        assert COMPOSE_FILE not in render_cmd
        # The render carries ${VAR:?} names that resolve from no committed
        # source and that this agent's own environment does not hold, so an
        # interpolating check would fail every deploy. Schema validation is the
        # strongest check that is actually true here.
        assert "--no-interpolate" in render_cmd
        assert "--no-interpolate" not in lane_cmd
        assert COMPOSE_FILE in lane_cmd
        assert COMPOSE_GEN_OUTPUT_FILE not in lane_cmd

    def test_compose_gen_render_failure_is_fatal(self) -> None:
        """An unrenderable catalog stops the deploy; it never reaches a lane."""
        from deploy_agent.executor import COMPOSE_GEN_OUTPUT_FILE

        executor = DeployExecutor()

        def fake_run(
            cmd: list[str], timeout: int, **kwargs
        ) -> subprocess.CompletedProcess:
            if "config" in cmd and COMPOSE_GEN_OUTPUT_FILE in cmd:
                return _make_result(returncode=1, stderr="required variable is unset")
            return _make_result()

        with (
            patch("deploy_agent.executor._run", side_effect=fake_run),
            pytest.raises(RuntimeError, match="invalid catalog render"),
        ):
            executor.compose_gen(["core", "runtime"], _noop_phase_update)

    def test_compose_env_injects_no_parse_only_sentinel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_compose_env invents no value to make a compose file parse.

        OMN-17291. Four sentinels used to be injected here so `config` could
        parse the catalog render that compose_gen had just written over the
        tracked compose file. With the render off that path, no compose command
        this agent issues references those names -- they appear in zero
        committed compose files -- and the sentinels were never sufficient
        anyway: the render still fails interpolation on ONEX_TENANT_DB_URL with
        all four supplied, in this agent's own environment.
        """
        for name in (
            "CI_CALLBACK_TOKEN",
            "LINEAR_WEBHOOK_SECRET",
            "WAITLIST_NOTIFIER_SLACK_BOT_TOKEN",
            "WAITLIST_NOTIFIER_SLACK_CHANNEL_ID",
        ):
            monkeypatch.delenv(name, raising=False)

        env = _compose_env()

        injected = {
            name: value
            for name, value in env.items()
            if value == "deploy-agent-compose-parse-only"
        }
        assert not injected, (
            f"_compose_env still injects parse-only sentinels: {sorted(injected)}"
        )
        assert "CI_CALLBACK_TOKEN" not in env

    def test_compose_env_still_derives_the_real_injection_dsn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deleting the sentinels must not take the real derived DSN with them."""
        monkeypatch.delenv(
            "OMNIBASE_INFRA_INJECTION_EFFECTIVENESS_POSTGRES_DSN", raising=False
        )
        monkeypatch.delenv("OMNIDASH_ANALYTICS_DB_URL", raising=False)

        dsn = _compose_env()["OMNIBASE_INFRA_INJECTION_EFFECTIVENESS_POSTGRES_DSN"]

        assert dsn.startswith("postgresql://")
        assert dsn.endswith("/omnidash_analytics")

    def test_compose_gen_prefers_repo_src_on_pythonpath(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """compose_gen must not import catalog code from a stale canonical clone."""
        from deploy_agent.executor import REPO_DIR

        executor = DeployExecutor()
        captured_envs: list[dict[str, str]] = []
        monkeypatch.setenv("PYTHONPATH", "/stale/canonical/src")

        def fake_run(
            cmd: list[str], timeout: int, **kwargs
        ) -> subprocess.CompletedProcess:
            captured_envs.append(kwargs["env"])
            return _make_result()

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            executor.compose_gen(["core", "runtime"], _noop_phase_update)

        captured_env = captured_envs[0]
        assert captured_env["PYTHONPATH"].split(":")[:2] == [
            f"{REPO_DIR}/src",
            "/stale/canonical/src",
        ]

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

    def test_compose_gen_validates_generated_lane_compose(self) -> None:
        """A successful compose_gen must validate the lane compose artifact."""
        executor = DeployExecutor()
        captured_cmds: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs
        ) -> subprocess.CompletedProcess:
            captured_cmds.append(cmd)
            return _make_result()

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            executor.compose_gen(
                ["core", "runtime"],
                _noop_phase_update,
                lane=EnumRuntimeLane.STABILITY_TEST,
            )

        # captured_cmds[1] is the render validation (OMN-17291); the lane
        # validation is the second `config --quiet` call.
        validate_cmd = captured_cmds[2]
        assert validate_cmd[:2] == ["docker", "compose"]
        assert "docker-compose.stability-test.yml" in " ".join(validate_cmd)
        assert "--profile" in validate_cmd
        assert validate_cmd[validate_cmd.index("--profile") + 1] == "runtime"
        assert validate_cmd[-2:] == ["config", "--quiet"]

    def test_compose_gen_rejects_invalid_generated_lane_compose(self) -> None:
        """compose_gen must fail before deploy effects if validation fails."""
        executor = DeployExecutor()
        phase_updates: list[tuple[Phase, PhaseStatus]] = []

        def track_updates(phase: Phase, status: PhaseStatus) -> None:
            phase_updates.append((phase, status))

        from deploy_agent.executor import COMPOSE_FILE

        def fake_run(
            cmd: list[str], timeout: int, **kwargs
        ) -> subprocess.CompletedProcess:
            # Fail only the LANE validation, so this test keeps asserting the
            # property it was written for and not the render check beside it.
            if "config" in cmd and COMPOSE_FILE in cmd:
                return _make_result(
                    returncode=1, stderr="service x has neither image nor build"
                )
            return _make_result(stdout="generated")

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            with pytest.raises(RuntimeError, match="invalid lane compose"):
                executor.compose_gen(
                    ["core", "runtime"],
                    track_updates,
                    lane=EnumRuntimeLane.STABILITY_TEST,
                )

        assert (
            Phase.COMPOSE_GEN,
            PhaseStatus.SUCCESS,
        ) not in phase_updates

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
