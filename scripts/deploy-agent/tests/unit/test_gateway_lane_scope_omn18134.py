# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18134 -- the gateway lane joins the deploy agent's DEV redeploy scope.

OMN-18108 brought the eight dev-lane-only runtime services under the agent.
The gateway forwarder was explicitly out of that ticket's scope for a reason
that is structural, not incidental: it is a SEPARATE COMPOSE PROJECT
(``omninode-gateway``), declared in ``docker/docker-compose.gateway.yml``,
which appears in no lane's ``compose_files``. Its only deploy path was a
hand-run ``scripts/deploy-gateway.sh``.

Two properties are asserted here and they pull in opposite directions, which
is why both are needed:

1. The gateway IS in the DEV lane's declared deploy scope (AC3). A DEV runtime
   deploy that leaves it behind is the OMN-18108 defect on a different
   service -- the forwarder keeps running a stale image indefinitely under
   ``restart: unless-stopped`` and nothing reports it.

2. The gateway is NEVER handed to the ``omnibase-infra`` compose project. That
   project's merged compose does not declare the service name at all, so
   ``docker compose -p omnibase-infra up -d --no-deps gateway-forwarder``
   aborts on ``no such service`` and takes the whole runtime phase with it --
   the exact FATAL failure the OMN-18108 comment in ``events.py`` warns about
   for its own list.

The reconciliation of the two is that the gateway is deployed by CALLING the
sanctioned ``scripts/deploy-gateway.sh``, which owns the gateway project's
build, digest pin, host-file sync, rollback record and systemd reload. The
agent contributes scope, sequencing and a fail-closed precondition; it does not
reimplement any of that.

AC1's "never from env" half is asserted directly: the two operator-supplied
maps (``GATEWAY_BROKER_REF_MAP_FILE``, ``GATEWAY_LANE_CREDENTIAL_MAP_FILE``)
are host files declared in the gateway's own env file, and no ``GATEWAY_``
value the agent happens to be carrying is allowed to reach the script's
environment and silently win.

AC2 -- prod/stability scope byte-unchanged -- is asserted twice over: once on
the resolved service lists, and once on the actual argv the executor issues,
because a list that is equal and an invocation that is equal are different
claims.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from deploy_agent.compose_budget import _ComposeLoader
from deploy_agent.events import (
    DEV_LANE_GATEWAY_SERVICES,
    DEV_LANE_ONLY_RUNTIME_SERVICES,
    GATEWAY_COMPOSE_PROJECT,
    BuildSource,
    EnumRuntimeLane,
    ModelRebuildRequested,
    Phase,
    PhaseStatus,
    Scope,
    gateway_services_in,
    services_for_scope,
    without_gateway_services,
)
from deploy_agent.executor import (
    GATEWAY_ENV_PASSTHROUGH_VARS,
    GATEWAY_REQUIRED_MAP_VARS,
    DeployExecutor,
    GatewayDeployScriptUnavailableError,
    GatewayLaneConfigError,
    _requested_services_for_up,
)

# ``gateway_lane`` opts this module out of the conftest stub that keeps the
# mutating gateway deploy step inert everywhere else -- this is the module
# that exercises it, so it controls the seams itself.
pytestmark = [pytest.mark.unit, pytest.mark.gateway_lane]

REPO_ROOT = Path(__file__).resolve().parents[4]
GATEWAY_COMPOSE_FILE = REPO_ROOT / "docker" / "docker-compose.gateway.yml"
INFRA_COMPOSE_FILE = REPO_ROOT / "docker" / "docker-compose.infra.yml"
DEV_LANE_OVERLAY = REPO_ROOT / "docker" / "docker-compose.dev-lane.yml"
GATEWAY_DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-gateway.sh"

CORRELATION_ID = "11111111-1111-4111-8111-111111111111"


def _noop_phase_update(phase: Phase, status: PhaseStatus) -> None:
    return None


def _ok() -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


def _compose_services(path: Path) -> set[str]:
    """Return the service names one compose file declares.

    Uses the SAME loader the ceiling derivation uses (``compose_budget``), not
    ``yaml.safe_load``: the dev-lane overlay carries compose's ``!override``
    tag, which plain safe_load refuses outright -- and a test that could not
    read the overlay at all would report the gateway as absent from it for the
    wrong reason.

    ``_ComposeLoader`` subclasses ``yaml.SafeLoader`` and adds exactly one
    multi-constructor, for compose's own ``!override`` / ``!reset`` tags, which
    constructs no objects. It is safe in the sense S506 is about; ruff cannot
    see the base class through the alias.
    """
    text = path.read_text(encoding="utf-8")
    model = yaml.load(text, Loader=_ComposeLoader)  # noqa: S506 - SafeLoader subclass
    return set(model.get("services", {}))


def _captured(
    lane: EnumRuntimeLane,
    *,
    services: list[str] | None = None,
    scope: Scope = Scope.RUNTIME,
) -> tuple[list[list[str]], list[dict[str, str]]]:
    """Run a rebuild with every subprocess faked; return the argv and envs."""
    executor = DeployExecutor()
    commands: list[list[str]] = []
    envs: list[dict[str, str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        commands.append(list(cmd))
        env = kwargs.get("env")
        envs.append(dict(env) if isinstance(env, dict) else {})
        return _ok()

    with (
        patch("deploy_agent.executor._run", side_effect=_fake_run),
        patch.object(DeployExecutor, "_compose_up", return_value=None),
    ):
        executor.rebuild_scope(
            scope,
            list(services or []),
            _noop_phase_update,
            git_sha="0" * 40,
            git_ref="origin/dev",
            build_source=BuildSource.RELEASE,
            lane=lane,
        )
    return commands, envs


def _gateway_invocations(commands: list[list[str]]) -> list[list[str]]:
    return [c for c in commands if any("deploy-gateway.sh" in tok for tok in c)]


def _write_gateway_env(
    tmp_path: Path, *, omit: str | None = None, dangling: str | None = None
) -> Path:
    """Write a gateway.env declaring both maps, with one flaw injected."""
    lines: list[str] = ["GATEWAY_AWS_PROFILE=gateway"]
    for name in sorted(GATEWAY_REQUIRED_MAP_VARS):
        if name == omit:
            continue
        target = tmp_path / f"{name.lower()}.yaml"
        if name != dangling:
            target.write_text("{}\n", encoding="utf-8")
        lines.append(f"{name}={target}")
    env_file = tmp_path / "gateway.env"
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env_file


class TestPositiveControls:
    """An empty parse must never read as agreement (CLAUDE.md rule 16)."""

    def test_the_gateway_compose_file_declares_the_forwarder(self) -> None:
        assert set(DEV_LANE_GATEWAY_SERVICES) <= _compose_services(
            GATEWAY_COMPOSE_FILE
        ), (
            "the names this module scopes are not declared in "
            f"{GATEWAY_COMPOSE_FILE}; the rest of this file would pass vacuously"
        )

    def test_the_sanctioned_script_exists_and_names_the_gateway_project(self) -> None:
        text = GATEWAY_DEPLOY_SCRIPT.read_text(encoding="utf-8")
        assert f'COMPOSE_PROJECT="{GATEWAY_COMPOSE_PROJECT}"' in text


class TestTheGatewayIsNotInTheInfraProject:
    """The structural reason the gateway needs its own path at all."""

    def test_no_lane_compose_file_declares_the_gateway_services(self) -> None:
        declared = _compose_services(INFRA_COMPOSE_FILE) | _compose_services(
            DEV_LANE_OVERLAY
        )
        assert declared.isdisjoint(set(DEV_LANE_GATEWAY_SERVICES)), (
            "a gateway service is declared in the omnibase-infra compose model; "
            "the separation this ticket is built on no longer holds"
        )

    def test_the_gateway_services_are_disjoint_from_the_base_runtime_family(
        self,
    ) -> None:
        assert set(DEV_LANE_GATEWAY_SERVICES).isdisjoint(
            set(services_for_scope(Scope.RUNTIME))
        )

    def test_the_gateway_services_are_disjoint_from_the_omn18108_addendum(self) -> None:
        # Two lists, opposite compose semantics: the OMN-18108 eight ARE handed
        # to the omnibase-infra project, these are handed to a different one.
        assert set(DEV_LANE_GATEWAY_SERVICES).isdisjoint(
            set(DEV_LANE_ONLY_RUNTIME_SERVICES)
        )


class TestDevScopeCarriesTheGateway:
    """AC3. Red at the parent commit: the gateway is absent from DEV scope."""

    def test_dev_runtime_scope_covers_the_gateway(self) -> None:
        resolved = services_for_scope(Scope.RUNTIME, lane=EnumRuntimeLane.DEV)
        missing = [s for s in DEV_LANE_GATEWAY_SERVICES if s not in resolved]
        assert not missing, (
            "a DEV-lane runtime deploy does not reach the gateway, so every "
            f"agent rebuild leaves the forwarder on a stale image: {missing}"
        )

    def test_dev_full_scope_covers_the_gateway(self) -> None:
        resolved = services_for_scope(Scope.FULL, lane=EnumRuntimeLane.DEV)
        assert not [s for s in DEV_LANE_GATEWAY_SERVICES if s not in resolved]

    def test_dev_core_scope_is_not_widened(self) -> None:
        assert services_for_scope(Scope.CORE, lane=EnumRuntimeLane.DEV) == (
            services_for_scope(Scope.CORE)
        )

    def test_the_base_runtime_family_still_leads_the_dev_list(self) -> None:
        base = services_for_scope(Scope.RUNTIME)
        resolved = services_for_scope(Scope.RUNTIME, lane=EnumRuntimeLane.DEV)
        assert resolved[: len(base)] == base

    def test_omitting_the_lane_does_not_acquire_the_gateway(self) -> None:
        assert set(services_for_scope(Scope.RUNTIME)).isdisjoint(
            set(DEV_LANE_GATEWAY_SERVICES)
        )


class TestProdAndStabilityScopeIsUnchanged:
    """AC2, on the resolved lists."""

    @pytest.mark.parametrize(
        "lane", [EnumRuntimeLane.PROD, EnumRuntimeLane.STABILITY_TEST]
    )
    def test_runtime_scope_is_the_base_list(self, lane: EnumRuntimeLane) -> None:
        assert services_for_scope(Scope.RUNTIME, lane=lane) == services_for_scope(
            Scope.RUNTIME
        )

    @pytest.mark.parametrize(
        "lane", [EnumRuntimeLane.PROD, EnumRuntimeLane.STABILITY_TEST]
    )
    def test_full_scope_is_the_base_list(self, lane: EnumRuntimeLane) -> None:
        assert services_for_scope(Scope.FULL, lane=lane) == services_for_scope(
            Scope.FULL
        )

    @pytest.mark.parametrize(
        "lane", [EnumRuntimeLane.PROD, EnumRuntimeLane.STABILITY_TEST]
    )
    def test_no_gateway_service_leaks_onto_another_lane(
        self, lane: EnumRuntimeLane
    ) -> None:
        assert set(services_for_scope(Scope.FULL, lane=lane)).isdisjoint(
            set(DEV_LANE_GATEWAY_SERVICES)
        )

    @pytest.mark.parametrize(
        "lane", [EnumRuntimeLane.PROD, EnumRuntimeLane.STABILITY_TEST]
    )
    def test_a_non_dev_command_may_not_name_a_gateway_service(
        self, lane: EnumRuntimeLane
    ) -> None:
        with pytest.raises(ValueError):
            ModelRebuildRequested(
                correlation_id=CORRELATION_ID,  # type: ignore[arg-type]
                requested_by="test",
                scope=Scope.RUNTIME,
                runtime_lane=lane,
                git_ref="origin/dev",
                image_digest="sha256:" + "a" * 64,
                services=list(DEV_LANE_GATEWAY_SERVICES),
            )


class TestProdAndStabilityInvocationsAreUnchanged:
    """AC2, on the argv. Equal lists and equal commands are different claims."""

    def test_stability_issues_no_gateway_invocation(self) -> None:
        commands, _ = _captured(EnumRuntimeLane.STABILITY_TEST)
        assert _gateway_invocations(commands) == []

    def test_stability_names_no_gateway_service_in_any_command(self) -> None:
        commands, _ = _captured(EnumRuntimeLane.STABILITY_TEST)
        flat = {tok for cmd in commands for tok in cmd}
        assert flat.isdisjoint(set(DEV_LANE_GATEWAY_SERVICES))

    def test_stability_never_names_the_gateway_compose_project(self) -> None:
        commands, _ = _captured(EnumRuntimeLane.STABILITY_TEST)
        flat = {tok for cmd in commands for tok in cmd}
        assert GATEWAY_COMPOSE_PROJECT not in flat


class TestTheGatewayIsNeverHandedToTheInfraComposeProject:
    """Property 2. Membership in DEV scope must not become a compose argument."""

    def test_the_dev_up_list_excludes_the_gateway(self) -> None:
        resolved = _requested_services_for_up(
            Scope.RUNTIME, [], lane=EnumRuntimeLane.DEV
        )
        assert set(resolved).isdisjoint(set(DEV_LANE_GATEWAY_SERVICES)), (
            "the gateway reached the omnibase-infra compose project's service "
            "list; that project does not declare the name and the runtime "
            "phase aborts on `no such service`"
        )

    def test_the_dev_up_list_still_carries_the_omn18108_addendum(self) -> None:
        resolved = _requested_services_for_up(
            Scope.RUNTIME, [], lane=EnumRuntimeLane.DEV
        )
        assert not [s for s in DEV_LANE_ONLY_RUNTIME_SERVICES if s not in resolved]

    def test_an_explicit_mixed_list_is_split(self) -> None:
        requested = ["omninode-runtime", *DEV_LANE_GATEWAY_SERVICES]
        assert _requested_services_for_up(
            Scope.RUNTIME, requested, lane=EnumRuntimeLane.DEV
        ) == ["omninode-runtime"]

    def test_the_split_helpers_are_complementary(self) -> None:
        requested = ["omninode-runtime", *DEV_LANE_GATEWAY_SERVICES]
        assert without_gateway_services(requested) + gateway_services_in(requested) == [
            "omninode-runtime",
            *DEV_LANE_GATEWAY_SERVICES,
        ]


class TestTheSanctionedScriptIsCalled:
    """AC1. The agent calls scripts/deploy-gateway.sh; it reimplements nothing."""

    def test_a_dev_runtime_rebuild_invokes_the_script_with_execute(self) -> None:
        commands, _ = _captured(EnumRuntimeLane.DEV)
        invocations = _gateway_invocations(commands)
        assert len(invocations) == 1, (
            "expected exactly one gateway deploy invocation on a DEV runtime "
            f"rebuild, got {invocations}"
        )
        assert "--execute" in invocations[0]

    def test_a_dev_full_rebuild_invokes_the_script(self) -> None:
        commands, _ = _captured(EnumRuntimeLane.DEV, scope=Scope.FULL)
        assert len(_gateway_invocations(commands)) == 1

    def test_a_dev_core_rebuild_does_not(self) -> None:
        commands, _ = _captured(EnumRuntimeLane.DEV, scope=Scope.CORE)
        assert _gateway_invocations(commands) == []

    def test_the_gateway_step_runs_after_the_infra_family(self) -> None:
        # The forwarder mirrors off the dev lane's broker; deploying it before
        # the lane it forwards from is a self-inflicted delivery gap.
        commands, _ = _captured(EnumRuntimeLane.DEV)
        gateway_index = next(
            i
            for i, cmd in enumerate(commands)
            if any("deploy-gateway.sh" in tok for tok in cmd)
        )
        build_indexes = [i for i, cmd in enumerate(commands) if "build" in cmd]
        assert build_indexes and gateway_index > max(build_indexes)

    def test_the_script_runs_from_the_deploy_source_clone(self) -> None:
        # AC1's "same sha": the script resolves its own repo root from $0, so
        # invoking the copy inside the clone the GIT phase just reset is what
        # binds the gateway build to the deployed sha. Nothing is passed.
        from deploy_agent import executor as executor_mod

        commands, _ = _captured(EnumRuntimeLane.DEV)
        invocation = _gateway_invocations(commands)[0]
        assert any(tok == executor_mod.deploy_gateway_script() for tok in invocation), (
            invocation
        )

    def test_a_gateway_only_dev_command_skips_the_infra_family(self) -> None:
        commands, _ = _captured(
            EnumRuntimeLane.DEV, services=list(DEV_LANE_GATEWAY_SERVICES)
        )
        assert len(_gateway_invocations(commands)) == 1
        assert [c for c in commands if "build" in c] == [], (
            "a gateway-only command rebuilt the whole runtime family; an empty "
            "residual service list must never read as 'everything'"
        )

    def test_a_dev_command_may_name_a_gateway_service(self) -> None:
        cmd = ModelRebuildRequested(
            correlation_id=CORRELATION_ID,  # type: ignore[arg-type]
            requested_by="test",
            scope=Scope.RUNTIME,
            runtime_lane=EnumRuntimeLane.DEV,
            git_ref="origin/dev",
            services=list(DEV_LANE_GATEWAY_SERVICES),
        )
        assert cmd.services == list(DEV_LANE_GATEWAY_SERVICES)

    def test_the_gateway_is_reported_as_restarted(self) -> None:
        executor = DeployExecutor()
        with (
            patch("deploy_agent.executor._run", return_value=_ok()),
            patch.object(DeployExecutor, "_compose_up", return_value=None),
        ):
            restarted = executor.rebuild_scope(
                Scope.RUNTIME,
                [],
                _noop_phase_update,
                git_sha="0" * 40,
                git_ref="origin/dev",
                build_source=BuildSource.RELEASE,
                lane=EnumRuntimeLane.DEV,
            )
        assert not [s for s in DEV_LANE_GATEWAY_SERVICES if s not in restarted]


class TestTheMapsNeverTravelThroughTheAgentEnvironment:
    """AC1's "never from env" half, asserted rather than assumed."""

    def test_no_required_map_variable_reaches_the_script_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in GATEWAY_REQUIRED_MAP_VARS:
            monkeypatch.setenv(name, "/nonexistent/an-agent-supplied-value.yaml")
        commands, envs = _captured(EnumRuntimeLane.DEV)
        index = next(
            i
            for i, cmd in enumerate(commands)
            if any("deploy-gateway.sh" in tok for tok in cmd)
        )
        leaked = sorted(set(GATEWAY_REQUIRED_MAP_VARS) & set(envs[index]))
        assert not leaked, (
            "the agent's own environment supplied gateway map paths to the "
            f"deploy script: {leaked}. The maps are host files declared in the "
            "gateway env file and must be read only from there."
        )

    def test_the_script_location_overrides_are_passed_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The strip is scoped: it removes gateway CONFIG, not the three
        # variables that tell the script where the lane's files live. Removing
        # those would silently send an overridden deploy back to the defaults.
        for name in GATEWAY_ENV_PASSTHROUGH_VARS:
            monkeypatch.setenv(name, f"/scratch/{name.lower()}")
        commands, envs = _captured(EnumRuntimeLane.DEV)
        index = next(
            i
            for i, cmd in enumerate(commands)
            if any("deploy-gateway.sh" in tok for tok in cmd)
        )
        for name in GATEWAY_ENV_PASSTHROUGH_VARS:
            assert envs[index].get(name) == f"/scratch/{name.lower()}"


class TestFailClosed:
    """A gateway step that cannot prove its preconditions refuses the deploy."""

    def test_a_missing_script_raises_its_own_error_class(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from deploy_agent import executor as executor_mod

        monkeypatch.setattr(
            executor_mod,
            "deploy_gateway_script",
            lambda: str(tmp_path / "absent" / "deploy-gateway.sh"),
        )
        with (
            patch("deploy_agent.executor._run", return_value=_ok()),
            patch.object(DeployExecutor, "_compose_up", return_value=None),
            pytest.raises(GatewayDeployScriptUnavailableError),
        ):
            DeployExecutor().rebuild_scope(
                Scope.RUNTIME,
                [],
                _noop_phase_update,
                git_sha="0" * 40,
                git_ref="origin/dev",
                build_source=BuildSource.RELEASE,
                lane=EnumRuntimeLane.DEV,
            )

    def test_a_missing_gateway_env_file_refuses(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from deploy_agent import executor as executor_mod

        monkeypatch.setattr(
            executor_mod, "gateway_env_file", lambda: str(tmp_path / "absent.env")
        )
        with (
            patch("deploy_agent.executor._run", return_value=_ok()),
            patch.object(DeployExecutor, "_compose_up", return_value=None),
            pytest.raises(GatewayLaneConfigError),
        ):
            DeployExecutor().rebuild_scope(
                Scope.RUNTIME,
                [],
                _noop_phase_update,
                git_sha="0" * 40,
                git_ref="origin/dev",
                build_source=BuildSource.RELEASE,
                lane=EnumRuntimeLane.DEV,
            )

    @pytest.mark.parametrize("omitted", sorted(GATEWAY_REQUIRED_MAP_VARS))
    def test_an_undeclared_map_refuses_and_names_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, omitted: str
    ) -> None:
        from deploy_agent import executor as executor_mod

        env_file = _write_gateway_env(tmp_path, omit=omitted)
        monkeypatch.setattr(executor_mod, "gateway_env_file", lambda: str(env_file))
        with (
            patch("deploy_agent.executor._run", return_value=_ok()),
            patch.object(DeployExecutor, "_compose_up", return_value=None),
            pytest.raises(GatewayLaneConfigError, match=omitted),
        ):
            DeployExecutor().rebuild_scope(
                Scope.RUNTIME,
                [],
                _noop_phase_update,
                git_sha="0" * 40,
                git_ref="origin/dev",
                build_source=BuildSource.RELEASE,
                lane=EnumRuntimeLane.DEV,
            )

    @pytest.mark.parametrize("absent", sorted(GATEWAY_REQUIRED_MAP_VARS))
    def test_a_declared_but_absent_map_file_refuses(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, absent: str
    ) -> None:
        from deploy_agent import executor as executor_mod

        env_file = _write_gateway_env(tmp_path, dangling=absent)
        monkeypatch.setattr(executor_mod, "gateway_env_file", lambda: str(env_file))
        with (
            patch("deploy_agent.executor._run", return_value=_ok()),
            patch.object(DeployExecutor, "_compose_up", return_value=None),
            pytest.raises(GatewayLaneConfigError, match=absent),
        ):
            DeployExecutor().rebuild_scope(
                Scope.RUNTIME,
                [],
                _noop_phase_update,
                git_sha="0" * 40,
                git_ref="origin/dev",
                build_source=BuildSource.RELEASE,
                lane=EnumRuntimeLane.DEV,
            )

    def test_a_failing_script_fails_the_deploy(self) -> None:
        def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
            if any("deploy-gateway.sh" in tok for tok in cmd):
                return subprocess.CompletedProcess(
                    args=cmd, returncode=1, stdout="", stderr="reload failed"
                )
            return _ok()

        with (
            patch("deploy_agent.executor._run", side_effect=_fake_run),
            patch.object(DeployExecutor, "_compose_up", return_value=None),
            pytest.raises(RuntimeError, match="GATEWAY_DEPLOY_FAILED"),
        ):
            DeployExecutor().rebuild_scope(
                Scope.RUNTIME,
                [],
                _noop_phase_update,
                git_sha="0" * 40,
                git_ref="origin/dev",
                build_source=BuildSource.RELEASE,
                lane=EnumRuntimeLane.DEV,
            )
