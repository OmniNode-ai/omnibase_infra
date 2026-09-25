# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19522: the executor runs the dev-202 instance's own composition.

Task B3 of the second-deploy-slot plan (epic OMN-19500). omnibase_infra#4097
(OMN-19506) taught the agent which routing INSTANCE it is and gave each instance
its own consumer group, but the executor still keyed its lane config by the wire
lane alone. The wire lane of a dev-202 command is ``dev`` (routing, not a new
enum value, picks the instance), so an agent that IS dev-202 rendered the .201
composition -- compose project ``omnibase-infra``, health ports 8085/8086 -- on
.202, and ran four phases that exist only on .201: the gateway deploy, the k3s
onex-lab overlay (and its repair build), the onex-api pin delivery and the
Infisical seed.

The test names carry the ticket's falsifier selectors:

* ``dev_202_lane_config``  -- AC1, the instance's composition;
* ``dev_202_phases_off``   -- AC2, the .201-only phases and services never run;
* ``dev_201_unchanged``    -- AC3, the control: dev-201 and a router-less agent
  are exactly what they were;
* ``dev_202_unit``         -- AC4, the unit file;
* ``dev_202_env_template`` -- AC5, the names-only env template.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import pytest
import yaml
from deploy_agent import agent as agent_mod
from deploy_agent import executor as executor_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.compose_budget import _ComposeLoader
from deploy_agent.events import (
    DEV_LANE_GATEWAY_SERVICES,
    DEV_LANE_ONLY_MIGRATION_SERVICES,
    EnumRuntimeLane,
    ModelRebuildRequested,
    Phase,
    PhaseStatus,
    Scope,
)
from deploy_agent.executor import (
    COMPOSE_FILE,
    COMPOSE_PROJECT,
    DEV_INSTANCE_LANE_CONFIGS,
    RUNTIME_HEALTH_TARGETS,
    DeployExecutor,
    DevLaneMigrationPreflightError,
    EnumInstancePhase,
    GatewayDeployScriptUnavailableError,
    _requested_services_for_up,
    active_dev_instance,
    lane_config_for,
    lane_runs_phase,
    select_dev_instance,
)
from deploy_agent.job_state import JobStore
from deploy_agent.routing import load_routing_table

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DOCKER = _REPO_ROOT / "docker"
_DEPLOY_DIR = Path(__file__).resolve().parents[2] / "deploy"
_DEV_202_OVERLAY = _DOCKER / "docker-compose.dev-202.yml"
_DEV_202_UNIT = _DEPLOY_DIR / "deploy-agent-dev-202.service"
_DEV_UNIT = _DEPLOY_DIR / "deploy-agent-dev.service"
_ENV_TEMPLATE = _DEPLOY_DIR / "dev-202.env.template"

SHA = "a" * 40

#: Every service the dev-202 overlay disables by profile. Naming one of them in
#: a compose argv would auto-activate its profile and start it on .202.
_DEV_202_DISABLED = {"onex-api", "cloud-migration-files", "cloud-migration"}

#: The four phases that exist only on .201.
_HOST_201_PHASES = set(EnumInstancePhase)


def _noop(phase: Phase, status: PhaseStatus) -> None:
    return None


def _ok() -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


def _capture_compose_up(
    executor: DeployExecutor, scope: Scope = Scope.RUNTIME
) -> list[list[str]]:
    captured: list[list[str]] = []

    def fake_run(
        cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        captured.append(cmd)
        return _ok()

    with (
        patch("deploy_agent.executor._run", side_effect=fake_run),
        patch.object(executor, "_ensure_runtime_migrations_ready"),
        patch("deploy_agent.executor.verify_containers_up", return_value=(True, [])),
    ):
        executor._compose_up(Phase.RUNTIME, scope, [], _noop, lane=EnumRuntimeLane.DEV)
    return captured


def _capture_build(executor: DeployExecutor) -> list[list[str]]:
    captured: list[list[str]] = []

    def fake_run(
        cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        captured.append(cmd)
        return _ok()

    with patch("deploy_agent.executor._run", side_effect=fake_run):
        executor._compose_build(
            Scope.RUNTIME, SHA, _noop, runtime_lane=EnumRuntimeLane.DEV
        )
    return [c for c in captured if "build" in c]


def _capture_migrations(executor: DeployExecutor) -> list[list[str]]:
    captured: list[list[str]] = []

    def fake_run(
        cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        captured.append(cmd)
        if "to_regclass" in " ".join(cmd):
            # The projection-table probe answers "the table exists".
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="t\n", stderr=""
            )
        return _ok()

    with (
        patch("deploy_agent.executor._run", side_effect=fake_run),
        patch("deploy_agent.executor.verify_containers_up", return_value=(True, [])),
        patch(
            "deploy_agent.executor.verify_oneshots_completed", return_value=(True, [])
        ),
    ):
        executor._ensure_runtime_migrations_ready(lane=EnumRuntimeLane.DEV)
    return captured


def _project_of(cmd: list[str]) -> str:
    return cmd[cmd.index("-p") + 1]


def _files_of(cmd: list[str]) -> list[str]:
    return [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-f"]


# --------------------------------------------------------------------------- #
# AC1 -- the dev-202 instance's composition                                    #
# --------------------------------------------------------------------------- #
class TestDev202LaneConfig:
    def test_dev_202_lane_config_names_its_own_project_files_and_targets(
        self,
    ) -> None:
        select_dev_instance("dev-202")
        cfg = lane_config_for(EnumRuntimeLane.DEV)

        assert cfg.compose_project == "omnibase-infra-dev-202"
        assert cfg.build_project == "omnibase-infra-dev-202"
        assert cfg.compose_files[0] == COMPOSE_FILE
        assert [Path(f).name for f in cfg.compose_files] == [
            "docker-compose.infra.yml",
            "docker-compose.dev-lane.yml",
            "docker-compose.dev-202.yml",
        ]
        assert cfg.postgres_container == "omnibase-infra-dev-202-postgres"
        assert [port for _, port in cfg.runtime_health_targets] == [61085, 61086]
        assert cfg.main_runtime_container == "omninode-dev-202-runtime"

    def test_dev_202_lane_config_agrees_with_the_overlay_it_renders(self) -> None:
        """Every literal above is read back out of docker-compose.dev-202.yml.

        On the dev lane ``runtime_health_targets`` carries compose SERVICE
        names (the verify recreate looks them up by label), so the service half
        must be a service the overlay declares and the port half the host port
        that service publishes.
        """
        cfg = DEV_INSTANCE_LANE_CONFIGS["dev-202"]
        overlay = yaml.load(_DEV_202_OVERLAY.read_text(), Loader=_ComposeLoader)  # noqa: S506
        services: dict[str, Any] = overlay["services"]

        assert overlay["name"] == cfg.compose_project
        assert services["postgres"]["container_name"] == cfg.postgres_container
        assert (
            services["omninode-runtime"]["container_name"] == cfg.main_runtime_container
        )
        for service, port in cfg.runtime_health_targets:
            published = [str(p).split(":")[-2] for p in services[service]["ports"]]
            assert str(port) in published, (service, port, published)

    def test_dev_202_lane_config_compose_up_uses_the_triple_and_its_project(
        self,
    ) -> None:
        select_dev_instance("dev-202")
        cmd = _capture_compose_up(DeployExecutor())[0]

        assert _project_of(cmd) == "omnibase-infra-dev-202"
        assert [Path(f).name for f in _files_of(cmd)] == [
            "docker-compose.infra.yml",
            "docker-compose.dev-lane.yml",
            "docker-compose.dev-202.yml",
        ]

    def test_dev_202_lane_config_builds_under_its_own_project(self) -> None:
        """Compose names a build's image ``<project>-<service>``. Built under
        ``omnibase-infra`` and brought up under ``omnibase-infra-dev-202``, the
        up would find no image and build a second one without the build args."""
        select_dev_instance("dev-202")
        builds = _capture_build(DeployExecutor())

        assert builds
        assert {_project_of(c) for c in builds} == {"omnibase-infra-dev-202"}

    def test_dev_202_lane_config_every_routing_instance_has_one(self) -> None:
        """The routing table's instances and the executor's are the same set,
        so an instance the table names always has a composition to run."""
        table = load_routing_table(_REPO_ROOT)
        assert set(table.instances) == set(DEV_INSTANCE_LANE_CONFIGS)

    def test_dev_202_lane_config_refuses_an_unknown_instance(self) -> None:
        with pytest.raises(ValueError, match="dev-999"):
            select_dev_instance("dev-999")
        assert active_dev_instance() == "dev-201"

    def test_dev_202_lane_config_is_selected_by_agent_construction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEPLOY_AGENT_INSTANCE", "dev-202")
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
        monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")

        DeployAgent(skip_self_update=True)

        assert active_dev_instance() == "dev-202"
        assert (
            lane_config_for(EnumRuntimeLane.DEV).compose_project
            == "omnibase-infra-dev-202"
        )
        assert (
            agent_mod._runtime_container_for_lane(EnumRuntimeLane.DEV)
            == "omninode-dev-202-runtime"
        )


# --------------------------------------------------------------------------- #
# Agent-level doubles for AC2 / AC3                                            #
# --------------------------------------------------------------------------- #
class _FakeExecutor:
    def __init__(self, *, rebuild_error: Exception | None = None) -> None:
        self.calls: list[str] = []
        self._rebuild_error = rebuild_error
        self.container_residue: list[object] = []
        self.sibling_source_refs: dict[str, str] = {}
        self.recreate_supervision: list[object] = []
        self.verify_recreate: list[object] = []
        self.deps_convergence: list[object] = []
        self.compose_invocations: list[object] = []
        self.health_checks: list[object] = []

    def preflight(self, **kwargs: object) -> None:
        self.calls.append("preflight")

    def git_pull(self, git_ref: str, **kwargs: object) -> str:
        self.calls.append("git_pull")
        return SHA

    def compose_gen(self, bundles: list[str], **kwargs: object) -> None:
        self.calls.append("compose_gen")

    def seed_infisical(self, **kwargs: object) -> None:
        # The executor decides whether the seed runs on this instance; what
        # the agent owes it is the lane.
        self.calls.append(f"seed_infisical lane={kwargs.get('lane')}")

    def validate_llm_endpoint_env_contract(self) -> None:
        self.calls.append("validate_llm_endpoint_env_contract")

    def rebuild_scope(self, *args: object, **kwargs: object) -> list[str]:
        self.calls.append("rebuild_scope")
        if self._rebuild_error is not None:
            raise self._rebuild_error
        return ["omninode-runtime"]

    def verify(self, **kwargs: object) -> list[object]:
        self.calls.append("verify")
        return []

    def deliver_onex_api_pin(self, **kwargs: object) -> object:
        self.calls.append("deliver_onex_api_pin")
        raise AssertionError("the onex-api pin delivery must not run here")


class _FakeApplier:
    calls: list[str] = []

    def __init__(self, **kwargs: Any) -> None:
        self.manifest_sha: str | None = "c" * 40

    def apply(self, *, sha: str, stamp: str, correlation_id: str) -> Path:
        _FakeApplier.calls.append("apply")
        return Path("/state/lab-overlay/x.json")

    def build_repair_migrate_image(
        self, *, sha: str, stamp: str, correlation_id: str
    ) -> Path:
        _FakeApplier.calls.append("repair")
        return Path("/state/lab-overlay/x.json")


@pytest.fixture
def _fake_applier(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeApplier.calls = []
    monkeypatch.setattr(agent_mod, "LabOverlayApplier", _FakeApplier)
    monkeypatch.setattr(agent_mod, "LAB_OVERLAY_ENABLED", True)


def _run_dev_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    instance: str,
    rebuild_error: Exception | None = None,
) -> tuple[DeployAgent, _FakeExecutor, JobStore, ModelRebuildRequested]:
    monkeypatch.setenv("DEPLOY_AGENT_INSTANCE", instance)
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    monkeypatch.setattr(agent_mod, "publish_result", lambda payload, config: False)
    cmd = ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.DEV,
    )
    store = JobStore(tmp_path)
    store.accept(cmd.correlation_id, cmd.model_dump(mode="json"))
    agent = DeployAgent(skip_self_update=True)
    fake = _FakeExecutor(rebuild_error=rebuild_error)
    agent.job_store = store
    agent.executor = fake  # type: ignore[assignment]
    agent._run_deploy(cmd)
    return agent, fake, store, cmd


# --------------------------------------------------------------------------- #
# AC2 -- the .201-only phases and services never run on dev-202                #
# --------------------------------------------------------------------------- #
class TestDev202PhasesOff:
    def test_dev_202_phases_off_declared_for_all_four(self) -> None:
        cfg = DEV_INSTANCE_LANE_CONFIGS["dev-202"]
        assert cfg.disabled_phases == frozenset(_HOST_201_PHASES)
        select_dev_instance("dev-202")
        for phase in EnumInstancePhase:
            assert not lane_runs_phase(EnumRuntimeLane.DEV, phase)

    @pytest.mark.usefixtures("_fake_applier")
    def test_dev_202_phases_off_on_a_successful_job(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, fake, store, cmd = _run_dev_job(
            tmp_path, monkeypatch, instance="dev-202"
        )

        job = store.load(cmd.correlation_id)
        assert job is not None and job.status == "success"
        assert "seed_infisical lane=dev" in fake.calls
        assert "deliver_onex_api_pin" not in fake.calls
        assert _FakeApplier.calls == []
        assert agent._onex_api_delivery is None

    @pytest.mark.usefixtures("_fake_applier")
    def test_dev_202_phases_off_on_a_failed_migration_preflight(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The repair build is the lab overlay's failing-path half."""
        _run_dev_job(
            tmp_path,
            monkeypatch,
            instance="dev-202",
            rebuild_error=DevLaneMigrationPreflightError("preflight"),
        )
        assert _FakeApplier.calls == []

    @pytest.mark.gateway_lane
    def test_dev_202_phases_off_gateway_deploy_is_a_no_op(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On dev-201 a missing gateway script raises; on dev-202 nothing is
        even looked up, because there is no gateway lane on .202."""
        monkeypatch.setattr(
            executor_mod, "deploy_gateway_script", lambda: "/nonexistent/deploy.sh"
        )
        select_dev_instance("dev-202")
        DeployExecutor()._deploy_gateway_lane(
            _noop,
            lane=EnumRuntimeLane.DEV,
            build_source="release",
            targets=list(DEV_LANE_GATEWAY_SERVICES),
            git_ref=SHA,
        )

    def test_dev_202_phases_off_seed_is_skipped_by_the_executor_too(self) -> None:
        select_dev_instance("dev-202")
        statuses: list[tuple[Phase, PhaseStatus]] = []
        with patch("deploy_agent.executor._run") as run:
            DeployExecutor().seed_infisical(
                on_phase_update=lambda p, s: statuses.append((p, s)),
                lane=EnumRuntimeLane.DEV,
            )
        run.assert_not_called()
        assert statuses == [(Phase.SEED, PhaseStatus.SKIPPED)]

    def test_dev_202_phases_off_disabled_services_are_never_compose_arguments(
        self,
    ) -> None:
        select_dev_instance("dev-202")
        requested = set(
            _requested_services_for_up(Scope.RUNTIME, [], lane=EnumRuntimeLane.DEV)
        )
        assert requested
        assert not requested & _DEV_202_DISABLED

        up = _capture_compose_up(DeployExecutor())[0]
        assert not set(up) & _DEV_202_DISABLED

        migrations = _capture_migrations(DeployExecutor())
        named = {tok for cmd in migrations for tok in cmd}
        assert not named & set(DEV_LANE_ONLY_MIGRATION_SERVICES)
        assert "forward-migration" in named

    def test_dev_202_phases_off_disabled_services_match_the_overlay(self) -> None:
        """What the executor refuses to name is exactly what the overlay
        disables by profile, among the services a dev deploy is responsible
        for."""
        overlay = yaml.load(_DEV_202_OVERLAY.read_text(), Loader=_ComposeLoader)  # noqa: S506
        disabled_by_overlay = {
            name
            for name, spec in overlay["services"].items()
            if (spec or {}).get("profiles") == ["dev-202-disabled"]
        }
        cfg = DEV_INSTANCE_LANE_CONFIGS["dev-202"]
        assert disabled_by_overlay >= _DEV_202_DISABLED
        assert cfg.disabled_services == frozenset(
            disabled_by_overlay | set(DEV_LANE_GATEWAY_SERVICES)
        )

    def test_dev_202_phases_off_restarted_services_exclude_disabled(self) -> None:
        select_dev_instance("dev-202")
        executor = DeployExecutor()
        with (
            patch.object(executor, "_compose_build"),
            patch.object(executor, "_compose_up"),
        ):
            restarted = executor.rebuild_scope(
                Scope.RUNTIME, [], _noop, git_sha=SHA, lane=EnumRuntimeLane.DEV
            )
        assert restarted
        assert (
            not set(restarted) & DEV_INSTANCE_LANE_CONFIGS["dev-202"].disabled_services
        )


# --------------------------------------------------------------------------- #
# AC3 -- dev-201 and a router-less agent are unchanged                         #
# --------------------------------------------------------------------------- #
class TestDev201Unchanged:
    def test_dev_201_unchanged_config_is_the_legacy_dev_lane(self) -> None:
        assert active_dev_instance() == "dev-201"
        default = lane_config_for(EnumRuntimeLane.DEV)
        select_dev_instance("dev-201")
        cfg = lane_config_for(EnumRuntimeLane.DEV)

        assert cfg == default
        assert cfg.compose_project == COMPOSE_PROJECT == "omnibase-infra"
        assert cfg.build_project == "omnibase-infra"
        assert [Path(f).name for f in cfg.compose_files] == [
            "docker-compose.infra.yml",
            "docker-compose.dev-lane.yml",
        ]
        assert cfg.postgres_container == "omnibase-infra-postgres"
        assert cfg.runtime_health_targets == RUNTIME_HEALTH_TARGETS
        assert cfg.main_runtime_container == "omninode-runtime"
        assert cfg.disabled_phases == frozenset()
        assert cfg.disabled_services == frozenset()
        for phase in EnumInstancePhase:
            assert lane_runs_phase(EnumRuntimeLane.DEV, phase)

    def test_dev_201_unchanged_compose_argv(self) -> None:
        select_dev_instance("dev-201")
        up = _capture_compose_up(DeployExecutor())[0]
        assert _project_of(up) == "omnibase-infra"
        assert len(_files_of(up)) == 2
        assert "onex-api" in up

        builds = _capture_build(DeployExecutor())
        assert {_project_of(c) for c in builds} == {"omnibase-infra"}

        migrations = _capture_migrations(DeployExecutor())
        named = {tok for cmd in migrations for tok in cmd}
        assert set(DEV_LANE_ONLY_MIGRATION_SERVICES) <= named

    @pytest.mark.gateway_lane
    def test_dev_201_unchanged_gateway_deploy_still_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            executor_mod, "deploy_gateway_script", lambda: "/nonexistent/deploy.sh"
        )
        select_dev_instance("dev-201")
        with pytest.raises(GatewayDeployScriptUnavailableError):
            DeployExecutor()._deploy_gateway_lane(
                _noop,
                lane=EnumRuntimeLane.DEV,
                build_source="release",
                targets=list(DEV_LANE_GATEWAY_SERVICES),
                git_ref=SHA,
            )

    @pytest.mark.usefixtures("_fake_applier")
    def test_dev_201_unchanged_job_runs_every_phase(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, fake, _, _ = _run_dev_job(tmp_path, monkeypatch, instance="dev-201")
        assert "seed_infisical lane=dev" in fake.calls
        assert "deliver_onex_api_pin" in fake.calls
        assert _FakeApplier.calls == ["apply"]

    def test_dev_201_unchanged_without_a_router(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An agent fenced away from dev builds no router and selects nothing."""
        monkeypatch.setenv("DEPLOY_AGENT_ALLOWED_LANES", "stability-test")
        monkeypatch.delenv("DEPLOY_AGENT_INSTANCE", raising=False)
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
        monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")

        agent = DeployAgent(skip_self_update=True)

        assert agent._router is None
        assert active_dev_instance() == "dev-201"
        assert lane_config_for(EnumRuntimeLane.DEV).compose_project == "omnibase-infra"


# --------------------------------------------------------------------------- #
# AC4 -- the unit file                                                         #
# --------------------------------------------------------------------------- #
_ENVIRONMENT_RE = re.compile(r'^Environment="?([A-Za-z_][A-Za-z0-9_]*)=')


def _directives(unit: Path) -> list[str]:
    return [
        line
        for line in unit.read_text().splitlines()
        if line and not line.lstrip().startswith("#")
    ]


def _env(unit: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in _directives(unit):
        match = _ENVIRONMENT_RE.match(line)
        if match:
            name = match.group(1)
            rest = line.split(f"{name}=", 1)[1]
            values[name] = rest.rstrip('"')
    return values


class TestDev202Unit:
    def test_dev_202_unit_names_its_instance_and_lane(self) -> None:
        env = _env(_DEV_202_UNIT)
        assert env["DEPLOY_AGENT_INSTANCE"] == "dev-202"
        assert env["DEPLOY_AGENT_ALLOWED_LANES"] == "dev"
        assert env["DEPLOY_AGENT_TRACKING_REF"] == "dev"
        # The one tie to .201: the dev control bus, SASL-declared like the
        # .201 dev unit's.
        dev = _env(_DEV_UNIT)
        for name in (
            "KAFKA_BOOTSTRAP_SERVERS",
            "KAFKA_ENVIRONMENT",
            "KAFKA_SECURITY_PROTOCOL",
            "KAFKA_SASL_MECHANISM",
            "KAFKA_SASL_ENV_PREFIX",
        ):
            assert env[name] == dev[name], name

    def test_dev_202_unit_has_its_own_port_state_and_env_file(self) -> None:
        env = _env(_DEV_202_UNIT)
        dev = _env(_DEV_UNIT)
        port = env["DEPLOY_AGENT_PORT"]
        assert port not in {"8098", "8099"}
        assert env["DEPLOY_AGENT_STATE_DIR"] != dev["DEPLOY_AGENT_STATE_DIR"]
        assert env["DEPLOY_AGENT_STATE_DIR"].endswith("jobs-dev-202")
        assert env["DEPLOY_AGENT_ENV_FILE"] != dev["DEPLOY_AGENT_ENV_FILE"]
        assert env["DEPLOY_AGENT_ENV_FILE"].endswith("dev-202.env")

        pre = [d for d in _directives(_DEV_202_UNIT) if d.startswith("ExecStartPre=")]
        assert pre == [
            "ExecStartPre=/data/omninode/omnibase_infra/scripts/deploy-agent/deploy/"
            f"preflight_port_free.sh {port}"
        ]
        start = [d for d in _directives(_DEV_202_UNIT) if d.startswith("ExecStart=")]
        assert start == [
            "ExecStart=/data/omninode/omnibase_infra/scripts/deploy-agent/deploy/"
            "deploy-agent-launch.sh"
        ]
        assert "SyslogIdentifier=deploy-agent-dev-202" in _directives(_DEV_202_UNIT)

    def test_dev_202_unit_protects_every_name_and_reads_no_env_file(self) -> None:
        env = _env(_DEV_202_UNIT)
        protected = set(env["DEPLOY_AGENT_ENV_PROTECTED"].split())
        assert sorted(set(env) - protected) == []
        assert not [
            d for d in _directives(_DEV_202_UNIT) if d.startswith("EnvironmentFile=")
        ]
        assert not [d for d in _directives(_DEV_202_UNIT) if "WatchdogSec" in d]

    def test_dev_202_unit_port_is_in_the_lane_block(self) -> None:
        """61000-61999 is dev-202's port block (docker-compose.dev-202.yml)."""
        port = int(_env(_DEV_202_UNIT)["DEPLOY_AGENT_PORT"])
        assert 61000 <= port <= 61999
        overlay_text = _DEV_202_OVERLAY.read_text()
        assert f":{port}:" not in overlay_text


# --------------------------------------------------------------------------- #
# AC5 -- the names-only env template                                           #
# --------------------------------------------------------------------------- #
_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:?[-?+])?")
_ASSIGN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def _required_by_the_triple() -> set[str]:
    """Every ``${NAME:?...}`` in the compose triple, minus runtime-policy.env.

    Compose interpolates the whole model, so a required name fails the command
    even when its service is profile-disabled.
    """
    required: set[str] = set()
    for name in (
        "docker-compose.infra.yml",
        "docker-compose.dev-lane.yml",
        "docker-compose.dev-202.yml",
    ):
        for line in (_DOCKER / name).read_text().splitlines():
            if line.lstrip().startswith("#"):
                continue
            code = line.split(" #", 1)[0]
            for match in _REF_RE.finditer(code):
                if match.group(2) in (":?", "?"):
                    required.add(match.group(1))
    policy = {
        m.group(1)
        for line in (_DOCKER / "runtime-policy.env").read_text().splitlines()
        if (m := _ASSIGN_RE.match(line.strip()))
    }
    return required - policy


def _template_names() -> dict[str, str]:
    names: dict[str, str] = {}
    for raw in _ENV_TEMPLATE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _ASSIGN_RE.match(line)
        assert match, f"not a NAME= line: {raw!r}"
        names[match.group(1)] = match.group(2)
    return names


class TestDev202EnvTemplate:
    def test_dev_202_env_template_carries_no_value(self) -> None:
        names = _template_names()
        assert names
        assert {k: v for k, v in names.items() if v} == {}

    def test_dev_202_env_template_covers_every_required_name(self) -> None:
        required = _required_by_the_triple()
        assert "POSTGRES_PASSWORD" in required  # positive control on the scan
        assert sorted(required - set(_template_names())) == []

    def test_dev_202_env_template_names_the_control_bus_principal(self) -> None:
        names = _template_names()
        prefix = _env(_DEV_202_UNIT)["KAFKA_SASL_ENV_PREFIX"]
        for suffix in ("KAFKA_SASL_USERNAME", "KAFKA_SASL_PASSWORD"):
            assert f"{prefix}{suffix}" in names
        assert "DEPLOY_AGENT_HMAC_SECRET" in names
