# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent brings the lane-mirror forwarders up itself (OMN-20867).

The forwarders that mirror a source lane's topics onto a dev lane join that
lane's network, which only the deploy agent may do. After the runtime phase is
verified the agent reads the private overlay's ``deploy-overlays/lane-mirror``
from its overlay clone at ``origin/dev``, picks the host vars whose lane network
the lane's runtime container is attached to, brings that vars file's profiles up
on the runtime container's image, and reports each forwarder's ready state in
the terminal event. No overlay is a typed no-op; an enabled profile that does
not come up is ``FAILED`` in the receipt, never silent, and never changes the
deploy's own verdict.

The second half: an instance whose receipt proves omnimarket only follows the
omnibase_infra dev head when idle, because the routing table sends
omnibase_infra rebuilds to the default instance.

Every lane, host and network name here is synthetic.
"""

from __future__ import annotations

import inspect
import json
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
import yaml
from deploy_agent import agent as agent_mod
from deploy_agent import executor as executor_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.events import (
    EnumLaneMirrorOutcome,
    EnumRuntimeLane,
    ModelLaneMirrorReceipt,
    Phase,
    PhaseStatus,
)
from deploy_agent.idle_converge import (
    IDLE_CONVERGE_REQUESTER,
    EnumIdleConvergeVerdict,
    EnumRuntimePresence,
    converge_command,
    read_running_ref,
)
from deploy_agent.instance_lanes import (
    LANE_ENV_FRAGMENT_REF,
    LANE_MIRROR_COMPOSE_FILE,
    LANE_MIRROR_CREDENTIAL_MAP_KEY,
    LANE_MIRROR_DIR,
    LANE_MIRROR_IMAGE_KEY,
    ModelLaneMirrorRequest,
    handle_lane_mirror,
)
from deploy_agent.job_state import JobState, JobStore
from deploy_agent.publisher import build_completion_payload
from deploy_agent.routing import (
    PINNED_DEFAULT_INSTANCE,
    DeployRouter,
    ModelLaneFlags,
    load_routing_table,
    parse_routing_table,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNTIME = "lane-a-runtime"
NETWORK = "lane-a-network"
IMAGE = "sha256:" + "1" * 64
OTHER_IMAGE = "sha256:" + "2" * 64

COMPOSE = {
    "services": {
        "mirror-one": {
            "image": "${LANE_MIRROR_IMAGE}",
            "profiles": ["one"],
            "container_name": "lane-mirror-one",
        },
        "mirror-two": {
            "image": "${LANE_MIRROR_IMAGE}",
            "profiles": ["two"],
            "container_name": "lane-mirror-two",
        },
    }
}
VARS_A = f"""# lane a
COMPOSE_PROJECT_NAME=lane-mirror
COMPOSE_PROFILES=one
LANE_MIRROR_LANE_NETWORK={NETWORK}
LANE_MIRROR_SOURCE_HOST_ADDRESS=host-gateway
"""
VARS_B = """COMPOSE_PROJECT_NAME=lane-mirror
COMPOSE_PROFILES=two
LANE_MIRROR_LANE_NETWORK=lane-b-network
"""


@pytest.fixture(autouse=True)
def _isolate_overlay_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep temporary-clone reads and writes independent of harness git paths."""
    real_run = subprocess.run

    def run(argv: list[str], **kwargs: Any) -> Any:
        if argv[:2] == ["git", "-C"]:
            repo = Path(argv[2])
            if repo.is_relative_to(tmp_path):
                argv = [
                    "git",
                    "--git-dir",
                    str(repo / ".git"),
                    "--work-tree",
                    str(repo),
                    *argv[1:],
                ]
        return real_run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    )


def _overlay_clone(tmp_path: Path, files: Mapping[str, str]) -> Path:
    """A git repo whose ``origin/dev`` holds ``files``."""
    repo = tmp_path / "overlay-clone"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "test")
    (repo / "README").write_text("overlay\n", encoding="utf-8")
    for relpath, text in files.items():
        target = repo / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "overlay")
    _git(repo, "update-ref", f"refs/remotes/{LANE_ENV_FRAGMENT_REF}", "HEAD")
    return repo


def _mirror_overlay(tmp_path: Path, vars_a: str = VARS_A) -> Path:
    return _overlay_clone(
        tmp_path,
        {
            f"{LANE_MIRROR_DIR}/{LANE_MIRROR_COMPOSE_FILE}": yaml.safe_dump(COMPOSE),
            f"{LANE_MIRROR_DIR}/host-a.vars": vars_a,
            f"{LANE_MIRROR_DIR}/host-b.vars": VARS_B,
            f"{LANE_MIRROR_DIR}/one.forwarder.yaml": "forwarder: {}\n",
        },
    )


def _proc(stdout: str = "", returncode: int = 0, stderr: str = "") -> Any:
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


class _Docker:
    """Answers the docker reads and writes; git passes through to the real clone."""

    def __init__(
        self,
        *,
        forwarder: str = f"running|healthy|{IMAGE}",
        up_returncode: int = 0,
        networks: tuple[str, ...] = (NETWORK,),
    ) -> None:
        self.forwarder = forwarder
        self.up_returncode = up_returncode
        self.networks = networks
        self.calls: list[list[str]] = []
        self.up_env: dict[str, str] | None = None

    def __call__(
        self, argv: list[str], timeout: float, env: Mapping[str, str] | None = None
    ) -> Any:
        if argv[0] == "git":
            return subprocess.run(argv, capture_output=True, text=True, check=False)
        self.calls.append(argv)
        if argv[:2] == ["docker", "compose"]:
            self.up_env = dict(env or {})
            return _proc(returncode=self.up_returncode, stderr="up refused")
        if argv[:2] == ["docker", "inspect"]:
            fmt, target = argv[3], argv[4]
            if target == RUNTIME and "Networks" in fmt:
                return _proc(json.dumps({n: {} for n in self.networks}))
            if target == RUNTIME:
                return _proc(IMAGE + "\n")
            return _proc(self.forwarder + "\n")
        raise AssertionError(f"unexpected command {argv}")


def _request(tmp_path: Path, clone: Path, **env: str) -> ModelLaneMirrorRequest:
    credential_map = tmp_path / "lane-credentials.yaml"
    credential_map.write_text("{}\n", encoding="utf-8")
    compose_env = {LANE_MIRROR_CREDENTIAL_MAP_KEY: str(credential_map), **env}
    return ModelLaneMirrorRequest(
        clone=clone,
        runtime_container=RUNTIME,
        materialize_root=tmp_path / "materialized",
        compose_env=compose_env,
        ready_timeout_seconds=10,
        poll_seconds=1,
    )


def _handle(request: ModelLaneMirrorRequest, docker: _Docker) -> ModelLaneMirrorReceipt:
    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    return handle_lane_mirror(request, run=docker, sleep=sleep, clock=lambda: now[0])


# -- absent overlay: a typed no-op ----------------------------------------------


def test_no_overlay_clone_is_a_typed_no_op(tmp_path: Path) -> None:
    docker = _Docker()
    receipt = _handle(_request(tmp_path, tmp_path / "absent"), docker)
    assert receipt.outcome is EnumLaneMirrorOutcome.NO_OVERLAY
    assert receipt.is_failure is False
    assert docker.calls == []


def test_clone_without_the_lane_mirror_directory_is_a_typed_no_op(
    tmp_path: Path,
) -> None:
    clone = _overlay_clone(tmp_path, {"deploy-overlays/other/x.json": "{}"})
    docker = _Docker()
    receipt = _handle(_request(tmp_path, clone), docker)
    assert receipt.outcome is EnumLaneMirrorOutcome.NO_OVERLAY
    assert receipt.is_failure is False
    assert receipt.overlay_sha
    assert docker.calls == [], "nothing is started without an overlay"


def test_no_vars_file_naming_this_lanes_network_is_a_typed_no_op(
    tmp_path: Path,
) -> None:
    docker = _Docker(networks=("some-other-network",))
    receipt = _handle(_request(tmp_path, _mirror_overlay(tmp_path)), docker)
    assert receipt.outcome is EnumLaneMirrorOutcome.NOT_THIS_LANE
    assert receipt.is_failure is False
    assert not any(c[:2] == ["docker", "compose"] for c in docker.calls)


def test_vars_enabling_no_profile_is_a_typed_no_op(tmp_path: Path) -> None:
    clone = _mirror_overlay(
        tmp_path, VARS_A.replace("COMPOSE_PROFILES=one", "COMPOSE_PROFILES=")
    )
    docker = _Docker()
    receipt = _handle(_request(tmp_path, clone), docker)
    assert receipt.outcome is EnumLaneMirrorOutcome.NO_PROFILES
    assert receipt.vars_file == "host-a.vars"
    assert not any(c[:2] == ["docker", "compose"] for c in docker.calls)


# -- an enabled profile is started on the lane's runtime image --------------------


def test_enabled_profile_is_started_with_the_lane_runtime_image(
    tmp_path: Path,
) -> None:
    docker = _Docker()
    receipt = _handle(_request(tmp_path, _mirror_overlay(tmp_path)), docker)

    assert receipt.outcome is EnumLaneMirrorOutcome.READY, receipt.detail
    assert receipt.is_failure is False
    assert receipt.vars_file == "host-a.vars"
    assert receipt.lane_network == NETWORK
    assert receipt.profiles == ("one",)
    assert receipt.runtime_image == IMAGE
    [up] = [c for c in docker.calls if c[:2] == ["docker", "compose"]]
    assert up[up.index("-p") + 1] == "lane-mirror"
    assert up[up.index("--profile") + 1] == "one"
    assert up[-3:] == ["up", "-d", "mirror-one"], "only the enabled profile's service"
    assert docker.up_env is not None
    assert docker.up_env[LANE_MIRROR_IMAGE_KEY] == IMAGE
    assert docker.up_env["LANE_MIRROR_LANE_NETWORK"] == NETWORK
    compose_path = Path(up[up.index("-f") + 1])
    assert compose_path.name == LANE_MIRROR_COMPOSE_FILE
    assert (compose_path.parent / "one.forwarder.yaml").is_file(), (
        "the overlay is materialised so the compose file's relative mounts resolve"
    )
    [forwarder] = receipt.forwarders
    assert (forwarder.container, forwarder.state, forwarder.health) == (
        "lane-mirror-one",
        "running",
        "healthy",
    )
    assert forwarder.ready is True
    assert forwarder.image_id == IMAGE


# -- an enabled profile that fails to come up is loud in the receipt -------------


@pytest.mark.parametrize(
    ("forwarder", "named"),
    [
        (f"running|unhealthy|{IMAGE}", "health=unhealthy"),
        (f"exited|unhealthy|{IMAGE}", "state=exited"),
        (f"running|healthy|{OTHER_IMAGE}", "image=not-the-runtime-image"),
    ],
)
def test_failed_start_is_loud_in_the_receipt(
    tmp_path: Path, forwarder: str, named: str
) -> None:
    receipt = _handle(
        _request(tmp_path, _mirror_overlay(tmp_path)), _Docker(forwarder=forwarder)
    )
    assert receipt.outcome is EnumLaneMirrorOutcome.FAILED
    assert receipt.is_failure is True
    assert "lane-mirror-one" in receipt.detail
    assert named in receipt.detail
    assert [f.ready for f in receipt.forwarders] == [False]


def test_compose_up_refused_is_loud_in_the_receipt(tmp_path: Path) -> None:
    receipt = _handle(
        _request(tmp_path, _mirror_overlay(tmp_path)),
        _Docker(up_returncode=1, forwarder="absent"),
    )
    assert receipt.is_failure is True
    assert "docker compose up exited 1" in receipt.detail
    assert "up refused" in receipt.detail


def test_enabled_profile_without_a_credential_map_is_loud(tmp_path: Path) -> None:
    request = _request(tmp_path, _mirror_overlay(tmp_path))
    request = request.model_copy(update={"compose_env": {}})
    docker = _Docker()
    receipt = _handle(request, docker)
    assert receipt.is_failure is True
    assert LANE_MIRROR_CREDENTIAL_MAP_KEY in receipt.detail
    assert not any(c[:2] == ["docker", "compose"] for c in docker.calls)


def test_a_failed_mirror_never_changes_the_deploy_verdict(tmp_path: Path) -> None:
    receipt = _handle(
        _request(tmp_path, _mirror_overlay(tmp_path)),
        _Docker(forwarder=f"running|unhealthy|{IMAGE}"),
    )
    now = datetime.now(UTC)
    job = JobState(
        correlation_id=uuid4(),
        command={"runtime_lane": "dev", "scope": "runtime", "git_ref": "a" * 40},
        status="success",
        accepted_at=now - timedelta(minutes=5),
        completed_at=now,
        phase_results={
            Phase.RUNTIME: PhaseStatus.SUCCESS,
            Phase.VERIFICATION: PhaseStatus.SUCCESS,
        },
    )
    payload = build_completion_payload(
        job, "a" * 40, services_restarted=["omninode-runtime"], lane_mirror=receipt
    )
    assert payload["status"] == "success"
    assert payload["lane_mirror"]["outcome"] == "failed"
    assert payload["lane_mirror"]["is_failure"] is True
    assert payload["lane_mirror"]["forwarders"][0]["container"] == "lane-mirror-one"
    assert ModelLaneMirrorReceipt.model_validate(payload["lane_mirror"]) == receipt


# -- the executor's fence and the agent's wiring ----------------------------------


def test_executor_runs_the_step_only_on_a_dev_fenced_agent_with_a_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = executor_mod.DeployExecutor()
    monkeypatch.setattr(executor_mod._LaneOverlayClone, "path", None)
    monkeypatch.setenv("DEPLOY_AGENT_ALLOWED_LANES", "dev")
    assert (
        executor.run_lane_mirror(lane=EnumRuntimeLane.DEV, materialize_root=tmp_path)
        is None
    )

    monkeypatch.setattr(executor_mod._LaneOverlayClone, "path", tmp_path / "absent")
    assert (
        executor.run_lane_mirror(
            lane=EnumRuntimeLane.STABILITY_TEST, materialize_root=tmp_path
        )
        is None
    )
    monkeypatch.setenv(
        "DEPLOY_AGENT_ALLOWED_LANES",
        f"{EnumRuntimeLane.DEV.value},{EnumRuntimeLane.STABILITY_TEST.value}",
    )
    assert (
        executor.run_lane_mirror(lane=EnumRuntimeLane.DEV, materialize_root=tmp_path)
        is None
    )

    monkeypatch.setenv("DEPLOY_AGENT_ALLOWED_LANES", "dev")
    monkeypatch.setattr(executor_mod, "_compose_env", lambda extra_env=None: {})
    receipt = executor.run_lane_mirror(
        lane=EnumRuntimeLane.DEV, materialize_root=tmp_path
    )
    assert receipt is not None
    assert receipt.outcome is EnumLaneMirrorOutcome.NO_OVERLAY


def test_the_agent_runs_the_step_after_verification_and_before_the_verdict() -> None:
    source = inspect.getsource(DeployAgent._run_deploy)
    verify = source.index("self.executor.verify(")
    mirror = source.index("self.executor.run_lane_mirror(")
    complete = source.index('status="success"')
    assert verify < mirror < complete


def test_the_receipt_rides_the_terminal_facts() -> None:
    agent = DeployAgent.__new__(DeployAgent)
    agent._current_git_sha = "a" * 40

    class _Executor:
        health_checks: list[object] = []
        container_residue: list[object] = []
        recreate_supervision: list[object] = []
        deps_convergence: list[object] = []
        compose_invocations: list[object] = []
        verify_recreate: list[object] = []
        sibling_source_refs: dict[str, str] = {}

    agent.executor = cast("Any", _Executor())
    receipt = ModelLaneMirrorReceipt(
        outcome=EnumLaneMirrorOutcome.NO_OVERLAY, detail="none"
    )
    facts = agent._terminal_facts([], ["omninode-runtime"], receipt)
    assert facts.lane_mirror is receipt


# -- an omnimarket-only instance follows the omnibase_infra dev head --------------

#: The committed table's shape: omnimarket rebuilds go to the synthetic
#: ``inst-b``; every other requester, omnibase_infra included, to the default,
#: whose name the router pins.
DEFAULT = PINNED_DEFAULT_INSTANCE
ROUTING_TEXT = f"""\
default_instance: {DEFAULT}
instances:
  {DEFAULT}:
    hostnames: [host-a]
    consumer_group: group-a
  inst-b:
    hostnames: [host-b]
    consumer_group: group-b
routes:
  - runtime_lane: dev
    requester_repository: omnimarket
    instance: inst-b
"""

INFRA_HEAD = "b" * 40
INFRA_RUNNING = "a" * 40
QUIET = datetime(2026, 9, 25, 4, 53, tzinfo=UTC)


def test_running_omnibase_infra_ref_is_the_images_infra_vcs_ref() -> None:
    manifest = json.dumps(
        {
            "infra_vcs_ref": INFRA_RUNNING,
            "per_repo_vcs_provenance": {
                "siblings": {"omnimarket": {"vcs_ref": "c" * 40}}
            },
        }
    )

    def run(argv: list[str], timeout: int) -> Any:
        return _proc(manifest)

    assert read_running_ref("runtime", "omnibase_infra", run=run) == INFRA_RUNNING
    assert read_running_ref("runtime", "omnimarket", run=run) == "c" * 40


class _FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.container_residue: list[object] = []
        self.sibling_source_refs: dict[str, str] = {}
        self.recreate_supervision: list[object] = []
        self.verify_recreate: list[object] = []
        self.deps_convergence: list[object] = []
        self.compose_invocations: list[object] = []
        self.health_checks: list[object] = []

    def __getattr__(self, name: str) -> Any:
        def record(*args: object, **kwargs: object) -> Any:
            self.calls.append(name)
            if name == "git_pull":
                return "d" * 40
            if name in ("rebuild_scope", "verify"):
                return []
            return None

        return record


def _agent_as(
    instance: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> DeployAgent:
    """An agent that is ``instance`` under :data:`ROUTING_TEXT`."""
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    monkeypatch.setattr(agent_mod, "publish_result", lambda payload, config: False)
    monkeypatch.setattr(agent_mod, "LAB_OVERLAY_ENABLED", False)
    agent = DeployAgent(skip_self_update=True)
    table = parse_routing_table(ROUTING_TEXT)
    agent._router = DeployRouter(table, table.instances[instance], lambda sha: table)
    agent.job_store = JobStore(tmp_path / "jobs")
    agent.executor = cast("Any", _FakeExecutor())
    agent._idle_converge_started_at = QUIET - timedelta(hours=1)
    agent._idle_converge_now = lambda: QUIET
    # The lane runs the omnimarket dev head and an older omnibase_infra.
    heads = {"omnimarket": "e" * 40, "omnibase_infra": INFRA_HEAD}
    running = {"omnimarket": "e" * 40, "omnibase_infra": INFRA_RUNNING}
    agent._idle_read_running_ref = lambda repo: running[repo]
    # OMN-20864: the lane's runtime is running, so the restore rule stays out.
    agent._idle_read_runtime_presence = lambda: (
        EnumRuntimePresence.RUNNING,
        "omninode-runtime is running",
    )
    agent._idle_read_head_ref = lambda repo: heads[repo]
    agent._idle_probe_blocker = lambda now: None
    agent._lane_flags = ModelLaneFlags()
    return agent


def _jobs(tmp_path: Path) -> list[JobState]:
    store = JobStore(tmp_path / "jobs")
    return [
        job
        for p in (tmp_path / "jobs").glob("*.json")
        if (job := store.load(UUID(p.stem))) is not None
    ]


def test_omnimarket_only_instance_converges_on_the_omnibase_infra_dev_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent_as("inst-b", tmp_path, monkeypatch)

    agent._maybe_idle_converge()

    [job] = _jobs(tmp_path)
    assert job.command["requested_by"] == IDLE_CONVERGE_REQUESTER
    assert job.command["runtime_lane"] == "dev"
    assert job.command["git_ref"] == converge_command().git_ref, "the tracking ref"
    assert INFRA_HEAD in agent._idle_converge_attempted

    agent._idle_converge_last_check = None
    agent._maybe_idle_converge()
    assert len(_jobs(tmp_path)) == 1, "the same omnibase_infra head converges once"


def test_default_instance_does_not_follow_omnibase_infra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """omnibase_infra rebuilds are routed to the default itself, so it never follows."""
    agent = _agent_as(DEFAULT, tmp_path, monkeypatch)

    agent._maybe_idle_converge()

    assert _jobs(tmp_path) == []
    assert (
        agent._idle_converge_last_verdicts["omnibase_infra"]
        is EnumIdleConvergeVerdict.NOT_ROUTED_ELSEWHERE
    )
    assert (
        agent._idle_converge_last_verdicts["omnimarket"]
        is EnumIdleConvergeVerdict.UP_TO_DATE
    )


def test_committed_routing_leaves_every_lab_pass_receipt_as_it_was() -> None:
    """omnibase_infra rebuilds still go to the default instance, and no other
    instance's receipt proves omnibase_infra, so a converge on any other
    instance adds no omnibase_infra lab-pass evidence anywhere."""
    table = load_routing_table(REPO_ROOT)
    assert (
        table.route(EnumRuntimeLane.DEV, "gha/omnibase_infra/pr-1")
        == table.default_instance
    )
    raw = yaml.safe_load(
        (REPO_ROOT / "config" / "deploy_lane_routing.yaml").read_text(encoding="utf-8")
    )
    others = [
        spec["lane"]
        for name, spec in raw["instances"].items()
        if name != table.default_instance
    ]
    assert others, "the committed table declares instances beside the default"
    assert all("omnibase_infra" not in lane["proves"] for lane in others)
