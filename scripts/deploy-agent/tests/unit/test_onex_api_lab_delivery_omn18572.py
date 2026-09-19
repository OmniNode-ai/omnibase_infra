# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18572 -- the agent's half of the onex-api lab delivery.

THREE DEFECTS, PINNED SEPARATELY BECAUSE THEY FAIL SEPARATELY
-------------------------------------------------------------
Measured on the ``.201`` compose dev lane on 2026-09-17, in one window.

**One, the lane lock covers the git phase and nothing else.** ``git_pull`` takes
the per-compose-project lock (OMN-18124) and releases it on the way out, so the
build, the ``compose up`` and the verify all run with the lane unlocked. At
11:20Z a hand ``deploy-runtime.sh`` acquired that lock uncontended while the
agent was mid-runtime-phase on job ``746a118a``; the lane dropped to ``000`` on
all three published ports for about six minutes. The agent's own
``single_flight_lock`` serializes agent against agent and says nothing about
anybody else, which is why nothing refused.

**Two, the agent cannot see a repointed pin.** ``deploy-agent-dev.service``
resolves the operator env store once, at unit start, so ``_compose_env``'s
``dict(os.environ)`` re-copies a snapshot frozen at that moment. A repoint
written by ``scripts/runtime_build/repoint_dev_lane_onex_api.py`` is therefore
inert for the automatic path until the agent restarts. Observed at 11:26Z: the
agent recreated ``onex-api`` on the OLD pin after an 11:18:53Z env-file write.

**Three, nothing on the agent path repoints at all.** ``lab_overlay._derive_pins``
builds ``onex-lab/omnicloud-core:<omninode_infra sha8>-<stamp>`` on every apply,
so a correct image EXISTS on the host after every deploy. ``ONEX_API_IMAGE`` is
what DELIVERS it, and no code in this repository writes that key on the agent
path. "Exists on the host" is not "delivered", and that gap is what left a fix
merged at 09:00:53Z off the lane at 11:15Z.

WHAT THESE TESTS DO NOT DO
--------------------------
None of them reads a line number or greps a source file for a call. The lock
test proves the lock is HELD by trying to take it from a second open file
description -- ``flock`` is keyed to the description, not the process, so a
second ``open()`` in this same process contends exactly as a peer lane would.
That is a behavioural proof; an assertion that a context manager was entered
would pass against a manager that exits immediately.

Every zero carries a positive control. The contention test asserts the lock is
free before the deploy and free again after it, so "contended during
rebuild_scope" cannot be satisfied by a lock file that is simply always stuck.
"""

from __future__ import annotations

import fcntl
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.events import EnumRuntimeLane, ModelRebuildRequested, Scope
from deploy_agent.executor import _compose_env, lane_config_for
from deploy_agent.job_state import JobStore
from deploy_agent.lane_lock_client import lane_lock_path

pytestmark = pytest.mark.unit

SHA = "a" * 40
#: The omninode_infra overlay commit the apply resolves -- the lineage the
#: lab image tags carry, and the one the pin delivery must ask for. See
#: ``test_onex_api_pin_lineage_omn18572`` for the defect this distinction
#: exists to make visible.
OVERLAY_SHA = "b" * 40
STALE_PIN = "onex-lab/omnicloud-core:f37261c2-20260917T050425Z"
FRESH_PIN = "onex-lab/omnicloud-core:99fdbd37-20260917T110254Z"


def _lock_is_free(path: Path) -> bool:
    """True when a SECOND open file description can take the lock.

    ``flock`` is associated with the open file description rather than the
    process, so opening the same path again here contends with a holder in this
    same process exactly as a peer lane's shell would. That property is what
    makes this a behavioural probe instead of a bookkeeping one.
    """
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return True
    finally:
        handle.close()


class _LockProbingExecutor:
    """A fake executor that records whether the lane lock was held per phase.

    Mirrors the ``_FakeExecutor`` shape the neighbouring agent-level tests use
    (``test_lab_overlay_build_order_omn18545``), plus the two attributes the
    agent reads when it builds its terminal event.
    """

    def __init__(self, lock_path: Path) -> None:
        self._lock_path = lock_path
        self.calls: list[str] = []
        #: phase name -> was the lane lock held while that phase ran
        self.lock_held_during: dict[str, bool] = {}
        self.container_residue: list[object] = []
        self.sibling_source_refs: dict[str, str] = {}
        # OMN-18692: the agent reads this when it builds the terminal event,
        # so a double that omits it no longer models the object it replaces.
        self.recreate_supervision: list[object] = []
        # OMN-18640: the terminal event now also carries what verification
        # recreated, so a fake executor has to declare it.
        self.verify_recreate: list[object] = []
        # OMN-18640: the terminal event now also carries what the deps leg
        # found before it acted, and the argv of every compose call, so a
        # fake executor has to declare both.
        self.deps_convergence: list[object] = []
        self.compose_invocations: list[object] = []
        self.delivered: list[dict[str, Any]] = []

    def _record(self, phase: str) -> None:
        self.calls.append(phase)
        self.lock_held_during[phase] = not _lock_is_free(self._lock_path)

    def resolve_stability_ready_digest(self, service: str = "") -> str | None:
        self._record("resolve_stability_ready_digest")
        return None

    def preflight(self, **kwargs: object) -> None:
        self._record("preflight")

    def git_pull(self, git_ref: str, **kwargs: object) -> str:
        self._record("git_pull")
        return SHA

    def compose_gen(self, bundles: list[str], **kwargs: object) -> None:
        self._record("compose_gen")

    def seed_infisical(self, **kwargs: object) -> None:
        self._record("seed_infisical")

    def validate_llm_endpoint_env_contract(self) -> None:
        self._record("validate_llm_endpoint_env_contract")

    def rebuild_scope(self, *args: object, **kwargs: object) -> list[str]:
        self._record("rebuild_scope")
        return ["omninode-runtime"]

    def verify(self, **kwargs: object) -> list[object]:
        self._record("verify")
        return []

    def deploy_and_verify(self, **kwargs: object) -> list[object]:
        self._record("deploy_and_verify")
        return []

    def deliver_onex_api_pin(self, **kwargs: Any) -> dict[str, Any]:
        self._record("deliver_onex_api_pin")
        self.delivered.append(kwargs)
        return {
            "result": "WRITTEN",
            "pin_before": STALE_PIN,
            "pin_after": FRESH_PIN,
            "tag_advanced": True,
            "recreated": True,
        }


class _FakeApplier:
    """Stands in for ``LabOverlayApplier``; records the entry point taken."""

    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.manifest_sha: str | None = None

    def apply(self, *, sha: str, stamp: str, correlation_id: str) -> Path:
        _FakeApplier.calls.append({"entry": "apply", "sha": sha})
        # OMN-18572: the overlay's OWN lineage, which is what the four image
        # tags carry. Deliberately different from `sha` (the merged
        # omnibase_infra commit) so a test cannot pass by conflating them.
        self.manifest_sha = OVERLAY_SHA
        return Path(f"/state/lab-overlay/{sha}.json")

    def build_repair_migrate_image(
        self, *, sha: str, stamp: str, correlation_id: str
    ) -> Path:
        _FakeApplier.calls.append({"entry": "repair", "sha": sha})
        return Path(f"/state/lab-overlay/{sha}.json")

    @classmethod
    def reset(cls) -> None:
        cls.calls = []


@pytest.fixture(autouse=True)
def _isolated_lane_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the shared lane-lock helper at a private directory for this test.

    ``ONEX_LANE_LOCK_DIR`` is the override ``scripts/runtime_build/lane_lock.py``
    declares for exactly this purpose. Without it a unit test would contend with
    the real lock on the developer's own machine, which is both wrong and slow.
    """
    lock_dir = tmp_path / "lane-locks"
    lock_dir.mkdir()
    monkeypatch.setenv("ONEX_LANE_LOCK_DIR", str(lock_dir))
    # The re-entrancy token must not leak in from an outer process, or the lock
    # this suite asserts on would be skipped as already held.
    monkeypatch.delenv("ONEX_LANE_LOCK_HELD", raising=False)
    _FakeApplier.reset()
    monkeypatch.setattr(agent_mod, "LabOverlayApplier", _FakeApplier)
    monkeypatch.setattr(agent_mod, "LAB_OVERLAY_ENABLED", True)
    return lock_dir


def _dev_cmd() -> ModelRebuildRequested:
    return ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.DEV,
    )


def _make_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cmd: ModelRebuildRequested,
    executor: _LockProbingExecutor,
) -> DeployAgent:
    store = JobStore(tmp_path / "jobs")
    store.accept(cmd.correlation_id, cmd.model_dump(mode="json"))
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    agent = DeployAgent(skip_self_update=True)
    agent.job_store = store
    agent.executor = executor  # type: ignore[assignment]
    monkeypatch.setattr(agent_mod, "publish_result", lambda payload, config: False)
    return agent


# --------------------------------------------------------------------------- #
# AC2 -- the lane lock spans the whole deploy, not only its git phase          #
# --------------------------------------------------------------------------- #


async def test_lane_lock_is_held_through_every_mutating_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The phases that touch the lane must run under the lane's own lock.

    RED against the tree where ``git_pull`` is the only holder: every phase
    after it reports the lock free, which is precisely the window the 11:20Z
    outage landed in.
    """
    cmd = _dev_cmd()
    lock_path = Path(
        lane_lock_path(lane_config_for(EnumRuntimeLane.DEV).compose_project)
    )
    executor = _LockProbingExecutor(lock_path)
    agent = _make_agent(tmp_path, monkeypatch, cmd, executor)

    # Positive control, before: the lock is genuinely free, so a later "held"
    # reading cannot be an artifact of a permanently stuck lock file.
    assert _lock_is_free(lock_path) is True

    agent._run_deploy(cmd)

    mutating_phases = ("git_pull", "compose_gen", "rebuild_scope", "verify")
    unlocked = [
        phase
        for phase in mutating_phases
        if not executor.lock_held_during.get(phase, False)
    ]
    assert unlocked == [], (
        "these phases mutate the lane with its lock released, so a concurrent "
        f"deploy-runtime.sh acquires it uncontended: {unlocked}"
    )

    # Positive control, after: the lock is released on the way out. A test that
    # only asserted "held" would also pass against a lock that is never freed,
    # which would wedge every subsequent deploy on the host.
    assert _lock_is_free(lock_path) is True


async def test_lane_lock_is_released_when_the_deploy_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing job must not leave the lane locked against the next one."""
    cmd = _dev_cmd()
    lock_path = Path(
        lane_lock_path(lane_config_for(EnumRuntimeLane.DEV).compose_project)
    )
    executor = _LockProbingExecutor(lock_path)

    def _boom(*args: object, **kwargs: object) -> list[str]:
        executor.calls.append("rebuild_scope")
        raise RuntimeError("synthetic rebuild failure")

    executor.rebuild_scope = _boom  # type: ignore[method-assign]
    agent = _make_agent(tmp_path, monkeypatch, cmd, executor)

    agent._run_deploy(cmd)

    assert _lock_is_free(lock_path) is True


# --------------------------------------------------------------------------- #
# AC3 -- the pin is read from the operator env file at job time                #
# --------------------------------------------------------------------------- #


def test_compose_env_reads_the_onex_api_pin_from_the_operator_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repoint written after the agent started must reach the next job.

    RED against the tree that trusts ``os.environ``: the value returned is the
    one systemd resolved at unit start, which is the 11:26Z recreate verbatim.
    """
    env_file = tmp_path / "operator.env"
    env_file.write_text(f"ONEX_API_IMAGE={FRESH_PIN}\n", encoding="utf-8")
    monkeypatch.setenv("DEPLOY_AGENT_ENV_FILE", str(env_file))
    # The stale snapshot the running process is carrying.
    monkeypatch.setenv("ONEX_API_IMAGE", STALE_PIN)

    assert _compose_env()["ONEX_API_IMAGE"] == FRESH_PIN


def test_compose_env_leaves_other_keys_to_the_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: the re-read is scoped to the repointed pin keys.

    Without this, a fix that simply overlaid the whole operator file would pass
    the test above while silently changing every other value the deploy runs
    with -- a far larger blast radius than the defect.
    """
    env_file = tmp_path / "operator.env"
    env_file.write_text(
        f"ONEX_API_IMAGE={FRESH_PIN}\nPOSTGRES_HOST=from-the-file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEPLOY_AGENT_ENV_FILE", str(env_file))
    monkeypatch.setenv("POSTGRES_HOST", "from-the-process")

    env = _compose_env()
    assert env["ONEX_API_IMAGE"] == FRESH_PIN
    assert env["POSTGRES_HOST"] == "from-the-process"


def test_compose_env_survives_an_absent_operator_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable store must degrade to the process value, never raise.

    The agent's ability to deploy at all cannot depend on a file that exists
    only on the lab host: refusing here would convert a delivery improvement
    into an outage on every other machine.
    """
    monkeypatch.setenv("DEPLOY_AGENT_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("ONEX_API_IMAGE", STALE_PIN)

    assert _compose_env()["ONEX_API_IMAGE"] == STALE_PIN


def test_compose_env_needs_no_operator_env_file_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no declaration at all the behaviour is exactly today's."""
    monkeypatch.delenv("DEPLOY_AGENT_ENV_FILE", raising=False)
    monkeypatch.setenv("ONEX_API_IMAGE", STALE_PIN)

    assert _compose_env()["ONEX_API_IMAGE"] == STALE_PIN


# --------------------------------------------------------------------------- #
# AC4 -- a successful dev deploy DELIVERS the pin the applier just built       #
# --------------------------------------------------------------------------- #


async def test_successful_dev_deploy_delivers_the_onex_api_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The applier builds; this asserts something then delivers.

    RED against the tree where the applier is the last statement of the deploy:
    the image exists on the host and the lane keeps running the old one.
    """
    cmd = _dev_cmd()
    lock_path = Path(
        lane_lock_path(lane_config_for(EnumRuntimeLane.DEV).compose_project)
    )
    executor = _LockProbingExecutor(lock_path)
    agent = _make_agent(tmp_path, monkeypatch, cmd, executor)

    agent._run_deploy(cmd)

    assert [call["entry"] for call in _FakeApplier.calls] == ["apply"]
    assert executor.delivered, (
        "the lab-overlay apply built a fresh onex-api image and nothing "
        "advanced ONEX_API_IMAGE, so the lane keeps running the old pin"
    )
    # CORRECTED BY THE SAME TICKET THAT WROTE IT. This asserted ``== SHA``, the
    # merged omnibase_infra sha, which is the value the shipped code passed and
    # is not a lineage any lab image has ever carried -- so the test agreed with
    # the defect and stayed green through thirty red canary runs. The lineage
    # belongs to the overlay; ``test_onex_api_pin_lineage_omn18572`` carries the
    # measurement.
    assert executor.delivered[0]["sha"] == OVERLAY_SHA


async def test_delivery_runs_after_the_apply_that_builds_the_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Order is load-bearing: delivering before the build pins a stale tag."""
    cmd = _dev_cmd()
    lock_path = Path(
        lane_lock_path(lane_config_for(EnumRuntimeLane.DEV).compose_project)
    )
    executor = _LockProbingExecutor(lock_path)

    order: list[str] = []
    applier_apply = _FakeApplier.apply

    def _apply(self: _FakeApplier, **kwargs: Any) -> Path:
        order.append("apply")
        return applier_apply(self, **kwargs)

    monkeypatch.setattr(_FakeApplier, "apply", _apply)
    deliver = executor.deliver_onex_api_pin

    def _deliver(**kwargs: Any) -> dict[str, Any]:
        order.append("deliver")
        return deliver(**kwargs)

    executor.deliver_onex_api_pin = _deliver  # type: ignore[method-assign]
    agent = _make_agent(tmp_path, monkeypatch, cmd, executor)

    agent._run_deploy(cmd)

    assert order == ["apply", "deliver"]


async def test_delivery_does_not_run_when_the_deploy_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: a failed job must not repoint the lane.

    The same argument OMN-18545 made for the repair build taking a narrower
    path than the apply. A job that failed has not proven the image it would
    pin corresponds to anything the lane can run.
    """
    cmd = _dev_cmd()
    lock_path = Path(
        lane_lock_path(lane_config_for(EnumRuntimeLane.DEV).compose_project)
    )
    executor = _LockProbingExecutor(lock_path)

    def _boom(*args: object, **kwargs: object) -> list[str]:
        executor.calls.append("rebuild_scope")
        raise RuntimeError("synthetic rebuild failure")

    executor.rebuild_scope = _boom  # type: ignore[method-assign]
    agent = _make_agent(tmp_path, monkeypatch, cmd, executor)

    agent._run_deploy(cmd)

    assert executor.delivered == []


async def test_delivery_is_dev_lane_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: no governed lane is ever repointed by this path.

    ``onex-api`` on a stability-class lane is not this mechanism's business,
    and a repoint that fired there would be a governed-lane mutation with no
    attribution record.
    """
    cmd = ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.PROD,
        image_digest="sha256:" + "c" * 64,
    )
    lock_path = Path(
        lane_lock_path(lane_config_for(EnumRuntimeLane.PROD).compose_project)
    )
    executor = _LockProbingExecutor(lock_path)
    agent = _make_agent(tmp_path, monkeypatch, cmd, executor)

    agent._run_deploy(cmd)

    assert executor.delivered == []
