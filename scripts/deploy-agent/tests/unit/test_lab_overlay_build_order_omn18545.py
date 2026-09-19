# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18545 -- the lab-overlay build must not sit behind the preflight it feeds.

THE DEFECT THESE TESTS PIN
--------------------------
The compose dev lane runs whatever ``ONEX_CLOUD_MIGRATE_IMAGE`` names. The deploy
job's dev-lane migration preflight requires the ``cloud-migration-files`` and
``cloud-migration`` one-shots to exit 0 **using that image**
(``executor._ensure_runtime_migrations_ready``, reached from ``_compose_up`` for
the RUNTIME phase, i.e. from inside ``rebuild_scope``). The only in-repo path that
builds a replacement CLOUD-migrate image -- ``docker/Dockerfile.migrate`` against
the archived ``omninode_infra`` overlay tree -- is the lab-overlay applier
(``lab_overlay.MIGRATE_DOCKERFILE`` / ``CLOUD_MIGRATE_IMAGE_NAME``).
``build-and-push-migrate-image.yml`` builds the same Dockerfile against THIS
repo's tree and pushes the INFRA migrate image to ECR; it is a different image
and does not satisfy the pin the dev lane resolves.

Before this change the applier was invoked from exactly one place: the LAST
statement of the deploy job's ``try``, after ``executor.verify``. So while the
pinned migrate image was broken the preflight raised, control jumped to the
``except``, and the build never ran -- the agent could not build the image that
would let the preflight pass. Observed live three times on 2026-09-16; the third
(job ``6d8316f0``) reached terminal ``failed`` at 20:44:18Z with
``verification: skipped``, and the newest ``onex-lab/omninode-cloud-migrate``
tag on the host stayed the 17:33Z pre-merge one throughout.

THE FAILING PATH TAKES A NARROWER CALL THAN THE SUCCESS PATH, and these tests
assert that difference rather than treating the two as interchangeable. Success
still runs the full ``apply``. Failure runs ``build_repair_migrate_image``: one
image build, no runtime promotion, no lane apply -- because the apply promotes
``omnibase-infra-omninode-runtime:latest`` on the premise that this agent just
built it from the merged sha, which a failed job can falsify, and the record's
own readback checks compare against tags the same run minted and so cannot catch
it. A full apply on the failing path would therefore roll the persistent k3s lane
to a tag NAMING the merged sha while it ran the previous commit's binary, and
report PASS.

WHAT THE TESTS ASSERT, AND WHY EACH IS SHAPED THIS WAY
------------------------------------------------------
Every test here drives the real ``DeployAgent._run_deploy`` -- the actual
consume/execute path -- with the ``_FakeExecutor`` pattern the neighbouring
agent-level tests already use. None of them reads the call's position in the
source, because a line-number assertion passes against a tree where the behaviour
regressed for any other reason (a new early ``return``, a second ``except``, a
guard clause), and fails against a tree where the behaviour is correct but the
statement moved. The failure path is EXERCISED instead.

The controls matter as much as the assertions. A suite that only proved "the
applier ran" would pass just as happily against an agent that applied the overlay
on every job on every lane, which is a different defect: the overlay is not a
stability surface and a merge to ``main`` targets stability-test, a governed lane
this agent's fence already refuses. So a prod job that fails early and a dev job
that fails before the sha is resolved are both asserted to leave the applier
untouched, in the same run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.events import EnumRuntimeLane, ModelRebuildRequested, Scope
from deploy_agent.executor import DevLaneMigrationPreflightError
from deploy_agent.job_state import JobStore

pytestmark = pytest.mark.unit

SHA = "a" * 40
OTHER_SHA = "b" * 40

#: The live message, copied from ``executor._ensure_runtime_migrations_ready``.
PREFLIGHT_ERROR = (
    "Dev-lane migration preflight did not complete cloud-migration: "
    "['cloud-migration']. A one-shot that has not exited 0 has not proven it "
    "did its job."
)
OVERLAY_ERROR = "lab overlay: containerd import refused"

#: An unrelated dev-lane failure. Its own measured shape, from the OMN-18134
#: incident: the gateway step refused an unpinned build and took the job down.
#: A fresh cloud-migrate image could not have helped, so the repair build must
#: not run for it.
GATEWAY_ERROR = RuntimeError(
    "GATEWAY_DEPLOY_FAILED: bash scripts/deploy-gateway.sh --execute exited 5. "
    "stderr: ERROR: DEPLOY_REF unset -- refusing to stage the AMBIENT host tree."
)

_DIGEST = "sha256:" + "c" * 64
_OTHER_DIGEST = "sha256:" + "d" * 64


class _FakeExecutor:
    """Records call order; each failure seam is injectable.

    Mirrors ``test_agent_prod_stability_digest_guard_wiring_omn15181._FakeExecutor``
    and carries the two attributes the agent reads when it builds the terminal
    event (``container_residue`` -- OMN-18057, ``sibling_source_refs`` --
    OMN-17135).
    """

    def __init__(
        self,
        *,
        rebuild_error: Exception | None = None,
        preflight_error: str | None = None,
        git_sha: str = SHA,
        stability_ready_digest: str | None = _DIGEST,
    ) -> None:
        self.calls: list[str] = []
        self._rebuild_error = rebuild_error
        self._preflight_error = preflight_error
        self._git_sha = git_sha
        self._stability_ready_digest = stability_ready_digest
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
        # OMN-18640 AC8: the agent publishes the executor's own probe
        # readings when its local list is empty, which is the case on
        # every job that failed verification.
        self.health_checks: list[object] = []

    def resolve_stability_ready_digest(
        self, service: str = "omninode-runtime"
    ) -> str | None:
        self.calls.append("resolve_stability_ready_digest")
        return self._stability_ready_digest

    def preflight(self, **kwargs: object) -> None:
        self.calls.append("preflight")
        if self._preflight_error:
            raise RuntimeError(self._preflight_error)

    def git_pull(self, git_ref: str, **kwargs: object) -> str:
        self.calls.append("git_pull")
        return self._git_sha

    def compose_gen(self, bundles: list[str], **kwargs: object) -> None:
        self.calls.append("compose_gen")

    def seed_infisical(self, **kwargs: object) -> None:
        self.calls.append("seed_infisical")

    def validate_llm_endpoint_env_contract(self) -> None:
        self.calls.append("validate_llm_endpoint_env_contract")

    def rebuild_scope(self, *args: object, **kwargs: object) -> list[str]:
        self.calls.append("rebuild_scope")
        if self._rebuild_error is not None:
            # The dev-lane migration preflight raises from inside here:
            # rebuild_scope -> _compose_up(RUNTIME) ->
            # _ensure_runtime_migrations_ready. The TYPE is what the agent acts
            # on, so the double raises the real one.
            raise self._rebuild_error
        return ["omninode-runtime"]

    def verify(self, **kwargs: object) -> list[object]:
        self.calls.append("verify")
        return []

    def deploy_and_verify(self, **kwargs: object) -> list[object]:
        self.calls.append("deploy_and_verify")
        return []


class _FakeApplier:
    """Stands in for ``LabOverlayApplier``; records every call it is asked for.

    Both entry points land in ``calls`` tagged with which one ran, so a test can
    assert not only THAT the overlay path was reached but WHICH of the two it
    took -- the distinction the repair build exists to make.
    """

    instances: list[_FakeApplier] = []
    calls: list[dict[str, Any]] = []
    #: Set by a test to make the call blow up the way a real one can.
    raise_with: str | None = None
    #: Read at call time so a test can prove the verdict was recorded FIRST.
    observed_status: list[str | None] = []
    status_source: JobStore | None = None
    status_cid: Any = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        _FakeApplier.instances.append(self)

    def _record(self, entry: str, sha: str, correlation_id: str) -> Path:
        _FakeApplier.calls.append(
            {"entry": entry, "sha": sha, "correlation_id": correlation_id}
        )
        store = _FakeApplier.status_source
        if store is not None and _FakeApplier.status_cid is not None:
            job = store.load(_FakeApplier.status_cid)
            _FakeApplier.observed_status.append(getattr(job, "status", None))
        if _FakeApplier.raise_with:
            raise RuntimeError(_FakeApplier.raise_with)
        return Path(f"/state/lab-overlay/{sha}.json")

    def apply(self, *, sha: str, stamp: str, correlation_id: str) -> Path:
        return self._record("apply", sha, correlation_id)

    def build_repair_migrate_image(
        self, *, sha: str, stamp: str, correlation_id: str
    ) -> Path:
        return self._record("repair", sha, correlation_id)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []
        cls.calls = []
        cls.raise_with = None
        cls.observed_status = []
        cls.status_source = None
        cls.status_cid = None


@pytest.fixture(autouse=True)
def _fake_applier(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeApplier.reset()
    monkeypatch.setattr(agent_mod, "LabOverlayApplier", _FakeApplier)
    monkeypatch.setattr(agent_mod, "LAB_OVERLAY_ENABLED", True)


def _dev_cmd() -> ModelRebuildRequested:
    return ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.DEV,
    )


def _prod_cmd(image_digest: str = _DIGEST) -> ModelRebuildRequested:
    return ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.PROD,
        image_digest=image_digest,
    )


def _make_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cmd: ModelRebuildRequested,
    fake_executor: _FakeExecutor,
) -> tuple[DeployAgent, JobStore]:
    store = JobStore(tmp_path)
    store.accept(cmd.correlation_id, cmd.model_dump(mode="json"))
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    agent = DeployAgent(skip_self_update=True)
    agent.job_store = store
    agent.executor = fake_executor  # type: ignore[assignment]
    monkeypatch.setattr(agent_mod, "publish_result", lambda payload, config: False)
    return agent, store


# --------------------------------------------------------------------------- #
# AC1 / AC3 -- the build is reachable on a job whose preflight failed          #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_a_failing_migration_preflight_still_reaches_the_overlay_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The circularity, exercised rather than read.

    RED against the tree where ``_apply_lab_overlay`` is the last statement of
    the job's ``try``: the preflight raises, control jumps to the ``except``, and
    the only path that builds a replacement migrate image never runs -- so the
    image the preflight needs can never come into existence.
    """
    cmd = _dev_cmd()
    fake_executor = _FakeExecutor(
        rebuild_error=DevLaneMigrationPreflightError(PREFLIGHT_ERROR)
    )
    agent, store = _make_agent(tmp_path, monkeypatch, cmd, fake_executor)

    agent._run_deploy(cmd)

    job = store.load(cmd.correlation_id)
    assert job is not None
    assert job.status == "failed", (
        "the preflight failure must still be the job's verdict; a reachable "
        "overlay build is not a reason to call a broken lane converged"
    )
    assert [(entry["entry"], entry["sha"]) for entry in _FakeApplier.calls] == [
        ("repair", SHA)
    ], (
        "the lab-overlay applier did not run on a job whose dev-lane migration "
        "preflight failed, so the only in-repo builder of a replacement "
        "cloud-migrate image never executes while the pinned one is broken -- "
        "the loop cannot open on its own (OMN-18545). The entry must be the "
        "REPAIR build, never the full apply: a failed job has not proven the "
        "runtime image the apply would promote was built from this sha."
    )


@pytest.mark.asyncio
async def test_the_overlay_build_runs_for_the_sha_this_job_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A build for the wrong sha is not a usable replacement image.

    The record is keyed by sha and the migrate image is built from the clone at
    that sha, so an applier invoked with a neighbour job's sha writes a record
    naming a commit this job never deployed.
    """
    cmd = _dev_cmd()
    fake_executor = _FakeExecutor(
        rebuild_error=DevLaneMigrationPreflightError(PREFLIGHT_ERROR), git_sha=OTHER_SHA
    )
    agent, _ = _make_agent(tmp_path, monkeypatch, cmd, fake_executor)

    agent._run_deploy(cmd)

    assert [entry["sha"] for entry in _FakeApplier.calls] == [OTHER_SHA]
    assert [entry["correlation_id"] for entry in _FakeApplier.calls] == [
        str(cmd.correlation_id)
    ]


# --------------------------------------------------------------------------- #
# AC2 -- the overlay never changes the compose lane's verdict                  #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_an_overlay_failure_does_not_break_a_lane_running_the_merged_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``agent.py``'s own words, pinned by a test rather than by a comment: "a
    lab-overlay failure must not report a lane that IS running the merged sha as
    broken"."""
    cmd = _dev_cmd()
    fake_executor = _FakeExecutor()
    agent, store = _make_agent(tmp_path, monkeypatch, cmd, fake_executor)
    _FakeApplier.raise_with = OVERLAY_ERROR

    agent._run_deploy(cmd)

    job = store.load(cmd.correlation_id)
    assert job is not None
    assert job.status == "success", (
        "a failing lab-overlay apply flipped a verified compose deploy to "
        "failed; the lab verdict travels in its own sha-keyed receipt"
    )
    assert not [error for error in job.errors if OVERLAY_ERROR in error], (
        "the overlay's failure leaked into the compose job's errors"
    )
    assert [entry["entry"] for entry in _FakeApplier.calls] == ["apply"], (
        "the success path must still take the FULL apply; the repair build is "
        "the failing path's narrower substitute, not a replacement for it"
    )


@pytest.mark.asyncio
async def test_neither_failure_masks_the_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both halves fail at once: the preflight error is the job's verdict, and
    the overlay still ran and still failed on its own surface.

    A fix that reported only one of the two would be the same false-green shape
    the lab receipt exists to remove -- "it failed" and "nobody ran it" have to
    stay distinguishable.
    """
    cmd = _dev_cmd()
    fake_executor = _FakeExecutor(
        rebuild_error=DevLaneMigrationPreflightError(PREFLIGHT_ERROR)
    )
    agent, store = _make_agent(tmp_path, monkeypatch, cmd, fake_executor)
    _FakeApplier.raise_with = OVERLAY_ERROR

    agent._run_deploy(cmd)

    job = store.load(cmd.correlation_id)
    assert job is not None
    assert job.status == "failed"
    assert any(PREFLIGHT_ERROR in error for error in job.errors), (
        f"the job recorded {job.errors!r} rather than the preflight's own "
        "message; the overlay masked the real failure"
    )
    assert not [error for error in job.errors if OVERLAY_ERROR in error]
    assert [entry["entry"] for entry in _FakeApplier.calls] == ["repair"], (
        "the repair attempt was not made, so its failure is invisible"
    )


@pytest.mark.asyncio
async def test_the_verdict_is_recorded_before_the_overlay_is_attempted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ordering is what makes the isolation structural rather than incidental.

    If the applier ran while the job were still ``in_progress``, an overlay
    failure could only be kept out of the verdict by remembering to, which is
    the property AC2 refuses to leave to a comment.
    """
    cmd = _dev_cmd()
    fake_executor = _FakeExecutor(
        rebuild_error=DevLaneMigrationPreflightError(PREFLIGHT_ERROR)
    )
    agent, store = _make_agent(tmp_path, monkeypatch, cmd, fake_executor)
    _FakeApplier.status_source = store
    _FakeApplier.status_cid = cmd.correlation_id

    agent._run_deploy(cmd)

    assert _FakeApplier.observed_status == ["failed"], (
        "the job's terminal status was not already written when the overlay "
        f"ran; observed {_FakeApplier.observed_status!r}"
    )


# --------------------------------------------------------------------------- #
# AC4 -- positive controls: paths where the overlay must stay skipped          #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_a_failing_prod_job_never_touches_the_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control that separates this fix from "apply on every job".

    A merge to ``main`` targets stability-test, a governed lane this agent's
    fence refuses, and the overlay is not a stability surface. This prod command
    is rejected by the OMN-15181 guard before any deploy effect -- the earliest
    possible failure -- and the overlay must still not run.
    """
    cmd = _prod_cmd(image_digest=_DIGEST)
    fake_executor = _FakeExecutor(stability_ready_digest=_OTHER_DIGEST)
    agent, store = _make_agent(tmp_path, monkeypatch, cmd, fake_executor)

    agent._run_deploy(cmd)

    job = store.load(cmd.correlation_id)
    assert job is not None
    assert job.status == "failed"
    assert fake_executor.calls == ["resolve_stability_ready_digest"]
    assert _FakeApplier.calls == [], (
        "a non-dev lane reached the lab-overlay applier; the overlay is a dev "
        "lane surface only"
    )
    assert _FakeApplier.instances == [], (
        "the applier was constructed for a prod job even though it did not "
        "apply -- it reads the operator env store on construction"
    )


@pytest.mark.asyncio
async def test_a_successful_prod_job_never_touches_the_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same control on the passing side, so the skip is not an artifact of
    the job having failed."""
    cmd = _prod_cmd(image_digest=_DIGEST)
    fake_executor = _FakeExecutor(stability_ready_digest=_DIGEST)
    agent, store = _make_agent(tmp_path, monkeypatch, cmd, fake_executor)

    agent._run_deploy(cmd)

    job = store.load(cmd.correlation_id)
    assert job is not None
    assert job.status == "success"
    assert "deploy_and_verify" in fake_executor.calls
    assert _FakeApplier.calls == []


@pytest.mark.asyncio
async def test_a_dev_job_that_fails_before_the_sha_is_resolved_applies_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stale-sha control.

    ``_current_git_sha`` lives on the long-lived agent, so once the overlay can
    run on a FAILED job a job that dies before ``git_pull`` would otherwise
    re-apply the previous job's sha -- building images for a commit this job
    never deployed and stamping a fresh record over it. The sha is per-job, and
    an unresolved one applies nothing.
    """
    first = _dev_cmd()
    fake_executor = _FakeExecutor()
    agent, store = _make_agent(tmp_path, monkeypatch, first, fake_executor)

    agent._run_deploy(first)
    assert [entry["sha"] for entry in _FakeApplier.calls] == [SHA]

    second = _dev_cmd()
    store.accept(second.correlation_id, second.model_dump(mode="json"))
    agent.executor = _FakeExecutor(  # type: ignore[assignment]
        preflight_error="preflight refused: required compose env unset"
    )

    agent._run_deploy(second)

    job = store.load(second.correlation_id)
    assert job is not None
    assert job.status == "failed"
    assert [entry["sha"] for entry in _FakeApplier.calls] == [SHA], (
        "the second job built for a sha it never resolved; the only candidate "
        "was the FIRST job's, which this job did not deploy"
    )


@pytest.mark.asyncio
async def test_the_overlay_stays_off_when_the_operator_disabled_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``DEPLOY_AGENT_LAB_OVERLAY=off`` is the operator's own kill switch, and
    re-ordering must not turn it into a switch that only works on success."""
    monkeypatch.setattr(agent_mod, "LAB_OVERLAY_ENABLED", False)
    cmd = _dev_cmd()
    fake_executor = _FakeExecutor(
        rebuild_error=DevLaneMigrationPreflightError(PREFLIGHT_ERROR)
    )
    agent, _ = _make_agent(tmp_path, monkeypatch, cmd, fake_executor)

    agent._run_deploy(cmd)

    assert _FakeApplier.calls == []
    assert _FakeApplier.instances == []


# --------------------------------------------------------------------------- #
# The happy path is unchanged                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_a_successful_dev_deploy_still_applies_the_overlay_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cmd = _dev_cmd()
    fake_executor = _FakeExecutor()
    agent, store = _make_agent(tmp_path, monkeypatch, cmd, fake_executor)
    _FakeApplier.status_source = store
    _FakeApplier.status_cid = cmd.correlation_id

    agent._run_deploy(cmd)

    job = store.load(cmd.correlation_id)
    assert job is not None
    assert job.status == "success"
    assert fake_executor.calls == [
        "preflight",
        "git_pull",
        "compose_gen",
        "seed_infisical",
        "validate_llm_endpoint_env_contract",
        "rebuild_scope",
        "verify",
    ]
    assert [(entry["entry"], entry["sha"]) for entry in _FakeApplier.calls] == [
        ("apply", SHA)
    ]
    assert _FakeApplier.observed_status == ["success"]


@pytest.mark.asyncio
async def test_an_unrelated_dev_failure_does_not_trigger_a_repair_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The repair build is targeted, and this is the control that proves it.

    Without this the agent would rebuild the cloud-migrate image on EVERY dev
    failure -- a gateway refusal, an out-of-memory build, an unset compose
    variable -- spending up to sixteen minutes under the single-flight lock,
    which rejects every concurrent rebuild command outright, on an image that
    has nothing to do with the failure. And that path is by construction the
    busy one: while the lane is broken, every merge fails.
    """
    cmd = _dev_cmd()
    fake_executor = _FakeExecutor(rebuild_error=GATEWAY_ERROR)
    agent, store = _make_agent(tmp_path, monkeypatch, cmd, fake_executor)

    agent._run_deploy(cmd)

    job = store.load(cmd.correlation_id)
    assert job is not None
    assert job.status == "failed"
    assert any("GATEWAY_DEPLOY_FAILED" in error for error in job.errors)
    assert _FakeApplier.calls == [], (
        "a failure a new migrate image cannot fix still triggered the repair build"
    )
    assert _FakeApplier.instances == []


@pytest.mark.asyncio
async def test_the_terminal_event_is_still_published_when_the_repair_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The repair build runs inside the deploy job's ``except``, and the publish
    block sits after it.

    An exception escaping the repair build would skip that block entirely: the
    job would be durably ``failed`` on disk with nothing on the bus and
    ``result_publish_pending`` never set, so the retry loop would not replay it
    either. Both docstrings name this as a reason for swallowing; this asserts
    it instead.
    """
    published: list[object] = []
    cmd = _dev_cmd()
    fake_executor = _FakeExecutor(
        rebuild_error=DevLaneMigrationPreflightError(PREFLIGHT_ERROR)
    )
    agent, store = _make_agent(tmp_path, monkeypatch, cmd, fake_executor)
    monkeypatch.setattr(
        agent_mod,
        "publish_result",
        lambda payload, config: bool(published.append(payload)) or True,
    )
    _FakeApplier.raise_with = OVERLAY_ERROR

    agent._run_deploy(cmd)

    assert len(_FakeApplier.calls) == 1
    assert len(published) == 1, (
        "the terminal event was never published; a raise in the repair build "
        "skipped the publish block of a job already recorded as failed"
    )
    job = store.load(cmd.correlation_id)
    assert job is not None
    assert job.status == "failed"
