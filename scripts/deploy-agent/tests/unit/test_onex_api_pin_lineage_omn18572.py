# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18572 -- the pin delivery asked for the wrong repository's lineage.

THE DEFECT, MEASURED ON THE LAB
-------------------------------
The delivery path merged on 2026-09-17 and ran on every dev deploy afterwards.
It never delivered anything. Read live from the agent's own journal on
``.201`` on 2026-09-19:

    onex-api pin not advanced (REFUSED); no recreate: positive control: the
    daemon lists 14 onex-lab/omnicloud-core tag(s), of which 14 are
    applier-built, so this is an absence and not an unreadable daemon; none of
    them for omninode_infra sha 306f7430 can be pinned.

``306f7430`` is an **omnibase_infra** commit -- the merge this deploy job was
keyed by. The four lab images are tagged ``<omninode_infra sha8>-<stamp>``
(``lab_overlay._derive_pins``), and in the very same job the applier had just
built ``onex-lab/omnicloud-core:551bca5b-20260919T094930Z``. The search was for
a lineage no image has ever carried, so the refusal was structural: it could not
have succeeded on any host, on any day, for any merge.

Two things made it survive a month of deploys:

* **The caller had the wrong sha to hand.** ``_resolve_lab_overlay_sha`` answers
  "may this job touch the lab at all", and its answer is the merged
  omnibase_infra sha. It was reused as the pin lineage because reusing the fence
  was the right instinct for the fence and the wrong value for the pin.
* **Its own error message named the wrong repository.** The refusal interpolates
  the caller's sha under the label ``omninode_infra sha``, so it read as an
  applier that had produced nothing rather than as a caller asking for the
  wrong lineage -- and the conclusion an operator drew from it was that the lab
  overlay had not run.

WHAT THESE TESTS PIN
--------------------
1. The delivery is asked for the **omninode_infra** commit the apply resolved,
   never the merged omnibase_infra sha. This is the correctness test and the one
   that is RED against the merged tree.
2. Every outcome reaches the terminal event and the job record as a typed
   verdict, so a refusal is legible without reading a journal.
3. A failing delivery is logged at ``ERROR``. Thirty consecutive refusals at
   ``INFO`` are indistinguishable from thirty successful no-ops, which is how
   this survived.
4. A delivery with no resolvable lineage is a NAMED non-attempt and does NOT
   fall back to "the newest resident image, whatever its lineage" -- that
   fallback would pin another merge's build and record it as this one's.
5. None of it changes the deploy's verdict. The compose lane converged on its
   own merits before any of this runs.

Every zero carries a positive control: the delivery-not-called assertions are
paired with a case in the same file where the same double IS called.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.events import (
    EnumOnexApiDeliveryResult,
    EnumRuntimeLane,
    ModelOnexApiDelivery,
    ModelRebuildRequested,
    Scope,
)
from deploy_agent.job_state import JobStore

pytestmark = pytest.mark.unit

#: The merged omnibase_infra sha this job is keyed by -- the WRONG lineage.
INFRA_SHA = "a" * 40
#: The omninode_infra overlay commit the apply archived -- the RIGHT lineage.
OVERLAY_SHA = "b" * 40

STALE_PIN = "onex-lab/omnicloud-core:99fdbd37-20260917T110254Z"
FRESH_PIN = "onex-lab/omnicloud-core:bbbbbbbb-20260919T094930Z"


class _FakeExecutor:
    """The executor surface the agent's deploy path touches, and nothing more."""

    def __init__(self, delivery_record: dict[str, Any] | None = None) -> None:
        self.delivered: list[dict[str, Any]] = []
        self.container_residue: list[object] = []
        self.sibling_source_refs: dict[str, str] = {}
        self.recreate_supervision: list[object] = []
        self.verify_recreate: list[object] = []
        self.deps_convergence: list[object] = []
        self.compose_invocations: list[object] = []
        # OMN-18640 AC8: the agent reads the executor's OWN recording of the
        # probe readings when its local is empty, which is precisely the job
        # whose readings matter -- the one where verification refused and the
        # assignment never happened. The real executor declares this in its
        # __init__ (executor.py) and records into it before it raises, so a
        # fake that omits it is not a smaller executor, it is a different one.
        self.health_checks: list[object] = []
        self.raise_on_deliver: Exception | None = None
        self._delivery_record = delivery_record or {
            "result": "WRITTEN",
            "pin_before": STALE_PIN,
            "pin_after": FRESH_PIN,
            "tag_advanced": True,
            "recreated": True,
        }

    def resolve_stability_ready_digest(self, service: str = "") -> str | None:
        return None

    def preflight(self, **kwargs: object) -> None:
        return None

    def git_pull(self, git_ref: str, **kwargs: object) -> str:
        return INFRA_SHA

    def compose_gen(self, bundles: list[str], **kwargs: object) -> None:
        return None

    def seed_infisical(self, **kwargs: object) -> None:
        return None

    def validate_llm_endpoint_env_contract(self) -> None:
        return None

    def rebuild_scope(self, *args: object, **kwargs: object) -> list[str]:
        return ["omninode-runtime"]

    def verify(self, **kwargs: object) -> list[object]:
        return []

    def deploy_and_verify(self, **kwargs: object) -> list[object]:
        return []

    def deliver_onex_api_pin(self, **kwargs: Any) -> dict[str, Any]:
        self.delivered.append(kwargs)
        if self.raise_on_deliver is not None:
            raise self.raise_on_deliver
        return dict(self._delivery_record)


class _FakeApplier:
    """Stands in for ``LabOverlayApplier``, carrying the overlay's own lineage.

    ``manifest_sha`` is the attribute under test: the real applier sets it from
    ``git rev-parse origin/dev`` inside the omninode_infra clone, and it is the
    sha the four image tags carry.
    """

    #: Set per test. ``None`` models an apply that never reached its source.
    manifest_sha_value: str | None = OVERLAY_SHA
    #: Set per test. Models an apply that raised partway through.
    raise_on_apply: Exception | None = None
    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.manifest_sha: str | None = None

    def apply(self, *, sha: str, stamp: str, correlation_id: str) -> Path:
        _FakeApplier.calls.append({"entry": "apply", "sha": sha})
        # The real applier sets this the moment it archives the overlay, before
        # anything that can fail later, which is why a raising apply can still
        # leave a resolvable lineage behind.
        self.manifest_sha = _FakeApplier.manifest_sha_value
        if _FakeApplier.raise_on_apply is not None:
            raise _FakeApplier.raise_on_apply
        return Path(f"/state/lab-overlay/{sha}.json")

    def build_repair_migrate_image(
        self, *, sha: str, stamp: str, correlation_id: str
    ) -> Path:
        _FakeApplier.calls.append({"entry": "repair", "sha": sha})
        return Path(f"/state/lab-overlay/{sha}.json")

    @classmethod
    def reset(cls) -> None:
        cls.calls = []
        cls.manifest_sha_value = OVERLAY_SHA
        cls.raise_on_apply = None


@pytest.fixture(autouse=True)
def _isolated_lane_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock_dir = tmp_path / "lane-locks"
    lock_dir.mkdir()
    monkeypatch.setenv("ONEX_LANE_LOCK_DIR", str(lock_dir))
    monkeypatch.delenv("ONEX_LANE_LOCK_HELD", raising=False)
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


def _make_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cmd: ModelRebuildRequested,
    executor: _FakeExecutor,
    published: list[dict[str, Any]] | None = None,
) -> DeployAgent:
    store = JobStore(tmp_path / "jobs")
    store.accept(cmd.correlation_id, cmd.model_dump(mode="json"))
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    agent = DeployAgent(skip_self_update=True)
    agent.job_store = store
    agent.executor = executor  # type: ignore[assignment]

    def _publish(payload: dict[str, Any], config: object) -> bool:
        if published is not None:
            published.append(payload)
        return True

    monkeypatch.setattr(agent_mod, "publish_result", _publish)
    return agent


# --------------------------------------------------------------------------- #
# 1. The lineage. This is the defect.                                          #
# --------------------------------------------------------------------------- #


async def test_delivery_asks_for_the_overlay_commit_not_the_merged_infra_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--sha`` must be the omninode_infra commit the tags actually carry.

    RED against the merged tree, which passes ``_resolve_lab_overlay_sha``'s
    answer -- the merged omnibase_infra sha -- and so searches for a lineage no
    lab image has ever been built from.
    """
    cmd = _dev_cmd()
    executor = _FakeExecutor()
    agent = _make_agent(tmp_path, monkeypatch, cmd, executor)

    agent._run_deploy(cmd)

    assert executor.delivered, "the delivery was never attempted at all"
    asked_for = executor.delivered[0]["sha"]
    assert asked_for == OVERLAY_SHA, (
        "the pin delivery was asked for "
        f"{asked_for!r}. The lab image tags carry the omninode_infra overlay "
        f"commit ({OVERLAY_SHA!r}); asking for the merged omnibase_infra sha "
        f"({INFRA_SHA!r}) matches no image that has ever been built, which is "
        "why every delivery on the lab refused."
    )
    # Positive control for the assertion above: the two shas differ, so the
    # test cannot pass by their being accidentally equal.
    assert OVERLAY_SHA != INFRA_SHA


async def test_the_apply_still_receives_the_merged_infra_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record stays keyed by the merge; only the PIN lineage changed.

    The lab-overlay record is read back by sha-keyed receipts in CI, so moving
    its key would break rule 24(a)'s delivery gate while fixing the pin.
    """
    cmd = _dev_cmd()
    executor = _FakeExecutor()
    agent = _make_agent(tmp_path, monkeypatch, cmd, executor)

    agent._run_deploy(cmd)

    assert [c["sha"] for c in _FakeApplier.calls] == [INFRA_SHA]


# --------------------------------------------------------------------------- #
# 2/3. Every outcome is surfaced, on the terminal event and the job record     #
# --------------------------------------------------------------------------- #


async def test_a_refusal_rides_the_terminal_event_as_a_named_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cmd = _dev_cmd()
    executor = _FakeExecutor(
        {"result": "REFUSED", "reason": "no candidate for that lineage"}
    )
    published: list[dict[str, Any]] = []
    agent = _make_agent(tmp_path, monkeypatch, cmd, executor, published)

    agent._run_deploy(cmd)

    assert len(published) == 1
    delivery = published[0]["onex_api_delivery"]
    assert delivery is not None, (
        "the terminal event carried nothing about the pin delivery, so a lane "
        "running a two-day-old image is indistinguishable on the bus from one "
        "running the merge"
    )
    assert delivery["result"] == EnumOnexApiDeliveryResult.REFUSED.value
    assert delivery["is_failure"] is True
    assert delivery["requested_sha"] == OVERLAY_SHA
    # A refusal carries neither boolean, and `None` would read as "not known".
    assert delivery["tag_advanced"] is False
    assert delivery["recreated"] is False
    # And the deploy itself still succeeded: the compose lane converged.
    assert published[0]["status"] == "success"


async def test_a_successful_delivery_is_not_reported_as_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positive control for the test above: the same field on the good path."""
    cmd = _dev_cmd()
    executor = _FakeExecutor()
    published: list[dict[str, Any]] = []
    agent = _make_agent(tmp_path, monkeypatch, cmd, executor, published)

    agent._run_deploy(cmd)

    delivery = published[0]["onex_api_delivery"]
    assert delivery["result"] == EnumOnexApiDeliveryResult.WRITTEN.value
    assert delivery["is_failure"] is False
    assert delivery["tag_advanced"] is True
    assert delivery["recreated"] is True
    assert delivery["pin_after"] == FRESH_PIN


async def test_the_verdict_is_durable_on_the_job_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator on the host reads the job record, not the bus."""
    cmd = _dev_cmd()
    executor = _FakeExecutor({"result": "REFUSED", "reason": "nothing to pin"})
    agent = _make_agent(tmp_path, monkeypatch, cmd, executor)

    agent._run_deploy(cmd)

    job = agent.job_store.load(cmd.correlation_id)
    assert job is not None
    assert job.onex_api_delivery is not None
    assert job.onex_api_delivery.is_failure is True
    assert job.onex_api_delivery.requested_sha == OVERLAY_SHA


# --------------------------------------------------------------------------- #
# 4. The log level is the difference between visible and invisible             #
# --------------------------------------------------------------------------- #


async def test_a_failing_delivery_is_logged_at_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cmd = _dev_cmd()
    executor = _FakeExecutor({"result": "REFUSED", "reason": "nothing to pin"})
    agent = _make_agent(tmp_path, monkeypatch, cmd, executor)

    with caplog.at_level(logging.INFO, logger="deploy_agent.agent"):
        agent._run_deploy(cmd)

    delivery_lines = [
        r for r in caplog.records if "onex-api delivery" in r.getMessage()
    ]
    assert delivery_lines, "the delivery verdict was not logged at all"
    assert all(r.levelno == logging.ERROR for r in delivery_lines), (
        "a delivery that did not deliver was logged at "
        f"{[r.levelname for r in delivery_lines]}. Thirty consecutive refusals "
        "at INFO read exactly like thirty successful no-ops, which is how this "
        "survived a month of deploys."
    )


async def test_a_successful_delivery_is_not_logged_at_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Positive control: the level tracks the verdict rather than being fixed."""
    cmd = _dev_cmd()
    executor = _FakeExecutor()
    agent = _make_agent(tmp_path, monkeypatch, cmd, executor)

    with caplog.at_level(logging.INFO, logger="deploy_agent.agent"):
        agent._run_deploy(cmd)

    delivery_lines = [
        r for r in caplog.records if "onex-api delivery" in r.getMessage()
    ]
    assert delivery_lines
    assert all(r.levelno == logging.INFO for r in delivery_lines)


# --------------------------------------------------------------------------- #
# 5. No lineage is a named non-attempt, never a floating-ref fallback          #
# --------------------------------------------------------------------------- #


async def test_no_resolvable_lineage_is_a_named_non_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no overlay commit, nothing is delivered and the event says so."""
    _FakeApplier.manifest_sha_value = None
    cmd = _dev_cmd()
    executor = _FakeExecutor()
    published: list[dict[str, Any]] = []
    agent = _make_agent(tmp_path, monkeypatch, cmd, executor, published)

    agent._run_deploy(cmd)

    assert executor.delivered == [], (
        "with no resolvable overlay lineage the delivery fell back to the "
        "newest resident image, which pins another merge's build and records "
        "it as this one's"
    )
    delivery = published[0]["onex_api_delivery"]
    assert delivery["result"] == EnumOnexApiDeliveryResult.NOT_ATTEMPTED.value
    assert delivery["is_failure"] is True
    assert delivery["requested_sha"] is None


async def test_an_apply_that_raises_after_resolving_its_source_still_delivers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lineage survives a partial apply, and so does the delivery.

    The applier resolves its source before anything that can fail later, and
    the image built from that lineage is on the host either way -- so a raising
    apply must still produce a delivery that NAMES the lineage rather than a
    silent nothing. Positive control for the zero above: the same double, the
    same code path, and here the delivery IS called.
    """
    _FakeApplier.raise_on_apply = RuntimeError("apply died at the migrate barrier")
    cmd = _dev_cmd()
    executor = _FakeExecutor()
    agent = _make_agent(tmp_path, monkeypatch, cmd, executor)

    agent._run_deploy(cmd)

    assert [c["sha"] for c in executor.delivered] == [OVERLAY_SHA]


# --------------------------------------------------------------------------- #
# 6. A raising delivery is recorded, not dropped                               #
# --------------------------------------------------------------------------- #


async def test_a_delivery_that_raises_is_recorded_and_does_not_fail_the_deploy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cmd = _dev_cmd()
    executor = _FakeExecutor()
    executor.raise_on_deliver = RuntimeError("docker socket went away")
    published: list[dict[str, Any]] = []
    agent = _make_agent(tmp_path, monkeypatch, cmd, executor, published)

    agent._run_deploy(cmd)

    assert len(published) == 1, "the terminal publish was skipped by the raise"
    assert published[0]["status"] == "success"
    delivery = published[0]["onex_api_delivery"]
    assert delivery["result"] == EnumOnexApiDeliveryResult.RAISED.value
    assert delivery["is_failure"] is True
    assert "docker socket went away" in (delivery["reason"] or "")


# --------------------------------------------------------------------------- #
# 7. The verdict vocabulary cannot silently read an unknown value as success   #
# --------------------------------------------------------------------------- #


def test_an_unknown_verdict_is_a_failure_and_keeps_its_spelling() -> None:
    """The verdict comes from a subprocess whose vocabulary can move.

    Mapping an unrecognised value onto anything benign would let a future
    refusal token read as a delivery.
    """
    record = ModelOnexApiDelivery.from_record(
        {"result": "SOMETHING_NEW", "reason": "a token this enum does not know"},
        requested_sha=OVERLAY_SHA,
    )

    assert record.result is EnumOnexApiDeliveryResult.UNRECOGNISED
    assert record.raw_result == "SOMETHING_NEW"
    assert record.is_failure is True


def test_unchanged_and_skipped_are_not_failures() -> None:
    """A lane already running this image is not a delivery failure."""
    for verdict in ("UNCHANGED", "SKIPPED"):
        record = ModelOnexApiDelivery.from_record(
            {"result": verdict, "reason": "n/a"}, requested_sha=OVERLAY_SHA
        )
        assert record.is_failure is False, verdict
