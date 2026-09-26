# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18276 — the rule 24 lab pass comes from the PERSISTENT lab, never kind.

``omni_home`` ``CLAUDE.md`` Operating Rule 24(b) says delivery to staging fails
closed without a passing lab receipt for the delivered sha. Until this change
the lane named ``onex-lab`` was the *throwaway* ``kind`` cluster the
``candidate-boot-gate`` job creates inside the delivery run, and
``evaluate_gate`` was satisfied by ANY lane passing — so the kind boot alone
satisfied rule 24(b) and the persistent lab's verdict was optional.

MEASURED, and this is the defect rather than a hypothesis. Merge
``17696113e3ccb15adaa9031e07043e41d1d45396`` (``fix(OMN-18640): effects
consumer rejoins its group after coordinator loss``, #3807, 543 changed lines
in ``src/omnibase_infra/event_bus/event_bus_kafka.py``) was delivered to
staging by run 35418114021 with:

* ``lab-pass-receipt-onex-lab-17696113…``      present — the kind boot;
* ``lab-pass-receipt-compose-dev-17696113…``   ABSENT;
* ``lab-pass-receipt-onex-lab-k3s-17696113…``  ABSENT.

Both persistent-lab jobs were ``skipped`` on rebuild run 35418113915. A Kafka
consume-loop rewrite reached the promotion surface having run on nothing but a
single-node ``kind`` cluster with one side-loaded image and an inert credential
store. That is precisely "a sha reaching staging that nothing has ever run".

What these tests pin:

* ``onex-lab`` IS the persistent k3s lab lane, and ``kind-smoke`` is the
  ephemeral render/wiring smoke lane — a separate value, never gate-eligible;
* a receipt whose ``lane`` claims a persistent lab lane while its evidence
  declares an ephemeral kind cluster is REFUSED, on write and on read;
* the gate requires the persistent lane's receipt and cannot be asked about
  ``kind-smoke`` at all;
* the positive control: a ``compose-dev`` receipt is still a valid receipt and
  still satisfies a gate that names it.
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.lab_pass_receipt import (
    CLUSTER_PROVENANCE_CHECK,
    DEFAULT_GATE_LANES,
    EPHEMERAL_CLUSTER_MARKER,
    GATE_ELIGIBLE_LANES,
    EnumLabLane,
    EnumLabPassResult,
    ModelLabPassCheck,
    ModelLabPassReceipt,
    artifact_name,
    build_parser,
    build_receipt,
    evaluate_gate,
)

pytestmark = pytest.mark.unit

SHA = "17696113e3ccb15adaa9031e07043e41d1d45396"
REPO = "OmniNode-ai/omnibase_infra"
AGENT_ID = "bdf92958-8ee6-4be4-9df0-04d3e15d9c59"

STARTED = datetime(2026, 9, 19, 3, 17, 0, tzinfo=UTC)
FINISHED = datetime(2026, 9, 19, 3, 51, 0, tzinfo=UTC)

DELIVER = Path(".github/workflows/deliver-dev-candidate-to-staging.yml")
REBUILD = Path(".github/workflows/runtime-rebuild-trigger.yml")


def _lane_checks() -> list[ModelLabPassCheck]:
    return [
        ModelLabPassCheck(name="ready_main", ok=True, evidence="GET /ready -> 200"),
        ModelLabPassCheck(
            name="overlay_applied",
            ok=True,
            evidence="apply_lab_lane.sh record for this sha, rollout complete",
        ),
    ]


def _smoke_checks() -> list[ModelLabPassCheck]:
    return [
        ModelLabPassCheck(
            name=CLUSTER_PROVENANCE_CHECK,
            ok=True,
            evidence=(
                f"{EPHEMERAL_CLUSTER_MARKER} kind v0.30.0 node "
                "kindest/node:v1.31.0 created by this run and destroyed with it"
            ),
        ),
        ModelLabPassCheck(
            name="manifests_render",
            ok=True,
            evidence="kubectl kustomize k8s/onex-lab against omninode_infra@dev",
        ),
    ]


def _persistent(lane: EnumLabLane = EnumLabLane.ONEX_LAB) -> ModelLabPassReceipt:
    return build_receipt(
        sha=SHA,
        lane=lane,
        started_at=STARTED,
        finished_at=FINISHED,
        checks=_lane_checks(),
        agent_command_id=AGENT_ID,
    )


# ---------------------------------------------------------------------------
# The lane taxonomy (AC1, AC3)
# ---------------------------------------------------------------------------
class TestLaneTaxonomy:
    def test_onex_lab_is_the_persistent_lab_and_kind_smoke_is_its_own_lane(
        self,
    ) -> None:
        """AC1/AC3. ``onex-lab`` names the persistent k3s lab, not a kind boot."""
        assert EnumLabLane.ONEX_LAB.value == "onex-lab"
        assert EnumLabLane.KIND_SMOKE.value == "kind-smoke"
        assert EnumLabLane.KIND_SMOKE is not EnumLabLane.ONEX_LAB

    def test_the_retired_k3s_alias_is_gone(self) -> None:
        """One lane, one name.

        ``onex-lab-k3s`` existed only because ``onex-lab`` was taken by the
        kind boot. With the kind boot moved to its own lane the alias is a
        second name for the thing ``onex-lab`` now IS, and two names for one
        lane is how a gate reads the wrong artifact.
        """
        assert not hasattr(EnumLabLane, "ONEX_LAB_K3S")
        with pytest.raises(ValueError, match="is not a valid EnumLabLane"):
            EnumLabLane("onex-lab-k3s")

    def test_kind_smoke_is_never_gate_eligible(self) -> None:
        """AC3: the kind boot is never CONSUMED as the lab pass."""
        assert EnumLabLane.KIND_SMOKE not in GATE_ELIGIBLE_LANES
        assert EnumLabLane.KIND_SMOKE not in DEFAULT_GATE_LANES

    def test_the_gate_requires_the_persistent_lab_by_default(self) -> None:
        """AC2: the default the delivery gate runs with names the persistent lane."""
        assert DEFAULT_GATE_LANES == (EnumLabLane.ONEX_LAB,)

    def test_the_gate_cli_refuses_kind_smoke_as_a_choice(self) -> None:
        """Argparse-level, so the refusal cannot be argued with in YAML."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["gate", "--sha", SHA, "--lane", "kind-smoke"])
        # Positive control: the two persistent lanes ARE accepted.
        for lane in ("onex-lab", "compose-dev"):
            parsed = parser.parse_args(["gate", "--sha", SHA, "--lane", lane])
            assert parsed.lane == [lane]


# ---------------------------------------------------------------------------
# Provenance: a persistent lane cannot carry ephemeral-kind evidence
# ---------------------------------------------------------------------------
class TestReceiptProvenance:
    def test_a_persistent_lane_receipt_naming_a_kind_cluster_is_refused(self) -> None:
        """THE refusal this ticket exists for.

        A receipt that claims the persistent lab while its own evidence says it
        ran on a throwaway kind node is the relabelling the gate must catch.
        """
        with pytest.raises(ValueError, match="ephemeral"):
            build_receipt(
                sha=SHA,
                lane=EnumLabLane.ONEX_LAB,
                started_at=STARTED,
                finished_at=FINISHED,
                checks=[
                    ModelLabPassCheck(
                        name="manifests_render",
                        ok=True,
                        evidence=f"{EPHEMERAL_CLUSTER_MARKER} kind v0.30.0",
                    )
                ],
                agent_command_id=AGENT_ID,
            )

    def test_the_refusal_also_holds_on_the_gate_s_READ_path(self) -> None:
        """``from_json`` constructs the model, so the reader refuses it too.

        The emitter and the reader are different processes on different hosts;
        a validation that ran only at emit time would be an honour system.
        """
        body = json.loads(_persistent().to_json())
        body["checks"][0]["evidence"] = f"{EPHEMERAL_CLUSTER_MARKER} kind v0.30.0"
        with pytest.raises(ValueError, match="ephemeral"):
            ModelLabPassReceipt.from_json(json.dumps(body))

    def test_compose_dev_is_a_persistent_lane_for_this_purpose_too(self) -> None:
        """The .201 dev lane is a real host; it never runs on kind either."""
        with pytest.raises(ValueError, match="ephemeral"):
            build_receipt(
                sha=SHA,
                lane=EnumLabLane.COMPOSE_DEV,
                started_at=STARTED,
                finished_at=FINISHED,
                checks=[
                    ModelLabPassCheck(
                        name="ready_main",
                        ok=True,
                        evidence=f"{EPHEMERAL_CLUSTER_MARKER} kind v0.30.0",
                    )
                ],
                agent_command_id=AGENT_ID,
            )

    def test_the_persistent_lane_owes_the_deploy_agent_correlation_id(self) -> None:
        """The structural discriminator, not a declared label.

        The persistent lane's apply IS the deploy agent's, as one rebuild
        correlation (OMN-18200/OMN-18573). A kind boot has no deploy agent at
        all, so it cannot produce this id — which is what makes the requirement
        something a relabelled kind receipt fails rather than merely asserts.
        """
        with pytest.raises(ValueError, match="agent_command_id"):
            build_receipt(
                sha=SHA,
                lane=EnumLabLane.ONEX_LAB,
                started_at=STARTED,
                finished_at=FINISHED,
                checks=_lane_checks(),
                agent_command_id=None,
            )

    def test_a_kind_smoke_receipt_must_declare_its_ephemeral_cluster(self) -> None:
        """AC3: the kind boot is NAMED as a smoke check in its own receipt."""
        with pytest.raises(ValueError, match=CLUSTER_PROVENANCE_CHECK):
            build_receipt(
                sha=SHA,
                lane=EnumLabLane.KIND_SMOKE,
                started_at=STARTED,
                finished_at=FINISHED,
                checks=[
                    ModelLabPassCheck(
                        name="manifests_render", ok=True, evidence="kubectl kustomize"
                    )
                ],
                agent_command_id=None,
            )

    def test_a_kind_smoke_receipt_carries_no_agent_correlation(self) -> None:
        with pytest.raises(ValueError, match="agent_command_id"):
            build_receipt(
                sha=SHA,
                lane=EnumLabLane.KIND_SMOKE,
                started_at=STARTED,
                finished_at=FINISHED,
                checks=_smoke_checks(),
                agent_command_id=AGENT_ID,
            )

    def test_a_well_formed_kind_smoke_receipt_is_accepted(self) -> None:
        """Positive control: the smoke check keeps working, it just is not a pass."""
        receipt = build_receipt(
            sha=SHA,
            lane=EnumLabLane.KIND_SMOKE,
            started_at=STARTED,
            finished_at=FINISHED,
            checks=_smoke_checks(),
            agent_command_id=None,
        )
        assert receipt.result is EnumLabPassResult.PASS
        assert artifact_name(receipt.lane, SHA) == f"lab-pass-receipt-kind-smoke-{SHA}"

    def test_a_well_formed_persistent_receipt_is_accepted(self) -> None:
        receipt = _persistent()
        assert receipt.result is EnumLabPassResult.PASS
        assert ModelLabPassReceipt.from_json(receipt.to_json()) == receipt


# ---------------------------------------------------------------------------
# The gate (AC2)
# ---------------------------------------------------------------------------
def _zip_of(body: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("receipt.json", body)
    return buffer.getvalue()


class _Surface:
    def __init__(self, bodies: dict[str, str]) -> None:
        self.bodies = bodies
        self.ids = {name: 2000 + i for i, name in enumerate(sorted(bodies))}

    def __call__(self, path: str) -> bytes:
        if "/actions/artifacts?name=" in path:
            name = path.split("name=")[1].split("&")[0]
            if name not in self.bodies:
                return json.dumps({"artifacts": []}).encode()
            return json.dumps(
                {
                    "artifacts": [
                        {
                            "id": self.ids[name],
                            "name": name,
                            "expired": False,
                            "created_at": "2026-09-19T03:51:00Z",
                        }
                    ]
                }
            ).encode()
        artifact_id = int(path.split("/artifacts/")[1].split("/")[0])
        name = next(n for n, i in self.ids.items() if i == artifact_id)
        return _zip_of(self.bodies[name])


def _gate(
    bodies: dict[str, str],
    monkeypatch: Any,
    lanes: tuple[EnumLabLane, ...] = DEFAULT_GATE_LANES,
) -> tuple[int, str]:
    monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", _Surface(bodies))
    out = io.StringIO()
    code = evaluate_gate(REPO, SHA, list(lanes), out)
    return code, out.getvalue()


class TestGateRequiresThePersistentLab:
    def test_a_kind_smoke_receipt_alone_does_not_satisfy_the_gate(
        self, monkeypatch: Any
    ) -> None:
        """The measured 2026-09-19 delivery, replayed.

        Only the kind receipt exists for this sha. Before OMN-18276 that
        returned 0 and the candidate was announced to staging.
        """
        smoke = build_receipt(
            sha=SHA,
            lane=EnumLabLane.KIND_SMOKE,
            started_at=STARTED,
            finished_at=FINISHED,
            checks=_smoke_checks(),
            agent_command_id=None,
        )
        code, output = _gate(
            {artifact_name(EnumLabLane.KIND_SMOKE, SHA): smoke.to_json()}, monkeypatch
        )
        assert code == 1
        assert SHA in output

    def test_the_persistent_receipt_satisfies_the_gate(self, monkeypatch: Any) -> None:
        receipt = _persistent()
        code, output = _gate(
            {artifact_name(EnumLabLane.ONEX_LAB, SHA): receipt.to_json()}, monkeypatch
        )
        assert code == 0
        assert "onex-lab" in output

    def test_compose_dev_positive_control(self, monkeypatch: Any) -> None:
        """A gate that NAMES compose-dev is still satisfied by its receipt.

        This is the sibling-delivery path (OMN-17057), which reads the pinned
        sibling repository's compose-dev receipt and nothing else. It must keep
        working exactly as it did.
        """
        receipt = _persistent(lane=EnumLabLane.COMPOSE_DEV)
        code, output = _gate(
            {artifact_name(EnumLabLane.COMPOSE_DEV, SHA): receipt.to_json()},
            monkeypatch,
            lanes=(EnumLabLane.COMPOSE_DEV,),
        )
        assert code == 0
        assert "compose-dev" in output

    def test_every_named_lane_must_pass(self, monkeypatch: Any) -> None:
        """Naming two lanes requires two passes, not one.

        The pre-OMN-18276 gate was satisfied by ANY named lane passing, which
        is what let the kind boot stand in for the persistent lab.
        """
        receipt = _persistent()
        code, _ = _gate(
            {artifact_name(EnumLabLane.ONEX_LAB, SHA): receipt.to_json()},
            monkeypatch,
            lanes=(EnumLabLane.ONEX_LAB, EnumLabLane.COMPOSE_DEV),
        )
        assert code == 1

    def test_the_gate_refuses_to_be_asked_about_kind_smoke(
        self, monkeypatch: Any
    ) -> None:
        """Defence in depth behind the argparse choice list.

        ``evaluate_gate`` is importable and is called directly by tests and by
        any future caller; the refusal lives with the function, not only with
        its CLI.
        """
        monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", _Surface({}))
        out = io.StringIO()
        code = evaluate_gate(REPO, SHA, [EnumLabLane.KIND_SMOKE], out)
        assert code == 1
        assert "kind-smoke" in out.getvalue()


# ---------------------------------------------------------------------------
# The wiring (AC1, AC2, AC3)
# ---------------------------------------------------------------------------
class TestEmitterWiring:
    def test_the_kind_boot_gate_emits_kind_smoke_and_never_the_lab_lane(self) -> None:
        text = DELIVER.read_text(encoding="utf-8")
        assert "--lane kind-smoke" in text
        assert "--lane onex-lab" not in text, (
            "the candidate boot gate is a kind cluster; emitting the "
            "persistent lab's lane from it is the OMN-18276 defect"
        )
        assert f"lab-pass-receipt-{EnumLabLane.KIND_SMOKE.value}-" in text

    def test_the_kind_boot_gate_declares_its_ephemeral_cluster(self) -> None:
        text = DELIVER.read_text(encoding="utf-8")
        assert CLUSTER_PROVENANCE_CHECK in text
        assert EPHEMERAL_CLUSTER_MARKER in text

    def test_the_persistent_lab_emits_the_onex_lab_lane(self) -> None:
        text = REBUILD.read_text(encoding="utf-8")
        assert "--lane onex-lab" in text
        assert "--lane onex-lab-k3s" not in text
        assert "lab-pass-receipt-onex-lab-${{" in text

    def test_the_kind_job_is_named_a_smoke_check_not_a_lab_pass(self) -> None:
        """AC3: "named as exactly that in its receipt AND its check name"."""
        workflow = yaml.safe_load(DELIVER.read_text(encoding="utf-8"))
        name = workflow["jobs"]["candidate-boot-gate"]["name"].lower()
        assert "smoke" in name
        assert "lab pass" not in name

    def test_the_dispatch_job_still_cannot_be_reached_without_the_gate(self) -> None:
        """The OMN-17530 ratchet, re-asserted because this change moved the gate.

        ``lab-pass-gate`` may now be SKIPPED (the premise job below), and a
        skipped dependency is not ``success()``, so the dispatch stays unfired.
        A skip can only close the path, never open it.
        """
        workflow = yaml.safe_load(DELIVER.read_text(encoding="utf-8"))
        assert "lab-pass-gate" in workflow["jobs"]["dispatch-to-staging"]["needs"]
        assert "if" not in workflow["jobs"]["dispatch-to-staging"]


# ---------------------------------------------------------------------------
# The wait budget (AC2 — the gate has to be satisfiable to be a gate)
# ---------------------------------------------------------------------------
BUDGET = Path("config/lab_pass_settle_budget.yaml")


class TestDeliveryGateWaitBudget:
    """Declared, observation-bounded, and DERIVED into the job's ceiling.

    The same shape OMN-18436 gave the settle budgets: the number lives in one
    declared place, observation bounds it from below, and the workflow's
    ceiling is arithmetic over it rather than a knob.
    """

    def _declared(self) -> dict[str, Any]:
        block = yaml.safe_load(BUDGET.read_text(encoding="utf-8"))["delivery_gate"]
        assert isinstance(block, dict)
        return block

    def test_the_budget_covers_the_worst_observed_wait(self) -> None:
        block = self._declared()
        worst = max(o["seconds"] for o in block["observed_wait_seconds"])
        assert block["wait_budget_seconds"] >= worst, (
            "the declared wait is below a wait that has actually been measured, "
            "so the gate would refuse a sha the lab was about to pass"
        )

    def test_every_observation_names_the_sha_it_was_read_from(self) -> None:
        for observation in self._declared()["observed_wait_seconds"]:
            assert len(observation["sha"]) == 40
            assert observation["note"].strip()

    def test_the_workflow_waits_exactly_the_declared_budget(self) -> None:
        text = DELIVER.read_text(encoding="utf-8")
        declared = self._declared()["wait_budget_seconds"]
        assert f"--wait-seconds {declared}" in text

    def test_the_job_ceiling_is_derived_from_the_declaration(self) -> None:
        block = self._declared()
        expected = (block["wait_budget_seconds"] + block["reserved_tail_seconds"]) / 60
        workflow = yaml.safe_load(DELIVER.read_text(encoding="utf-8"))
        assert workflow["jobs"]["lab-pass-gate"]["timeout-minutes"] == expected

    def test_a_present_fail_receipt_ends_the_wait_immediately(
        self, monkeypatch: Any
    ) -> None:
        """A recorded verdict does not improve with time.

        Without this the gate would spend its whole 90-minute budget on a lane
        that has already said no, and the refusal would read as a timeout.
        """
        failing = build_receipt(
            sha=SHA,
            lane=EnumLabLane.ONEX_LAB,
            started_at=STARTED,
            finished_at=FINISHED,
            checks=[
                ModelLabPassCheck(
                    name="overlay_applied", ok=False, evidence="rollout wedged"
                )
            ],
            agent_command_id=AGENT_ID,
        )
        slept: list[float] = []
        monkeypatch.setattr(
            "scripts.ci.lab_pass_receipt._gh_api",
            _Surface({artifact_name(EnumLabLane.ONEX_LAB, SHA): failing.to_json()}),
        )
        out = io.StringIO()
        code = evaluate_gate(
            REPO,
            SHA,
            list(DEFAULT_GATE_LANES),
            out,
            wait_seconds=5400,
            sleep=slept.append,
            monotonic=lambda: 0.0,
        )
        assert code == 1
        assert slept == []

    def test_the_wait_ends_as_soon_as_the_receipt_lands(self, monkeypatch: Any) -> None:
        """The receipt appears on the third read; the gate stops there."""
        receipt = _persistent()
        name = artifact_name(EnumLabLane.ONEX_LAB, SHA)
        bodies: dict[str, str] = {}
        surface = _Surface(bodies)
        surface.ids = {name: 2000}
        reads = {"n": 0}

        def counting(path: str) -> bytes:
            if "/actions/artifacts?name=" in path:
                reads["n"] += 1
                if reads["n"] >= 3:
                    bodies[name] = receipt.to_json()
            return surface(path)

        clock = {"t": 0.0}
        monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", counting)
        out = io.StringIO()
        code = evaluate_gate(
            REPO,
            SHA,
            list(DEFAULT_GATE_LANES),
            out,
            wait_seconds=5400,
            poll_interval_seconds=60,
            sleep=lambda s: clock.__setitem__("t", clock["t"] + s),
            monotonic=lambda: clock["t"],
        )
        assert code == 0
        assert clock["t"] == 120.0

    def test_an_expired_wait_is_a_refusal_naming_both_causes(
        self, monkeypatch: Any
    ) -> None:
        clock = {"t": 0.0}
        monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", _Surface({}))
        out = io.StringIO()
        code = evaluate_gate(
            REPO,
            SHA,
            list(DEFAULT_GATE_LANES),
            out,
            wait_seconds=180,
            poll_interval_seconds=60,
            sleep=lambda s: clock.__setitem__("t", clock["t"] + s),
            monotonic=lambda: clock["t"],
        )
        assert code == 1
        text = out.getvalue()
        assert "still running" in text
        assert "never asked" in text
        assert SHA in text

    def test_there_is_still_no_override_flag(self) -> None:
        """OMN-17530's ratchet, re-asserted over the parser this change edited."""
        parser = build_parser()
        actions = {
            option
            for action in parser._subparsers._group_actions
            for choice in getattr(action, "choices", {}).values()
            for option in choice._option_string_actions
        }
        for banned in ("--force", "--skip", "--allow-missing", "--warn-only"):
            assert banned not in actions


class TestPremiseJob:
    """The premise can only CLOSE the path to staging (OMN-18276)."""

    def _jobs(self) -> dict[str, Any]:
        return yaml.safe_load(DELIVER.read_text(encoding="utf-8"))["jobs"]

    def test_the_gate_depends_on_the_resolved_premise(self) -> None:
        assert "resolve-lab-premise" in self._jobs()["lab-pass-gate"]["needs"]

    def test_the_premise_uses_the_canonical_classifier_not_a_second_copy(
        self,
    ) -> None:
        text = DELIVER.read_text(encoding="utf-8")
        assert "validate_pr_deploy_required.py" in text
        assert "load_runtime_path_classifier" in text

    def test_the_premise_fails_closed(self) -> None:
        """An unreadable diff or an unloadable classifier REQUIRES the receipt."""
        text = DELIVER.read_text(encoding="utf-8")
        assert "verdict = True" in text
        assert "requiring the lab receipt" in text

    def test_the_premise_job_has_no_write_permission_anywhere(self) -> None:
        assert self._jobs()["resolve-lab-premise"]["permissions"] == {
            "contents": "read"
        }
