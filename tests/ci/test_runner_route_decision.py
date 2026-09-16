# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the per-run runner routing decision (OMN-18031).

WHAT THIS GUARDS. ``scripts/ci/runner_route_decision.py`` chooses, per CI run,
between the lab fleet labels and GitHub-hosted, from live org-runner capacity
and lab load. The trusted seam variable is read as a CEILING and never written
(CLAUDE.md rule 14, single-owner). Every step of the decision can only move the
answer TOWARD hosted, so the module can never place a job on self-hosted compute
that the seam did not already allow.

The load-bearing invariant is ``test_never_widens_beyond_ceiling``: a
cross-product sweep over every input dimension asserting a self-hosted label can
appear in the result only when the seam already contained it. Fork isolation
(OMN-16683/16684) lives INSIDE this module rather than at the 46 call sites in
ci.yml, precisely so it cannot be edited wrong in one of them.

Fail-closed is the whole safety story: an unreadable probe, a malformed payload,
a stale lab record and an outright exception are all hosted, and nothing raises
to the caller -- a crashing route job still hands the run a usable ``runs-on``.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "ci" / "runner_route_decision.py"

sys.path.insert(0, str(REPO_ROOT / "src"))

_spec = importlib.util.spec_from_file_location("runner_route_decision", MODULE_PATH)
assert _spec is not None and _spec.loader is not None
route = importlib.util.module_from_spec(_spec)
sys.modules["runner_route_decision"] = route
_spec.loader.exec_module(route)

# THE DECISION LIVES IN THE NODE. `route` above is the I/O boundary -- probes,
# request assembly, CLI -- and is loaded for the probe tests. Everything that
# decides a placement is the handler below, declared by the node's contract.
from omnibase_infra.nodes.node_ci_runner_route_compute.handlers.handler_ci_runner_route import (
    HandlerCIRunnerRoute,
)
from omnibase_infra.nodes.node_ci_runner_route_compute.models.enum_ci_runner_route_force import (
    EnumCIRunnerRouteForce,
)
from omnibase_infra.nodes.node_ci_runner_route_compute.models.enum_ci_runner_route_reason import (
    EnumCIRunnerRouteReason,
)
from omnibase_infra.nodes.node_ci_runner_route_compute.models.enum_ci_runner_route_visibility import (
    EnumCIRunnerRouteVisibility,
)
from omnibase_infra.nodes.node_ci_runner_route_compute.models.model_ci_runner_route_decision import (
    ModelCIRunnerRouteDecision,
)
from omnibase_infra.nodes.node_ci_runner_route_compute.models.model_ci_runner_route_evidence import (
    ModelCIRunnerRouteEvidence,
)
from omnibase_infra.nodes.node_ci_runner_route_compute.models.model_ci_runner_route_policy import (
    ModelCIRunnerRoutePolicy,
)
from omnibase_infra.nodes.node_ci_runner_route_compute.models.model_ci_runner_route_request import (
    ModelCIRunnerRouteRequest,
)

CONTRACT = (
    REPO_ROOT / "src/omnibase_infra/nodes/node_ci_runner_route_compute/contract.yaml"
)
VIS_PUBLIC = EnumCIRunnerRouteVisibility.PUBLIC
VIS_PRIVATE = EnumCIRunnerRouteVisibility.PRIVATE
VIS_UNKNOWN = EnumCIRunnerRouteVisibility.UNKNOWN
FLEET_EXPECTED = 60


LAB_SEAM = '["self-hosted","omnibase-ci"]'
HOSTED_SEAM = '["ubuntu-latest"]'
HOSTED_LABELS = ["ubuntu-latest"]
LAB_LABELS = ["self-hosted", "omnibase-ci"]


def _policy(**overrides: Any) -> dict[str, Any]:
    """A complete, valid ``route:`` policy section.

    Every key is spelled out because the loader must FAIL on a missing key
    rather than apply a default (CLAUDE.md rule 8) -- see
    ``test_missing_policy_key_raises_and_applies_no_default``.
    """
    base: dict[str, Any] = {
        "policy_version": 1,
        "runner_group": "omnibase-ci",
        "lab_labels": ["self-hosted", "omnibase-ci"],
        "hosted_labels": ["ubuntu-latest"],
        "min_idle_runners": 12,
        "max_busy_fraction": 0.80,
        "min_online_fraction": 0.67,
        "lab_record_max_age_seconds": 600,
        "max_lab_load_ratio": 1.0,
        "min_lab_free_mem_mib": 4096,
        "capacity_downgrade_reasons": [
            "fleet_saturated",
            "fleet_degraded",
            "lab_unknown",
            "lab_saturated",
            "probe_error",
        ],
        "private_repo_hosted_placement": "refuse",
    }
    base.update(overrides)
    return base


def _fleet(online: int = 88, busy: int = 10) -> dict[str, Any]:
    """A well-formed fleet probe result: ``online`` runners, ``busy`` of them busy."""
    return {"ok": True, "online": online, "busy": busy, "total": online}


def _lab(
    age_seconds: int = 30, ratio: float = 0.19, free_mem_mib: int = 40000
) -> dict[str, Any]:
    """A well-formed lab-load record, ``age_seconds`` old, one host."""
    return {
        "ok": True,
        "age_seconds": age_seconds,
        "hosts": [{"label": "h201", "ratio": ratio, "free_mem_mib": free_mem_mib}],
    }


class _Result:
    """A typed decision, read the way a behaviour test asks about one.

    The assertions below are about PLACEMENT -- which labels, which reason,
    which evidence -- and they are unchanged by the decision moving into the
    node. This adapter is what keeps them unchanged: it exposes the wire forms
    the calling workflow and the monitor actually consume (`labels`, the
    rendered `reason`) rather than restating the model's field names in two
    hundred assertions.
    """

    def __init__(self, decision: ModelCIRunnerRouteDecision) -> None:
        self.decision_model = decision
        self.labels = list(decision.runs_on)
        self.decision = decision.decision.value
        self.reason = decision.reason_wire
        evidence = decision.evidence
        self.inputs: dict[str, Any] = {
            "event_name": evidence.github_event,
            "repository": evidence.repository,
            "workflow_path": evidence.workflow_path,
            "seam_json": evidence.seam_json,
            "visibility": evidence.visibility.value,
            "idle": evidence.idle,
            "busy_fraction": evidence.busy_fraction,
            "lab_error": evidence.lab_error,
            "lab_host": evidence.lab_host,
            "downgrade_refused_from": evidence.downgrade_refused_from,
        }
        if evidence.violation:
            self.inputs["violation"] = evidence.violation

    def to_record(self) -> dict[str, Any]:
        return self.decision_model.model_dump(mode="json")


def _request(**overrides: Any) -> ModelCIRunnerRouteRequest:
    """Assemble the node's typed request from loose test inputs."""
    kwargs: dict[str, Any] = {
        "event_name": "push",
        "head_repo": "OmniNode-ai/omnibase_infra",
        "repository": "OmniNode-ai/omnibase_infra",
        "workflow_path": ".github/workflows/ci.yml",
        "seam_json": LAB_SEAM,
        "public_json": HOSTED_SEAM,
        "fleet": _fleet(),
        "lab": _lab(),
        "policy": _policy(),
        "allowlist": [],
        # omnibase_infra is a PUBLIC repository, so this is what its own runs
        # carry. It is spelled out rather than defaulted: a defaulted
        # visibility would make the private-repository rule fail OPEN for any
        # caller that forgot to wire it.
        "visibility": VIS_PUBLIC,
        "force": EnumCIRunnerRouteForce.AUTO,
        "fleet_expected_count": FLEET_EXPECTED,
    }
    kwargs.update(overrides)
    policy = kwargs.pop("policy")
    if isinstance(policy, dict):
        policy = ModelCIRunnerRoutePolicy.model_validate(policy)
    fleet = route.build_request(
        event_name=kwargs["event_name"],
        head_repo=kwargs["head_repo"] or "",
        repository=kwargs["repository"],
        workflow_path=kwargs["workflow_path"],
        seam_json=kwargs["seam_json"] or "",
        public_json=kwargs["public_json"] or "",
        visibility=str(kwargs["visibility"]),
        fleet=(
            kwargs["fleet"]
            if isinstance(kwargs["fleet"], dict)
            else {"ok": False, "error": "unexpected_shape"}
        ),
        lab=(
            kwargs["lab"]
            if isinstance(kwargs["lab"], dict)
            else {"ok": False, "error": "unexpected_shape"}
        ),
        hosted_workflows=tuple(kwargs["allowlist"] or ()),
        force=str(kwargs["force"]),
        policy=policy,
        fleet_expected_count=kwargs["fleet_expected_count"],
    )
    return fleet


def _decide(**overrides: Any) -> Any:
    return _Result(HandlerCIRunnerRoute().handle(_request(**overrides)))


# --- 1. the happy path: capacity is available and the seam permits the lab ---


def test_capacity_available_routes_to_the_lab() -> None:
    """Seam allows the lab, 78/88 idle, lab at 0.19x -> lab labels."""
    result = _decide(fleet=_fleet(online=88, busy=10), lab=_lab(ratio=0.19))
    assert result.labels == LAB_LABELS
    assert result.decision == "self_hosted"
    assert result.reason == "capacity_available"


# --- 2/3. fleet saturation, on either of its two independent thresholds ------


def test_too_few_idle_runners_falls_back_to_hosted() -> None:
    """idle 3 < min_idle_runners 12 -> hosted. Headroom floor, not a capacity match."""
    result = _decide(fleet=_fleet(online=88, busy=85))
    assert result.labels == HOSTED_LABELS
    assert result.reason == "fleet_saturated"


def test_busy_fraction_at_or_above_the_ceiling_falls_back_to_hosted() -> None:
    """busy 80/88 = 0.909 >= max_busy_fraction 0.80 -> hosted."""
    result = _decide(
        fleet=_fleet(online=88, busy=80), policy=_policy(min_idle_runners=1)
    )
    assert result.labels == HOSTED_LABELS
    assert result.reason == "fleet_saturated"


def test_a_degraded_fleet_falls_back_to_hosted() -> None:
    """online 40 < min_online_runners 60 -> hosted, a trust floor below the canary's own."""
    result = _decide(fleet=_fleet(online=40, busy=0))
    assert result.labels == HOSTED_LABELS
    assert result.reason == "fleet_degraded"


# --- 4. every probe failure mode is hosted, and NOTHING escapes -------------


@pytest.mark.parametrize(
    ("fleet", "expected_class"),
    [
        ({"ok": False, "error": "http_403"}, "probe_error:http_403"),
        ({"ok": False, "error": "timeout"}, "probe_error:timeout"),
        ({"ok": False, "error": "malformed_json"}, "probe_error:malformed_json"),
        ({"ok": False, "error": "missing_token"}, "probe_error:missing_token"),
        (
            {"ok": True, "online": "not-a-number", "busy": 0},
            "probe_error:unexpected_shape",
        ),
        ({"ok": True}, "probe_error:unexpected_shape"),
        (None, "probe_error:unexpected_shape"),
    ],
)
def test_any_probe_error_falls_back_to_hosted(fleet: Any, expected_class: str) -> None:
    """A probe that cannot prove capacity is hosted -- never 'assume ample'."""
    result = _decide(fleet=fleet)
    assert result.labels == HOSTED_LABELS
    assert result.reason == expected_class


def test_an_unusable_observation_still_yields_hosted() -> None:
    """FAIL-CLOSED, at the seam it now lives on.

    The decision is a pure function of a TYPED request, so a malformed
    observation can no longer reach it as a surprise: the boundary that builds
    the request turns anything it cannot understand into a NAMED failure class,
    and the node resolves every one of those to hosted. The failure that used
    to be an exception is now a value, and this pins that it still cannot
    produce a fleet placement.
    """
    for broken in ({"ok": True, "online": "many", "busy": None}, {}, {"ok": False}):
        result = _decide(fleet=broken, seam_json=LAB_SEAM)
        assert result.labels == HOSTED_LABELS
        assert result.reason.startswith("probe_error")


# --- 5. fork isolation is inviolable and lives inside the module ------------


@pytest.mark.parametrize("event_name", ["pull_request", "pull_request_target"])
def test_a_fork_event_is_hosted_even_with_a_completely_idle_fleet(
    event_name: str,
) -> None:
    """OMN-16683/16684: untrusted code NEVER reaches self-hosted, at any capacity."""
    result = _decide(
        event_name=event_name,
        head_repo="somebody-else/omnibase_infra",
        fleet=_fleet(online=88, busy=0),
        seam_json=LAB_SEAM,
    )
    assert result.labels == HOSTED_LABELS
    assert result.reason == "fork_isolation"


def test_a_same_repo_pull_request_is_not_treated_as_a_fork() -> None:
    """The trusted seam still applies to a branch PR from the repo itself."""
    result = _decide(event_name="pull_request", head_repo="OmniNode-ai/omnibase_infra")
    assert result.labels == LAB_LABELS
    assert result.reason == "capacity_available"


# --- 6. the seam is a CEILING: today's live state is inert -----------------


def test_a_hosted_seam_is_returned_verbatim_even_with_a_fully_idle_fleet() -> None:
    """TODAY'S LIVE STATE. OMNI_TRUSTED_CI_RUNS_ON_JSON is '["ubuntu-latest"]',
    so the mechanism must be byte-identically inert: hosted on every run,
    regardless of how much lab capacity exists. This is the test that proves
    landing this ticket changes no placement until a separate single-owner
    claim flips the seam (that is OMN-16682, not this ticket).
    """
    result = _decide(seam_json=HOSTED_SEAM, fleet=_fleet(online=88, busy=0))
    assert result.labels == HOSTED_LABELS
    assert result.reason == "seam_ceiling_hosted"


def test_a_hosted_seam_is_returned_verbatim_not_normalised() -> None:
    """A seam naming a non-default hosted image is passed through untouched."""
    result = _decide(seam_json='["ubuntu-24.04","x64"]')
    assert result.labels == ["ubuntu-24.04", "x64"]
    assert result.reason == "seam_ceiling_hosted"


# --- 7/8. the lab half: unknown and saturated are both hosted --------------


def test_a_stale_lab_record_is_hosted() -> None:
    """age 900s > lab_record_max_age_seconds 600 -> hosted. Stale is never 'fine'."""
    result = _decide(lab=_lab(age_seconds=900))
    assert result.labels == HOSTED_LABELS
    assert result.reason == "lab_unknown"


def test_a_missing_lab_record_is_hosted() -> None:
    """No record at all is 'unknown', which fails closed exactly like stale."""
    result = _decide(lab={"ok": False, "error": "no_record"})
    assert result.labels == HOSTED_LABELS
    assert result.reason == "lab_unknown"


def test_a_loaded_lab_host_is_hosted() -> None:
    """ratio 1.9 > max_lab_load_ratio 1.0 -> hosted.

    The threshold is COPIED from the pre-push picker's committed
    PREPUSH_LOAD_THRESHOLD default so the lab has one definition of
    'too loaded' shared by pre-push placement and CI routing.
    """
    result = _decide(lab=_lab(ratio=1.9))
    assert result.labels == HOSTED_LABELS
    assert result.reason == "lab_saturated"


def test_a_memory_starved_lab_host_is_hosted_even_at_zero_load() -> None:
    """OMN-17392's lesson, carried over: load ranks, memory ADMITS."""
    result = _decide(lab=_lab(ratio=0.01, free_mem_mib=512))
    assert result.labels == HOSTED_LABELS
    assert result.reason == "lab_saturated"


# --- 9. THE load-bearing invariant -----------------------------------------


def test_never_widens_beyond_ceiling() -> None:
    """Cross-product sweep: a self-hosted label appears ONLY when the seam had one.

    This is the mechanical form of the never-widen rule. It is a sweep rather
    than a review convention because widening is the one failure this design
    cannot tolerate, and a reviewer cannot check 2000 combinations.
    """
    seams = [LAB_SEAM, HOSTED_SEAM, '["ubuntu-24.04"]', '["self-hosted","omnibase-ci"]']
    events = ["push", "pull_request", "pull_request_target", "merge_group", "schedule"]
    head_repos = ["OmniNode-ai/omnibase_infra", "fork/omnibase_infra"]
    fleets = [
        _fleet(88, 0),
        _fleet(88, 87),
        _fleet(0, 0),
        {"ok": False, "error": "timeout"},
        None,
    ]
    labs = [
        _lab(),
        _lab(age_seconds=99999),
        _lab(ratio=9.9),
        {"ok": False, "error": "x"},
    ]
    paths = [".github/workflows/ci.yml", ".github/workflows/anything.yml"]
    # OMN-18412: the visibility rule can REVERSE a hosted answer back onto the
    # fleet, which is the one kind of edit that could widen. It is swept here
    # rather than tested only in isolation for exactly that reason.
    visibilities = [
        VIS_PUBLIC,
        VIS_PRIVATE,
        VIS_UNKNOWN,
    ]

    checked = 0
    for seam, event, head, fleet, lab, path, visibility in itertools.product(
        seams, events, head_repos, fleets, labs, paths, visibilities
    ):
        result = _decide(
            seam_json=seam,
            event_name=event,
            head_repo=head,
            fleet=fleet,
            lab=lab,
            workflow_path=path,
            visibility=visibility,
        )
        seam_labels = set(json.loads(seam))
        returned = set(result.labels)
        if result.decision == "blocked":
            # A refusal names no labels at all. That is neither hosted nor a
            # widening; it is the third outcome OMN-18412 added.
            assert returned == set()
            checked += 1
            continue
        # Either exactly the hosted set, or a subset of what the seam allowed.
        assert returned == set(HOSTED_LABELS) or returned <= seam_labels, (
            f"WIDENED: seam={seam} event={event} head={head} -> {result.labels}"
        )
        if "self-hosted" in returned:
            assert "self-hosted" in seam_labels, (
                f"self-hosted invented from seam={seam}"
            )
        checked += 1
    # Non-vacuity: the sweep must actually have run the combinations.
    assert checked == len(seams) * len(events) * len(head_repos) * len(fleets) * len(
        labs
    ) * len(paths) * len(visibilities)
    assert checked > 1500


def test_the_sweep_is_not_vacuous_because_some_inputs_do_reach_the_lab() -> None:
    """Positive control for the sweep above: a sweep that never returns lab
    labels would pass ``test_never_widens_beyond_ceiling`` trivially. At least
    one combination must actually route to self-hosted, or the invariant is
    proving nothing.
    """
    result = _decide(
        seam_json=LAB_SEAM, event_name="push", fleet=_fleet(88, 0), lab=_lab()
    )
    assert "self-hosted" in result.labels


# --- 10. policy is fail-fast, never defaulted ------------------------------


@pytest.mark.parametrize(
    "missing_key",
    [
        "min_idle_runners",
        "max_busy_fraction",
        "min_online_fraction",
        "lab_record_max_age_seconds",
        "max_lab_load_ratio",
        "min_lab_free_mem_mib",
        "lab_labels",
        "hosted_labels",
        "policy_version",
        "runner_group",
    ],
)
def test_missing_policy_key_raises_and_applies_no_default(
    missing_key: str, tmp_path: Path
) -> None:
    """CLAUDE.md rule 8: a missing threshold raises at load. A silently-defaulted
    threshold is how a routing gate quietly stops gating.
    """
    section = _policy()
    del section[missing_key]
    contract = tmp_path / "contract.yaml"
    contract.write_text(json.dumps({"config": section}), encoding="utf-8")
    with pytest.raises(ValidationError) as excinfo:
        route.load_contract_policy(contract)
    assert missing_key in str(excinfo.value)


def test_a_contract_with_no_config_block_raises(tmp_path: Path) -> None:
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        json.dumps({"name": "node_ci_runner_route_compute"}), encoding="utf-8"
    )
    with pytest.raises(KeyError):
        route.load_contract_policy(contract)


def test_the_committed_contract_loads_and_carries_every_threshold() -> None:
    """The real contract must satisfy the same typed loader.

    This is the binding the operator direction asks for: the thresholds are
    DECLARED in the node's contract, and the test reads them from there rather
    than from a config file, an environment variable or a literal.
    """
    loaded = route.load_contract_policy(CONTRACT)
    assert loaded.runner_group == "omnibase-ci"
    assert loaded.hosted_labels == ("ubuntu-latest",)
    assert loaded.lab_labels == ("self-hosted", "omnibase-ci")
    assert loaded.max_lab_load_ratio == 1.0
    assert loaded.min_lab_free_mem_mib == 4096
    assert loaded.private_repo_hosted_placement == "refuse"


def test_the_routing_policy_file_no_longer_restates_a_threshold() -> None:
    """One home, not two.

    A threshold left behind in the old config file is a second policy that
    agrees with the contract until the day someone edits one of them, and the
    copy nobody reads is the one that would have been edited.
    """
    import yaml as _yaml

    section = _yaml.safe_load(
        (REPO_ROOT / "config" / "runner_routing_policy.yaml").read_text(
            encoding="utf-8"
        )
    )["route"]
    for moved in (
        "min_idle_runners",
        "max_busy_fraction",
        "min_online_fraction",
        "min_online_fraction",
        "lab_labels",
        "hosted_labels",
        "max_lab_load_ratio",
        "runner_group",
    ):
        assert moved not in section, f"{moved} is declared in the contract now"
    # The per-repository data that legitimately stays.
    assert "hosted_workflows" in section
    assert "saturation_alert" in section


def test_the_degraded_floor_is_read_from_the_fleet_inventory() -> None:
    """The floor tracks the fleet, and the fleet is declared in one place.

    Measured 2026-09-15: the fleet was capped to 60 runners while the old
    literal floor was also 60, which would have refused the fleet on every run
    the moment one runner went offline -- silently, as an ordinary return to
    GitHub-hosted. Read as a fraction of the declared inventory, the same two
    thirds moves with the cap.
    """
    expected = route.load_fleet_expected_count(
        REPO_ROOT / "config" / "runner_fleet.yaml"
    )
    assert expected > 0
    policy = route.load_contract_policy(CONTRACT)
    floor = math.ceil(expected * policy.min_online_fraction)
    assert floor < expected, (
        "the degraded floor must leave room for a runner to go offline; at "
        "floor == inventory the fleet is refused on every blip"
    )
    at_floor = _decide(
        fleet=_fleet(online=floor, busy=0),
        fleet_expected_count=expected,
        policy=_policy(min_idle_runners=1),
    )
    assert at_floor.reason == "capacity_available"
    below = _decide(
        fleet=_fleet(online=floor - 1, busy=0),
        fleet_expected_count=expected,
        policy=_policy(min_idle_runners=1),
    )
    assert below.reason == "fleet_degraded"


def test_the_hosted_workflow_list_is_per_repository_data_and_still_loads() -> None:
    """The exceptions stay in the repository's own config file, and one of them
    is load-bearing.

    THE REGRESSION A LIVE DRY RUN CAUGHT: ci.yml is in hosted_runner_allowlist
    (for its one bare-hosted summary job) but must NOT be in the route hosted
    list, or all 54 of its jobs are pinned hosted forever and the mechanism is a
    silent no-op on its largest consumer.
    """
    hosted = route.load_hosted_workflows(
        REPO_ROOT / "config" / "runner_routing_policy.yaml"
    )
    assert ".github/workflows/ci.yml" not in hosted
    assert ".github/workflows/runner-route-reusable.yml" in hosted


# --- 11. the hosted allowlist wins over every other input ------------------


def test_an_allowlisted_workflow_is_hosted_regardless_of_everything_else() -> None:
    """A workflow the policy pins hosted stays hosted even on an idle fleet."""
    result = _decide(
        workflow_path=".github/workflows/runner-fleet-canary.yml",
        allowlist=[".github/workflows/runner-fleet-canary.yml"],
        seam_json=LAB_SEAM,
        fleet=_fleet(88, 0),
    )
    assert result.labels == HOSTED_LABELS
    assert result.reason == "policy_allowlist"


# --- the emitted record is complete enough to audit after the fact ---------


def test_the_decision_record_carries_its_evidence_and_is_json_serialisable() -> None:
    """The artifact is the evidence surface, not the run log.

    Every field a later reader needs is DECLARED, so a record that omits one
    fails to construct rather than reading as a decision nobody can audit.
    """
    result = _decide(fleet=_fleet(online=88, busy=85))
    payload = json.loads(json.dumps(result.to_record()))
    assert payload["runs_on"] == HOSTED_LABELS
    assert payload["reason"] == "fleet_saturated"
    assert payload["policy_version"] == 1
    assert payload["evidence"]["fleet_online"] == 88
    assert payload["evidence"]["fleet_busy"] == 85
    assert payload["evidence"]["idle"] == 3
    assert payload["evidence"]["visibility"] == "public"
    assert payload["decided_at"]


# --- 12. the runtime never-widen guard checks the ANSWER, not the ceiling -----


def test_runtime_guard_degrades_a_widened_answer_to_hosted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A widening return from the elimination is caught on the way out.

    WHY THIS TEST EXISTS. The first build called
    ``assert_never_widens(ceiling, ceiling, hosted)`` from inside the S7 branch
    and both the module docstring and the PR body advertised that as the
    mechanical runtime guard. It compares the ceiling with itself, so it can
    never fail. Proven by injecting a widening bug into S7 and re-running: the
    runtime check passed and the widened label set was emitted unchanged; only
    the cross-product sweep above caught it. A guard that cannot fail is
    documentation, not enforcement (CLAUDE.md rule 5).

    ``decide`` now validates the labels ``_decide_unchecked`` actually returned
    against that run's own ceiling. This monkeypatch stands in for the future
    edit to one of the eight return points that the sweep would catch in CI but
    that nothing would catch at RUNTIME, which is where an untrusted or
    unbudgeted job would actually reach the fleet.
    """
    widened = ModelCIRunnerRouteDecision(
        runs_on=("self-hosted", "omnibase-ci", "invented"),
        decision="self_hosted",
        reason=EnumCIRunnerRouteReason.CAPACITY_AVAILABLE,
        policy_version=1,
        decided_at="2026-09-15T00:00:00+00:00",
        evidence=ModelCIRunnerRouteEvidence(
            github_event="push",
            repository="OmniNode-ai/omnibase_infra",
            workflow_path=".github/workflows/ci.yml",
            seam_json=HOSTED_SEAM,
            visibility=VIS_PUBLIC,
        ),
    )
    handler = HandlerCIRunnerRoute()
    monkeypatch.setattr(handler, "_eliminate", lambda *_a, **_k: widened)

    result = _Result(handler.handle(_request(seam_json=HOSTED_SEAM)))

    assert result.labels == HOSTED_LABELS
    assert result.decision == "hosted"
    assert result.reason == "never_widen_violation"
    assert "violation" in result.inputs


def test_runtime_guard_passes_a_legitimate_lab_answer_through() -> None:
    """Positive control for the guard: it must not reject a valid decision.

    A guard that rejected everything would satisfy the test above while making
    the mechanism permanently hosted -- indistinguishable, from a green run,
    from a guard that works.
    """
    result = _decide(fleet=_fleet(online=88, busy=10), lab=_lab(ratio=0.19))
    assert result.labels == LAB_LABELS
    assert result.reason == "capacity_available"


def test_decide_and_the_elimination_are_separate_callables() -> None:
    """Pins the structure the guard depends on.

    Re-collapsing ``decide`` back into the elimination would silently restore
    the tautology: there would be no 'on the way out' left to check from.
    """
    handler = HandlerCIRunnerRoute()
    assert handler.handle is not handler._eliminate
    assert callable(handler._eliminate)
    assert callable(handler._guard_never_widens)
    assert callable(handler._guard_visibility)


# --- 13. OMN-18031 follow-up (2026-09-12): the lab-load-probe defects -------
#
# Three measured problems on the self-hosted `lab-load-probe` job: (1) a
# module-scope `import yaml` made merely IMPORTING this module raise
# `ModuleNotFoundError` on the fleet image's bare python3 (10 of the last 12
# scheduled runs), even though nothing the probe calls touches YAML; (2) the
# probe read `os.getloadavg()` from inside the runner CONTAINER, which does
# not describe the `.201` HOST `max_lab_load_ratio` was calibrated against
# (measured: 1.524x from an on-host dry run, 0.2232x from inside the same
# host's own runner container, same window); (3) a zero-byte/corrupt artifact
# must resolve to a NAMED cause, not the same bare `lab_unknown` a
# genuinely-missing record also produces.


def _run_with_site_packages_hidden(snippet: str) -> subprocess.CompletedProcess[str]:
    """Run ``snippet`` under ``python -S`` (skip ``site``, so nothing in
    site-packages -- including this venv's own PyYAML -- is importable).

    Reproduces the shape of interpreter the fleet runner's bare `python3 -`
    invocation has: only stdlib modules survive `-S`, and PyYAML is not
    stdlib.
    """
    return subprocess.run(
        [sys.executable, "-S", "-c", snippet],
        check=False,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_probe_local_lab_load_does_not_require_pyyaml() -> None:
    """RED before the fix: importing the module at all pulled in PyYAML at
    module scope, so ``from runner_route_decision import probe_local_lab_load``
    raised ``ModuleNotFoundError: No module named 'yaml'`` on the self-hosted
    fleet image's bare system python3, which has no PyYAML installed.
    """
    result = _run_with_site_packages_hidden(
        "import sys; sys.path.insert(0, 'scripts/ci'); "
        "from runner_route_decision import probe_local_lab_load; "
        "print(probe_local_lab_load()['ok'])"
    )
    assert result.returncode == 0, result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert result.stdout.strip() == "True"


def test_probe_lab_saturation_from_fleet_also_does_not_require_pyyaml() -> None:
    """Same reproduction for the function the lab-load-probe job now calls."""
    result = _run_with_site_packages_hidden(
        "import sys; sys.path.insert(0, 'scripts/ci'); "
        "from runner_route_decision import probe_lab_saturation_from_fleet; "
        "print(probe_lab_saturation_from_fleet(None, 'omnibase-ci', "
        "'https://api.github.com')['error'])"
    )
    assert result.returncode == 0, result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert result.stdout.strip() == "missing_token"


def test_load_contract_policy_still_needs_pyyaml_when_actually_called() -> None:
    """Positive control for the two tests above: proves ``-S`` really does
    remove PyYAML from view, so their passing is the lazy-import fix and not
    an accident of the interpreter already lacking PyYAML for some other
    reason. ``load_route_policy`` is the one function still allowed to need
    it, and only when called.
    """
    result = _run_with_site_packages_hidden(
        "import sys; sys.path.insert(0, 'scripts/ci'); "
        "from runner_route_decision import load_contract_policy; "
        "from pathlib import Path; "
        "load_contract_policy(Path('config/runner_routing_policy.yaml'))"
    )
    assert result.returncode != 0
    assert "ModuleNotFoundError: No module named 'yaml'" in result.stderr


def test_a_missing_lab_record_names_the_cause_as_lab_error() -> None:
    """The ``lab_unknown`` reason stays the stable enum consumers key off,
    but the underlying cause now rides along in ``inputs["lab_error"]`` -- a
    zero-byte/corrupt artifact reads as a NAMED, distinguishable state rather
    than the same bare "unknown" a genuinely-missing record also produces.
    """
    result = _decide(lab={"ok": False, "error": "no_record"})
    assert result.reason == "lab_unknown"
    assert result.inputs["lab_error"] == "no_record"


def test_an_unreadable_lab_record_names_the_cause_distinctly() -> None:
    """The zero-byte/corrupt-artifact case specifically -- the failure mode a
    dead or crashing lab-load-probe actually produces.
    """
    result = _decide(lab={"ok": False, "error": "unreadable_record"})
    assert result.reason == "lab_unknown"
    assert result.inputs["lab_error"] == "unreadable_record"


def test_a_stale_lab_record_names_the_cause_as_stale() -> None:
    result = _decide(lab=_lab(age_seconds=900))
    assert result.reason == "lab_unknown"
    assert result.inputs["lab_error"] == "stale"


def test_read_lab_record_on_a_zero_byte_file_is_unreadable_not_silently_ok(
    tmp_path: Path,
) -> None:
    """The literal artifact shape a crashed probe publishes: the file EXISTS
    (``upload-artifact``'s ``if-no-files-found: warn`` accepts an empty file)
    but has no content. This must resolve to a distinct error, never to a
    healthy-looking ``{"ok": True, ...}`` that would let S6 silently read it
    as "no load".
    """
    zero_byte = tmp_path / "lab-load.json"
    zero_byte.write_text("", encoding="utf-8")
    assert route.read_lab_record(zero_byte) == {
        "ok": False,
        "error": "unreadable_record",
    }


def test_probe_lab_saturation_from_fleet_missing_token_is_fail_closed() -> None:
    result = route.probe_lab_saturation_from_fleet(
        None, "omnibase-ci", "https://api.github.com"
    )
    assert result == {"ok": False, "error": "missing_token"}


def test_probe_lab_saturation_from_fleet_uses_the_org_busy_idle_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of the fix: ``ratio`` now comes from the org runner
    registry's busy/idle counts, not this container's own
    ``load1 / os.cpu_count()``.
    """
    monkeypatch.setattr(
        route,
        "probe_fleet",
        lambda token, runner_group, api_url: {
            "ok": True,
            "online": 88,
            "busy": 22,
            "total": 88,
        },
    )
    monkeypatch.setattr(route, "_free_mem_mib", lambda: 60000)
    result = route.probe_lab_saturation_from_fleet(
        "tok", "omnibase-ci", "https://api.github.com"
    )
    assert result["ok"] is True
    assert result["hosts"] == [
        {"label": "org:omnibase-ci", "ratio": 0.25, "free_mem_mib": 60000}
    ]


def test_probe_lab_saturation_from_fleet_propagates_a_fleet_probe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        route,
        "probe_fleet",
        lambda token, runner_group, api_url: {"ok": False, "error": "http_403"},
    )
    result = route.probe_lab_saturation_from_fleet(
        "tok", "omnibase-ci", "https://api.github.com"
    )
    assert result == {"ok": False, "error": "http_403"}


def test_probe_lab_saturation_from_fleet_treats_zero_online_as_fully_saturated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Matches the same convention ``_decide_unchecked`` uses for
    ``busy_fraction`` when ``online`` is 0: never divide by zero into a false
    "0 load".
    """
    monkeypatch.setattr(
        route,
        "probe_fleet",
        lambda token, runner_group, api_url: {
            "ok": True,
            "online": 0,
            "busy": 0,
            "total": 0,
        },
    )
    monkeypatch.setattr(route, "_free_mem_mib", lambda: 1000)
    result = route.probe_lab_saturation_from_fleet(
        "tok", "omnibase-ci", "https://api.github.com"
    )
    assert result["hosts"][0]["ratio"] == 1.0


# --- OMN-18412. A private repository is never placed on a hosted runner -----
#
# Operator ruling, 2026-09-14, firm. These are the tests the module did not have
# when that ruling landed: every one of them passes trivially on a public repo,
# and the first four fail against the pre-OMN-18412 module.


def test_a_private_repo_is_not_downgraded_to_hosted_by_saturation() -> None:
    """The load-bearing case. Fleet saturated, ceiling names the fleet, repo
    private -> the FLEET, not hosted. The fleet being busy is not a reason to
    break the ruling; the job waits for a runner instead.
    """
    result = _decide(
        visibility=VIS_PRIVATE,
        seam_json=LAB_SEAM,
        fleet=_fleet(online=88, busy=85),
    )
    assert result.labels == LAB_LABELS
    assert result.decision == "self_hosted"
    assert result.reason == "private_repo_no_hosted_downgrade"
    assert result.inputs["downgrade_refused_from"] == "fleet_saturated"


@pytest.mark.parametrize(
    ("fleet", "lab", "expected_from"),
    [
        (_fleet(88, 85), _lab(), "fleet_saturated"),
        (_fleet(30, 0), _lab(), "fleet_degraded"),
        (_fleet(88, 10), _lab(age_seconds=99999), "lab_unknown"),
        (_fleet(88, 10), _lab(ratio=9.9), "lab_saturated"),
        (_fleet(88, 10), _lab(free_mem_mib=10), "lab_saturated"),
        ({"ok": False, "error": "timeout"}, _lab(), "probe_error:timeout"),
        (None, _lab(), "probe_error:unexpected_shape"),
    ],
)
def test_every_capacity_downgrade_is_reversed_for_a_private_repo(
    fleet: Any, lab: Any, expected_from: str
) -> None:
    """All six capacity/probe reasons, not just the one in the headline case.

    A rule that only covered `fleet_saturated` would leave five other paths
    placing a private repository hosted, and each of them is reachable on an
    ordinary run.
    """
    result = _decide(visibility=VIS_PRIVATE, seam_json=LAB_SEAM, fleet=fleet, lab=lab)
    assert result.labels == LAB_LABELS
    assert result.reason == "private_repo_no_hosted_downgrade"
    assert result.inputs["downgrade_refused_from"] == expected_from


def test_a_private_repo_with_a_hosted_only_ceiling_is_refused_not_placed() -> None:
    """Hosted is the only placement the seam permits, and this repository may
    not be placed hosted. There is no answer, so the router says so instead of
    inventing one: blocked, no labels.
    """
    result = _decide(visibility=VIS_PRIVATE, seam_json=HOSTED_SEAM)
    assert result.decision == "blocked"
    assert result.labels == []
    assert result.reason == "private_repo_hosted_forbidden:seam_ceiling_hosted"


def test_a_private_repo_fork_pull_request_is_refused_not_put_on_the_fleet() -> None:
    """THE CASE THE RULE MUST NOT GET WRONG. Two inviolable rules collide:
    untrusted code never reaches self-hosted compute (OMN-16683), and a private
    repository never runs hosted (2026-09-14). Reversing the fork isolation
    would resolve the collision by putting untrusted code on the lab fleet,
    which no capacity argument may ever do. The refusal is the only safe answer.
    """
    result = _decide(
        visibility=VIS_PRIVATE,
        seam_json=LAB_SEAM,
        event_name="pull_request",
        head_repo="fork/omnibase_infra",
        fleet=_fleet(88, 0),
    )
    assert result.decision == "blocked"
    assert result.labels == []
    assert result.reason == "private_repo_hosted_forbidden:fork_isolation"
    assert "self-hosted" not in result.labels


def test_a_private_repo_on_an_allowlisted_workflow_is_refused() -> None:
    """The hosted allowlist encodes reasons that outrank capacity -- fate
    isolation from the fleet, clean egress for a registry push. Reversing one
    would put an ECR push or a fleet canary on the very fleet it is isolated
    from, so this is a refusal too, not a conversion.
    """
    result = _decide(
        visibility=VIS_PRIVATE,
        seam_json=LAB_SEAM,
        workflow_path=".github/workflows/build-and-push-runtime.yml",
        allowlist=[".github/workflows/build-and-push-runtime.yml"],
        fleet=_fleet(88, 0),
    )
    assert result.decision == "blocked"
    assert result.reason == "private_repo_hosted_forbidden:policy_allowlist"


def test_a_private_repo_with_capacity_available_is_untouched() -> None:
    """The rule only ever inspects a HOSTED answer. An answer that already
    names the fleet passes through with its own reason intact, so a private
    repository's ordinary run is not relabelled into looking like a refusal.
    """
    result = _decide(visibility=VIS_PRIVATE, seam_json=LAB_SEAM, fleet=_fleet(88, 0))
    assert result.labels == LAB_LABELS
    assert result.reason == "capacity_available"


def test_unknown_visibility_is_not_downgraded_when_the_ceiling_names_the_fleet() -> (
    None
):
    """Fail-closed for the half that is safe to fail closed: if the router
    cannot PROVE the repository is public, it does not move the run off the
    fleet. Named distinctly from the private case so the record does not claim
    a fact the probe never established.
    """
    result = _decide(
        visibility=VIS_UNKNOWN,
        seam_json=LAB_SEAM,
        fleet=_fleet(online=88, busy=85),
    )
    assert result.labels == LAB_LABELS
    assert result.reason == "visibility_unknown_no_hosted_downgrade"


def test_unknown_visibility_with_a_hosted_ceiling_never_refuses() -> None:
    """THE ASYMMETRY, AND WHY IT IS DELIBERATE. Every public repository's seam
    reads `["ubuntu-latest"]` today, so refusing on unknown visibility would
    turn one transient metadata read failure into a fleet-wide CI outage. The
    misconfigured-private case is covered statically by the exported
    `private-repo-runner-placement` gate, which reads visibility itself.
    """
    result = _decide(visibility=VIS_UNKNOWN, seam_json=HOSTED_SEAM)
    assert result.decision == "hosted"
    assert result.labels == HOSTED_LABELS
    assert result.reason == "seam_ceiling_hosted"
    assert result.inputs["visibility"] == VIS_UNKNOWN.value


def test_a_public_repo_is_completely_unaffected() -> None:
    """Regression control: the rule must be invisible to every public repo,
    which is all seven routing scopes that could route today.
    """
    saturated = _decide(seam_json=LAB_SEAM, fleet=_fleet(88, 85))
    assert saturated.labels == HOSTED_LABELS
    assert saturated.reason == "fleet_saturated"
    inert = _decide(seam_json=HOSTED_SEAM)
    assert inert.reason == "seam_ceiling_hosted"


def test_a_blocked_decision_is_json_serialisable_and_names_its_cause() -> None:
    """A refusal is an audit record like any other decision: the run log is not
    the evidence surface, the artifact is.
    """
    result = _decide(visibility=VIS_PRIVATE, seam_json=HOSTED_SEAM)
    record = json.loads(json.dumps(result.to_record()))
    assert record["decision"] == "blocked"
    assert record["runs_on"] == []
    assert record["evidence"]["visibility"] == "private"
    assert record["reason"] == "private_repo_hosted_forbidden"
    assert record["reason_detail"] == "seam_ceiling_hosted"


def test_visibility_is_required_and_has_no_default() -> None:
    """Rule 8: a defaulted visibility fails OPEN. A caller that does not supply
    it is a wiring error, and it is loud.
    """
    with pytest.raises(ValidationError) as excinfo:
        ModelCIRunnerRouteRequest(
            github_event="push",
            repository="OmniNode-ai/omnibase_infra",
            workflow_path=".github/workflows/ci.yml",
            seam_json=LAB_SEAM,
            public_json=HOSTED_SEAM,
            fleet={"ok": True, "online": 88, "busy": 10},
            fleet_expected_count=FLEET_EXPECTED,
            lab={"ok": True, "age_seconds": 30, "hosts": []},
            policy=ModelCIRunnerRoutePolicy.model_validate(_policy()),
        )
    assert "visibility" in str(excinfo.value)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("public", "public"),
        ("private", "private"),
        # `internal` is not publicly readable and its hosted minutes are billed
        # exactly like a private repository's, so it is private here. Treating
        # it as public because the API spells it differently would be a silent
        # exemption from the ruling.
        ("internal", "private"),
        ("unknown", "unknown"),
        (None, "unknown"),
        ("", "unknown"),
        (True, "unknown"),
        ("PUBLIC", "unknown"),
    ],
)
def test_visibility_normalisation_never_guesses_public(raw: Any, expected: str) -> None:
    assert route.normalise_visibility(raw) == expected


# --- the probe, at the I/O boundary ----------------------------------------


class _Response:
    def __init__(self, payload: Any) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"visibility": "public"}, "public"),
        ({"visibility": "private"}, "private"),
        ({"visibility": "internal"}, "private"),
        # the older boolean form, read ONLY when `visibility` is absent
        ({"private": True}, "private"),
        ({"private": False}, "public"),
        ({}, "unknown"),
        ([], "unknown"),
        ({"visibility": "something-new"}, "unknown"),
    ],
)
def test_probe_repo_visibility_reads_the_live_repository(
    payload: Any, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        route.urllib.request, "urlopen", lambda *a, **k: _Response(payload)
    )
    assert (
        route.probe_repo_visibility(
            "t", "OmniNode-ai/omniweb", "https://api.github.com"
        )
        == expected
    )


@pytest.mark.parametrize(
    ("token", "repository", "api_url"),
    [
        (None, "OmniNode-ai/omniweb", "https://api.github.com"),
        ("", "OmniNode-ai/omniweb", "https://api.github.com"),
        ("t", "omniweb", "https://api.github.com"),
        ("t", "", "https://api.github.com"),
        ("t", "OmniNode-ai/omniweb", "file:///etc/passwd"),
        ("t", "OmniNode-ai/omniweb", "http://api.github.com"),
    ],
)
def test_probe_repo_visibility_is_unknown_on_every_bad_input(
    token: Any, repository: str, api_url: str
) -> None:
    """Including the scheme pin: the answer decides where jobs execute, so it
    may only ever come from the real API over https.
    """
    assert route.probe_repo_visibility(token, repository, api_url) == "unknown"


def test_probe_repo_visibility_is_unknown_on_a_transport_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*_a: Any, **_k: Any) -> None:
        raise route.urllib.error.URLError("down")

    monkeypatch.setattr(route.urllib.request, "urlopen", _raise)
    assert (
        route.probe_repo_visibility(
            "t", "OmniNode-ai/omniweb", "https://api.github.com"
        )
        == "unknown"
    )


def test_the_probe_test_is_not_vacuous(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive control for the two tests above: a stub that returns a real
    payload must produce a real answer, or "always unknown" would pass them
    both while the probe was broken.
    """
    monkeypatch.setattr(
        route.urllib.request,
        "urlopen",
        lambda *a, **k: _Response({"private": True}),
    )
    assert (
        route.probe_repo_visibility(
            "t", "OmniNode-ai/omniweb", "https://api.github.com"
        )
        == "private"
    )


def test_the_policy_file_declares_no_repository_visibility() -> None:
    """Visibility is read live, never declared. A declared list is a second
    copy of a fact GitHub owns and goes stale silently the first time a
    repository changes visibility -- the failure rule 14 names for the routing
    table it tells you not to enumerate from.
    """
    text = (REPO_ROOT / "config" / "runner_routing_policy.yaml").read_text(
        encoding="utf-8"
    )
    import yaml as _yaml

    section = _yaml.safe_load(text)["route"]
    assert "visibility" not in section
    assert "private_repositories" not in section
