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
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "ci" / "runner_route_decision.py"

_spec = importlib.util.spec_from_file_location("runner_route_decision", MODULE_PATH)
assert _spec is not None and _spec.loader is not None
route = importlib.util.module_from_spec(_spec)
sys.modules["runner_route_decision"] = route
_spec.loader.exec_module(route)


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
        "min_online_runners": 60,
        "lab_record_max_age_seconds": 600,
        "max_lab_load_ratio": 1.0,
        "min_lab_free_mem_mib": 4096,
        "hosted_workflows": [],
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


def _decide(**overrides: Any) -> Any:
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
    }
    kwargs.update(overrides)
    return route.decide(**kwargs)


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


def test_an_exception_inside_the_decision_still_yields_hosted() -> None:
    """The boundary catches everything: a crashing route job still emits a usable runs-on.

    A mapping that raises on access stands in for any unforeseen internal fault.
    """

    class Exploding(dict):  # type: ignore[type-arg]
        def get(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("probe exploded")

    result = _decide(fleet=Exploding())
    assert result.labels == HOSTED_LABELS
    assert result.reason.startswith("probe_error:")


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

    checked = 0
    for seam, event, head, fleet, lab, path in itertools.product(
        seams, events, head_repos, fleets, labs, paths
    ):
        result = _decide(
            seam_json=seam,
            event_name=event,
            head_repo=head,
            fleet=fleet,
            lab=lab,
            workflow_path=path,
        )
        seam_labels = set(json.loads(seam))
        returned = set(result.labels)
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
    ) * len(paths)
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
        "min_online_runners",
        "lab_record_max_age_seconds",
        "max_lab_load_ratio",
        "min_lab_free_mem_mib",
        "lab_labels",
        "hosted_labels",
        "policy_version",
        "runner_group",
        "hosted_workflows",
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
    policy_file = tmp_path / "runner_routing_policy.yaml"
    policy_file.write_text(json.dumps({"route": section}), encoding="utf-8")
    with pytest.raises(KeyError) as excinfo:
        route.load_route_policy(policy_file)
    assert missing_key in str(excinfo.value)


def test_a_policy_file_with_no_route_section_raises(tmp_path: Path) -> None:
    policy_file = tmp_path / "runner_routing_policy.yaml"
    policy_file.write_text(
        json.dumps({"repositories": ["omnibase_infra"]}), encoding="utf-8"
    )
    with pytest.raises(KeyError):
        route.load_route_policy(policy_file)


def test_the_committed_policy_file_loads_and_carries_every_threshold() -> None:
    """The real config/runner_routing_policy.yaml must satisfy the same loader."""
    loaded = route.load_route_policy(
        REPO_ROOT / "config" / "runner_routing_policy.yaml"
    )
    assert loaded["runner_group"] == "omnibase-ci"
    assert loaded["hosted_labels"] == ["ubuntu-latest"]
    assert loaded["max_lab_load_ratio"] == 1.0
    assert loaded["min_lab_free_mem_mib"] == 4096
    # THE REGRESSION A LIVE DRY RUN CAUGHT: ci.yml is in hosted_runner_allowlist
    # (for its one bare-hosted CI Summary job) but must NOT be in the route
    # hosted list, or all 54 of its jobs are pinned hosted forever and the
    # mechanism is a silent no-op on its largest consumer.
    assert ".github/workflows/ci.yml" not in loaded["hosted_workflows"]
    assert ".github/workflows/runner-route-reusable.yml" in loaded["hosted_workflows"]


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


def test_the_decision_record_carries_its_inputs_and_is_json_serialisable() -> None:
    """The artifact the audit reads must explain itself without the run log."""
    result = _decide()
    record = result.to_record()
    payload = json.loads(json.dumps(record))
    assert payload["decision"] == "self_hosted"
    assert payload["reason"] == "capacity_available"
    assert payload["labels"] == LAB_LABELS
    assert payload["policy_version"] == 1
    assert "decided_at" in payload
    assert payload["inputs"]["event_name"] == "push"
    assert payload["inputs"]["fleet"]["online"] == 88
