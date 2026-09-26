# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Offline grading tests for the C28 consumer-flow producer (OMN-19812).

The green observation below is built in this file, shaped exactly like what
``observe_lane`` returned on the lab dev lane on 2026-09-26 (409 live groups,
two pages, the two named kinds, the applied topic advancing). Every red case is
that observation with ONE property broken, so each test says which property of
the lane the verdict turns on.

Three inputs are REAL BYTES, captured from that lane rather than typed:

* ``runtime_seam_probe_prefixed.captured.txt`` -- the seam's own log lines for a
  malformed payload this probe published (correlation prefixed ``c28c28c2-c28c-``).
* ``runtime_seam_unprefixed.captured.txt`` -- the same seam refusing a payload
  published by hand with an ordinary correlation id, which is exactly what a
  natural error looks like to the probe and must count as one.
* ``rpk_describe_partitions.captured.txt`` -- ``rpk topic describe -p`` output.

Both logs are the lines naming the seam (``ModelConsumerFlowStallAlert``,
``boundary_swallow_prevented``, ``Auto-wiring callback error``) filtered, in
order and unmodified, out of ``docker logs omninode-runtime`` windows at
2026-09-26T19:34Z and 19:44Z.

Clause 2's mutation is exercised against the REAL ``handler_wiring.py`` in this
checkout. Nothing here touches docker, the network or a lane.
"""

from __future__ import annotations

import ast
import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci import c28_consumer_flow_probe as probe

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "omn19812"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "chain-canary-c28-consumer-flow.yml"

AUDIT = "local.omnibase_infra.node_gateway_link_health_projection_compute.consume.1.0.0"
REDUCER = "local.omnimarket.projection_consumer_flow.consume.1.0.1"


def _row(
    group: str, topic: str, start: str, *, n_in: Any, n_out: Any, state: str
) -> dict[str, Any]:
    return {
        "consumer_group": group,
        "topic": topic,
        "window_start": start,
        "messages_in": n_in,
        "messages_out": n_out,
        "flow_state": state,
    }


def _sample(i: int) -> list[dict[str, Any]]:
    start = f"2026-09-26T19:4{i}:26.257435+00:00"
    return [
        _row(
            AUDIT,
            "onex.evt.omnibase-infra.gateway-heartbeat.v1",
            start,
            n_in=2,
            n_out=0,
            state="STALLED",
        ),
        _row(
            REDUCER,
            "onex.evt.platform.node-heartbeat.v1",
            start,
            n_in=565,
            n_out=565,
            state="FLOWING",
        ),
        _row(
            "local.omnimarket.other.consume.1.0.0",
            "onex.evt.x.v1",
            start,
            n_in=0,
            n_out=0,
            state="IDLE",
        ),
    ]


def _outcomes(failing: set[str] = frozenset()) -> dict[str, str]:  # type: ignore[assignment]
    names = [
        "test_the_gate_enumerates_more_than_one_subscription_branch",
        probe.AST_GATE_TEST,
        "test_every_selected_subscription_factory_is_passed_the_consumer_group",
    ]
    for base in (
        "test_every_wiring_branch_registers_a_flow_counter_at_wiring_time",
        "test_every_wiring_branch_emits_a_zero_row_when_it_takes_nothing",
        "test_every_wiring_branch_counts_a_stalled_consumer_apart_from_a_flowing_one",
        "test_no_branch_fabricates_a_group_when_none_is_wired",
    ):
        names += [f"{base}[event_bus]", f"{base}[raw_event_projection]"]
    return {n: ("failed" if n in failing else "passed") for n in names}


def _bites(branch: str) -> dict[str, Any]:
    failing = {
        probe.AST_GATE_TEST,
        f"test_every_wiring_branch_registers_a_flow_counter_at_wiring_time[{branch}]",
        f"test_every_wiring_branch_emits_a_zero_row_when_it_takes_nothing[{branch}]",
    }
    return {
        "factory": probe.BRANCHES[branch],
        "applied": True,
        "returncode": 1,
        "outcomes": _outcomes(failing),
        "restored": True,
    }


LIVE = [f"local.group.{i}" for i in range(409)]


def _green() -> dict[str, Any]:
    return {
        "kinds": {"samples": [_sample(i) for i in range(4)]},
        "negative": {
            "clean": {"returncode": 0, "outcomes": _outcomes()},
            "mutations": {b: _bites(b) for b in probe.BRANCHES},
        },
        "cursor": {
            "pages": [
                {"row_count": 500, "row_limit": 500, "next_cursor": "40334485"},
                {"row_count": 61, "row_limit": 500, "next_cursor": None},
            ],
            "terminated": True,
            "second_page_differs": True,
            "live_groups": LIVE,
            "walked_groups": [*LIVE, "local.group.retained"],
        },
        "boot": {
            "applied_hwm_before": 271649,
            "applied_hwm_after": 271661,
            "applied_window_seconds": 90,
            "natural": {
                c: {"lines": 30000, "total": 0, "probe": 0, "natural": 0}
                for c in probe.RUNTIME_CONTAINERS
            },
            "injection": {
                "offset": 271662,
                "correlation_id": "c28c28c2-c28c-4456-b263-05e789085e69",
                "validation_errors_after": 2,
                "boundary_lines": 1,
                "dlq_copies": 1,
            },
        },
    }


def _failed(rec: probe.Record) -> set[str]:
    return {c.name for c in rec.checks if not c.ok}


def test_the_green_observation_passes_on_every_clause() -> None:
    rec = probe.grade(_green())
    assert rec.verdict == "pass", rec.failures
    assert rec.exit_code == probe.EXIT_OK
    assert {c.clause for c in rec.checks} == {"kinds", "negative", "cursor", "boot"}


def test_an_empty_observation_cannot_pass() -> None:
    rec = probe.grade({})
    assert rec.verdict == "fail"
    assert rec.exit_code == probe.EXIT_FINDINGS
    assert all(not c.ok for c in rec.checks), [c.name for c in rec.checks if c.ok]


def _mutate(fn: Callable[[dict[str, Any]], None]) -> set[str]:
    obs = copy.deepcopy(_green())
    fn(obs)
    return _failed(probe.grade(obs))


# ---- RED cases: one property broken each -----------------------------------


def _drop_kind(group: str) -> Callable[[dict[str, Any]], None]:
    def mutate(obs: dict[str, Any]) -> None:
        obs["kinds"]["samples"] = [
            [r for r in s if r["consumer_group"] != group]
            for s in obs["kinds"]["samples"]
        ]

    return mutate


def test_red_when_the_audit_kind_emits_no_row() -> None:
    """The OMN-17214 condition: an audit-purpose subscription with no counter."""
    assert _mutate(_drop_kind(AUDIT)) >= {
        "audit_projection_consumer_present_in_every_sample",
        "audit_projection_consumer_instrumented",
    }


def test_red_when_the_publishing_reducer_emits_no_row() -> None:
    assert "publishing_reducer_present_in_every_sample" in _mutate(_drop_kind(REDUCER))


def test_red_when_a_kind_is_missing_from_one_sample_only() -> None:
    def mutate(obs: dict[str, Any]) -> None:
        obs["kinds"]["samples"][2] = [
            r for r in obs["kinds"]["samples"][2] if r["consumer_group"] != AUDIT
        ]

    assert _mutate(mutate) == {"audit_projection_consumer_present_in_every_sample"}


def test_red_when_a_kind_row_is_unknown_with_null_counters() -> None:
    """A gap row (UNKNOWN, NULL counters) is a row, but not an instrumented one."""

    def mutate(obs: dict[str, Any]) -> None:
        for s in obs["kinds"]["samples"]:
            for r in s:
                if r["consumer_group"] == AUDIT:
                    r.update(flow_state="UNKNOWN", messages_in=None, messages_out=None)

    failed = _mutate(mutate)
    assert "audit_projection_consumer_instrumented" in failed
    assert "audit_projection_consumer_messages_in_counted" in failed


def test_red_when_the_reducer_publishes_are_not_attributed() -> None:
    """The other OMN-17214 half: reducer in=N out=0 read as STALLED."""

    def mutate(obs: dict[str, Any]) -> None:
        for s in obs["kinds"]["samples"]:
            for r in s:
                if r["consumer_group"] == REDUCER:
                    r.update(messages_out=0, flow_state="STALLED")

    assert _mutate(mutate) == {"publishing_reducer_messages_out_counted"}


def test_red_when_the_row_is_retained_but_not_live() -> None:
    def mutate(obs: dict[str, Any]) -> None:
        for s in obs["kinds"]["samples"]:
            for r in s:
                r["window_start"] = "2026-09-19T12:00:00+00:00"

    assert _mutate(mutate) >= {
        "audit_projection_consumer_window_advances",
        "publishing_reducer_window_advances",
    }


def test_a_version_bump_of_a_named_consumer_still_matches() -> None:
    def mutate(obs: dict[str, Any]) -> None:
        for s in obs["kinds"]["samples"]:
            for r in s:
                if r["consumer_group"] == REDUCER:
                    r["consumer_group"] = (
                        "local.omnimarket.projection_consumer_flow.consume.1.0.2"
                    )

    assert _mutate(mutate) == set()


def test_red_when_a_live_group_is_unreachable() -> None:
    """The OMN-17215 condition: a group the table has, the exposure cannot reach."""

    def mutate(obs: dict[str, Any]) -> None:
        obs["cursor"]["walked_groups"] = obs["cursor"]["walked_groups"][50:]

    assert _mutate(mutate) == {"every_live_group_reachable"}


def test_red_when_a_truncated_page_carries_no_cursor() -> None:
    def mutate(obs: dict[str, Any]) -> None:
        obs["cursor"]["pages"] = [
            {"row_count": 500, "row_limit": 500, "next_cursor": None}
        ]

    assert "truncated_pages_carry_a_cursor" in _mutate(mutate)


def test_red_when_the_walk_does_not_terminate() -> None:
    def mutate(obs: dict[str, Any]) -> None:
        obs["cursor"]["terminated"] = False

    assert _mutate(mutate) == {"walk_terminates"}


def test_red_when_the_cursor_does_not_advance() -> None:
    def mutate(obs: dict[str, Any]) -> None:
        obs["cursor"]["second_page_differs"] = False

    assert _mutate(mutate) == {"since_advances"}


def test_red_when_the_live_window_is_empty() -> None:
    """An empty live set would make 'nothing unreachable' vacuous."""

    def mutate(obs: dict[str, Any]) -> None:
        obs["cursor"]["live_groups"] = []

    assert _mutate(mutate) >= {"live_window_is_non_empty", "every_live_group_reachable"}


def test_red_when_the_negative_test_does_not_pass_clean() -> None:
    def mutate(obs: dict[str, Any]) -> None:
        obs["negative"]["clean"] = {
            "returncode": 1,
            "outcomes": _outcomes({probe.AST_GATE_TEST}),
        }

    assert "tests_pass_on_this_checkout" in _mutate(mutate)


def test_red_when_a_mutation_does_not_bite() -> None:
    def mutate(obs: dict[str, Any]) -> None:
        obs["negative"]["mutations"]["raw_event_projection"]["outcomes"] = _outcomes()

    assert _mutate(mutate) == {
        "mutation_raw_event_projection_bites_its_own_branch",
        "mutation_raw_event_projection_fails_the_ast_gate",
    }


def test_red_when_a_mutation_bites_the_wrong_branch() -> None:
    def mutate(obs: dict[str, Any]) -> None:
        run = obs["negative"]["mutations"]["event_bus"]
        run["outcomes"][
            "test_no_branch_fabricates_a_group_when_none_is_wired[raw_event_projection]"
        ] = "failed"

    assert _mutate(mutate) == {"mutation_event_bus_is_branch_specific"}


def test_red_when_the_mutated_module_errors_everything() -> None:
    """Every case erroring 'bites' -- the other branch passing is what refutes it."""

    def mutate(obs: dict[str, Any]) -> None:
        run = obs["negative"]["mutations"]["event_bus"]
        run["outcomes"] = dict.fromkeys(run["outcomes"], "error")

    assert "mutation_event_bus_is_branch_specific" in _mutate(mutate)


def test_red_when_the_mutation_could_not_be_applied_or_restored() -> None:
    def mutate(obs: dict[str, Any]) -> None:
        obs["negative"]["mutations"]["event_bus"].update(applied=False, restored=False)

    assert _mutate(mutate) >= {
        "mutation_event_bus_bites_its_own_branch",
        "mutation_event_bus_restored",
    }


def test_red_when_no_applied_event_arrived() -> None:
    def mutate(obs: dict[str, Any]) -> None:
        obs["boot"]["applied_hwm_after"] = obs["boot"]["applied_hwm_before"]

    assert _mutate(mutate) == {"applied_event_on_this_boot"}


def test_red_on_a_natural_validation_error() -> None:
    def mutate(obs: dict[str, Any]) -> None:
        obs["boot"]["natural"]["omninode-runtime"].update(total=2, natural=2)

    assert _mutate(mutate) == {"zero_natural_stall_alert_validation_errors"}


def test_red_when_the_seam_did_not_observe_the_injection() -> None:
    """A zero from a dead probe: the injected payload was not seen."""

    def mutate(obs: dict[str, Any]) -> None:
        obs["boot"]["injection"].update(
            validation_errors_after=0, boundary_lines=0, dlq_copies=0
        )

    assert _mutate(mutate) == {
        "seam_raised_a_validation_error_after_the_publish",
        "seam_dead_lettered_the_injected_correlation",
        "injected_marker_durably_on_the_dlq",
    }


def test_red_when_the_logs_were_not_read() -> None:
    def mutate(obs: dict[str, Any]) -> None:
        del obs["boot"]["natural"]["omninode-runtime-effects"]

    assert "runtime_logs_read" in _mutate(mutate)


# ---- real bytes -------------------------------------------------------------


def _lines(name: str) -> list[str]:
    return (FIXTURES / name).read_text(encoding="utf-8").splitlines()


def test_errors_this_probe_caused_are_attributed_to_it() -> None:
    total, own, natural = probe.natural_validation_errors(
        _lines("runtime_seam_probe_prefixed.captured.txt")
    )
    assert total > 0
    assert own == total
    assert natural == 0


def test_an_error_with_an_ordinary_correlation_counts_as_natural() -> None:
    total, own, natural = probe.natural_validation_errors(
        _lines("runtime_seam_unprefixed.captured.txt")
    )
    assert total > 0
    assert own == 0
    assert natural == total


def test_an_unpaired_error_is_never_excused() -> None:
    lines = [
        ln
        for ln in _lines("runtime_seam_probe_prefixed.captured.txt")
        if "boundary_swallow" not in ln
    ]
    total, own, natural = probe.natural_validation_errors(lines)
    assert total > 0 and own == 0 and natural == total


def test_both_model_names_count() -> None:
    for name in ("Request", "Trigger"):
        line = f"x failed: ValidationError: 6 validation errors for ModelConsumerFlowStallAlert{name}"
        assert probe.natural_validation_errors([line]) == (1, 0, 1)


def test_the_probe_correlation_prefix_is_a_valid_uuid_and_recognised() -> None:
    cid = probe.probe_correlation_id()
    assert cid.startswith(probe.PROBE_CID_PREFIX)
    line = (
        "metric_name=boundary_swallow_prevented dlq_routed=true dlq_enabled=True "
        f"topic={probe.APPLIED_TOPIC} error_type=HandlerDispatchFailureError correlation_id={cid}"
    )
    match = probe.BOUNDARY_RE.search(line)
    assert match and match.group("cid") == cid


def test_high_watermark_parses_the_captured_describe_output() -> None:
    text = (FIXTURES / "rpk_describe_partitions.captured.txt").read_text(
        encoding="utf-8"
    )
    assert probe.parse_high_watermark(text) == 271676


def test_high_watermark_refuses_output_without_partitions() -> None:
    with pytest.raises(probe.ProbeInputError):
        probe.parse_high_watermark(
            "unable to request topic metadata: SASL authentication failed"
        )


# ---- clause 2 mutation against the real module ------------------------------


@pytest.mark.parametrize("branch", sorted(probe.BRANCHES))
def test_the_mutation_removes_exactly_one_registration(branch: str) -> None:
    source = (REPO_ROOT / probe.WIRING_MODULE).read_text(encoding="utf-8")
    mutated = probe._mutated_source(source, probe.BRANCHES[branch])
    assert mutated is not None
    ast.parse(mutated)
    assert (
        mutated.count("flow_counters.register(")
        == source.count("flow_counters.register(") - 1
    )
    fn = next(
        n
        for n in ast.parse(mutated).body
        if isinstance(n, ast.FunctionDef) and n.name == probe.BRANCHES[branch]
    )
    assert "flow_counters.register(" not in ast.unparse(fn)


def test_the_mutation_refuses_a_factory_that_does_not_exist() -> None:
    source = (REPO_ROOT / probe.WIRING_MODULE).read_text(encoding="utf-8")
    assert probe._mutated_source(source, "_no_such_factory") is None


def test_junit_outcomes_reads_failures_and_param_ids() -> None:
    xml = (
        '<testsuites><testsuite><testcase name="a[event_bus]"/>'
        '<testcase name="b[raw_event_projection]"><failure message="x"/></testcase>'
        '<testcase name="c"><error message="y"/></testcase></testsuite></testsuites>'
    )
    assert probe.junit_outcomes(xml) == {
        "a[event_bus]": "passed",
        "b[raw_event_projection]": "failed",
        "c": "error",
    }


# ---- could-not-look is exit 2 and still leaves a record ---------------------


def test_a_lane_that_cannot_be_read_exits_2_with_a_record(tmp_path: Path) -> None:
    class DeadLane(probe.Lane):
        def identity(self) -> dict[str, dict[str, Any]]:
            raise probe.ProbeInputError("docker inspect omninode-runtime exited 1")

    rec_path = tmp_path / "c28.json"
    code = probe.main(
        ["--record", str(rec_path), "--scratch", str(tmp_path), "--pytest-cmd", "true"],
        lane_factory=DeadLane,
    )
    assert code == probe.EXIT_INPUT
    body = json.loads(rec_path.read_text(encoding="utf-8"))
    assert body["verdict"] == "could_not_look"


def test_replay_of_the_green_observation_exits_0(tmp_path: Path) -> None:
    obs = tmp_path / "obs.json"
    obs.write_text(json.dumps(_green()), encoding="utf-8")
    rec_path = tmp_path / "rec.json"
    assert (
        probe.main(["--replay", str(obs), "--record", str(rec_path)]) == probe.EXIT_OK
    )
    assert json.loads(rec_path.read_text(encoding="utf-8"))["verdict"] == "pass"


# ---- the workflow ------------------------------------------------------------


def test_the_workflow_runs_the_probe_on_the_host_201_verify_runner() -> None:
    wf = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    job = wf["jobs"]["c28-consumer-flow"]
    assert job["runs-on"] == ["self-hosted", "omnibase-verify", "host-201"]
    assert job["name"] == "C28 consumer flow (dev lane)"
    triggers = wf[True] if True in wf else wf["on"]
    assert "schedule" in triggers and "workflow_dispatch" in triggers
    assert "pull_request" not in triggers
    runs = "\n".join(str(s.get("run", "")) for s in job["steps"])
    assert "scripts/ci/c28_consumer_flow_probe.py" in runs
    uploads = [
        s
        for s in job["steps"]
        if str(s.get("uses", "")).startswith("actions/upload-artifact")
    ]
    assert uploads and uploads[0]["with"]["name"] == "c28-consumer-flow"
    assert uploads[0].get("if") == "always()"
