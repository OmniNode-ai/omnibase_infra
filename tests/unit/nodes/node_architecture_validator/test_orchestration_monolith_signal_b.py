# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for ARCH-004 Signal B — orchestration-monolith complexity (OMN-13486).

Signal B is a DISTINCT rule from Signal A (the decorative-FSM + handler-driven
``_transition`` triad in
``validator_contract_declared_orchestrator_workflow``). Signal B catches the
Class-B monolith — a single oversized orchestrator handler carrying high
branch/event complexity and a fan-in/fan-out routing table — *independent of FSM
decorativeness*. A node can fail Signal B while PASSING Signal A (no decorative
FSM, no handler-owned ``_transition``), which is precisely the build_loop /
session / memory_lifecycle / swarm_dispatch / redeploy shape.

Headline proofs:
    (a) synthetic orchestrator monolith (large handler + many branch/event
        markers + fan-in/out, NO decorative FSM) -> Signal A PASSES, Signal B
        FAILS;
    (b) thin executor-bound orchestrator -> both pass;
    (c) the real Class-B catalog is detected by Signal B in a full-audit scan.

All synthetic fixtures are vendored mini node directories under ``tmp_path``;
no test depends on a sibling-repo path.

Refs OMN-13486 (epic OMN-13485), OMN-13472 (Signal A), OMN-12550 / OMN-13325.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.enums import EnumValidationSeverity
from omnibase_infra.nodes.node_architecture_validator.validators.scanner_orchestration_monolith_ratchet import (
    MONOLITH_BASELINE_RELATIVE_PATH,
    MonolithBaselineEntry,
    discover_node_dirs,
    load_monolith_baseline,
    main,
    monolith_ratchet_violations,
    render_monolith_baseline_yaml,
    scan_node_dirs_for_monolith,
    write_monolith_baseline,
)
from omnibase_infra.nodes.node_architecture_validator.validators.validator_contract_declared_orchestrator_workflow import (
    validate_contract_declared_orchestrator_workflow,
)
from omnibase_infra.nodes.node_architecture_validator.validators.validator_orchestration_monolith import (
    MONOLITH_RULE_ID,
    RuleOrchestrationMonolith,
    analyze_monolith,
    validate_orchestration_monolith,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Synthetic node-directory fixture builder
# --------------------------------------------------------------------------- #
def _make_node_dir(
    root: Path,
    *,
    node_name: str,
    contract_yaml: str,
    handler_name: str = "handler_workflow",
    handler_source: str | None = None,
) -> Path:
    node_dir = root / "src" / "pkg" / "nodes" / node_name
    handlers_dir = node_dir / "handlers"
    handlers_dir.mkdir(parents=True, exist_ok=True)
    (node_dir / "contract.yaml").write_text(contract_yaml, encoding="utf-8")
    if handler_source is not None:
        (handlers_dir / f"{handler_name}.py").write_text(
            handler_source, encoding="utf-8"
        )
    return node_dir


def _build_routing_table(strategy: str, n_ops: int) -> str:
    rows: list[str] = []
    for i in range(n_ops):
        rows.append(
            f"    - operation: op{i}\n"
            f"      handler:\n"
            f"        name: HandlerWorkflow\n"
            f"        module: pkg.nodes.node_x.handlers.handler_workflow\n"
        )
    return f"handler_routing:\n  routing_strategy: {strategy}\n  handlers:\n" + "".join(
        rows
    )


def _monolith_handler_source(
    *, n_branches: int, n_events: int, total_lines: int = 800
) -> str:
    """Synthetic monolith handler: many branch + event markers, NO _transition.

    Deliberately avoids the Signal A state-drive symbols (no ``self._transition(``,
    no ``Enum*State`` mirroring contract states) so Signal A does NOT fire — only
    Signal B's complexity signal should. ``total_lines`` pads the handler body
    with inert comment lines so it crosses the monolith line threshold while the
    branch/event marker densities stay calibrated to the test's intent.
    """
    branches = "\n".join(
        f"        if value == {i} or value > {i}:\n            value += 1"
        for i in range(n_branches)
    )
    events = "\n".join(
        f"        results.append(ModelStepEvent(step={i}))" for i in range(n_events)
    )
    body = (
        '"""Synthetic orchestrator monolith handler (no FSM, no _transition)."""\n'
        "from __future__ import annotations\n\n\n"
        "class HandlerWorkflow:\n"
        "    def handle(self, envelope):\n"
        "        value = 0\n"
        "        results = []\n"
        f"{branches}\n"
        f"{events}\n"
        "        return results\n"
    )
    current = body.count("\n") + 1
    if current < total_lines:
        padding = "\n".join(
            f"# inert filler line {i}" for i in range(total_lines - current)
        )
        body = body + padding + "\n"
    return body


# --------------------------------------------------------------------------- #
# (a) Monolith with NO decorative FSM: Signal A passes, Signal B fails.
# --------------------------------------------------------------------------- #
def test_monolith_no_fsm_signal_a_passes_signal_b_fails(tmp_path: Path) -> None:
    """HEADLINE: monolith (no FSM) — Signal A PASSES, Signal B FAILS."""
    contract = (
        "name: node_synth_monolith\n"
        "node_type: ORCHESTRATOR_GENERIC\n"
        # NOTE: no fsm: block at all -> Signal A's H1/H4 cannot fire.
        + _build_routing_table("operation_match", n_ops=6)
    )
    handler = _monolith_handler_source(n_branches=80, n_events=30)
    node_dir = _make_node_dir(
        tmp_path,
        node_name="node_synth_monolith",
        contract_yaml=contract,
        handler_source=handler,
    )

    # Signal A: no decorative FSM, no handler-owned _transition, operation_match
    # (not payload_type_match) -> no hard fail.
    signal_a = validate_contract_declared_orchestrator_workflow(str(node_dir))
    assert signal_a.valid, (
        "Signal A must PASS the no-FSM monolith (no decorative FSM, no "
        "handler-owned _transition, operation_match routing)."
    )

    # Signal B: the orchestration-monolith complexity signal must FAIL.
    signal_b = validate_orchestration_monolith(str(node_dir))
    assert not signal_b.valid, "Signal B must FAIL the orchestration monolith."
    codes = {
        str(v.details["finding_code"])
        for v in signal_b.violations
        if v.details and "finding_code" in v.details
    }
    assert "B1" in codes, "Expected B1 (monolith hard-fail)."
    assert all(v.rule_id == MONOLITH_RULE_ID for v in signal_b.violations)


# --------------------------------------------------------------------------- #
# (b) Thin executor-bound orchestrator: both pass.
# --------------------------------------------------------------------------- #
def test_thin_executor_bound_orchestrator_passes_both(tmp_path: Path) -> None:
    """A thin, executor-bound orchestrator passes BOTH Signal A and Signal B."""
    contract = (
        "name: node_synth_bound\n"
        "node_type: ORCHESTRATOR_GENERIC\n"
        "workflow_coordination:\n"
        "  execution_graph:\n"
        "    - step: route\n"
        "      next: emit\n"
        "    - step: emit\n"
        "      next: done\n" + _build_routing_table("operation_match", n_ops=2)
    )
    handler = (
        '"""Thin per-step handler; executor-bound, no monolith."""\n\n\n'
        "class HandlerWorkflow:\n"
        "    def handle(self, envelope):\n"
        "        return envelope\n"
    )
    node_dir = _make_node_dir(
        tmp_path,
        node_name="node_synth_bound",
        contract_yaml=contract,
        handler_source=handler,
    )

    signal_a = validate_contract_declared_orchestrator_workflow(str(node_dir))
    assert signal_a.valid, "Signal A must PASS the thin executor-bound orchestrator."

    signal_b = validate_orchestration_monolith(str(node_dir))
    assert signal_b.valid, "Signal B must PASS the thin executor-bound orchestrator."
    assert len(signal_b.violations) == 0


def test_executor_bound_large_handler_is_not_a_monolith(tmp_path: Path) -> None:
    """A LARGE handler that is executor-bound is debt-managed by the executor.

    An oversized handler whose workflow is executor-bound (declares an
    ``execution_graph``) is NOT the Class-B monolith Signal B targets — the
    coordination has been delegated to the executor even if the handler body is
    large. Signal B must not hard-fail it.
    """
    contract = (
        "name: node_synth_bound_large\n"
        "node_type: ORCHESTRATOR_GENERIC\n"
        "workflow_coordination:\n"
        "  execution_graph:\n"
        "    - step: a\n"
        "      next: b\n" + _build_routing_table("operation_match", n_ops=6)
    )
    handler = _monolith_handler_source(n_branches=80, n_events=30)
    node_dir = _make_node_dir(
        tmp_path,
        node_name="node_synth_bound_large",
        contract_yaml=contract,
        handler_source=handler,
    )
    result = validate_orchestration_monolith(str(node_dir))
    assert result.valid, (
        "An executor-bound orchestrator must not hard-fail Signal B even when "
        "its handler is large (coordination is delegated to the executor)."
    )


# --------------------------------------------------------------------------- #
# Reducer / non-orchestrator exemption
# --------------------------------------------------------------------------- #
def test_reducer_is_exempt_from_signal_b(tmp_path: Path) -> None:
    """Reducers legitimately own state and are exempt from Signal B."""
    contract = (
        "name: node_synth_reducer\n"
        "node_type: REDUCER_GENERIC\n"
        "state_machine:\n  states:\n    - A\n    - B\n"
    )
    handler = _monolith_handler_source(n_branches=80, n_events=30)
    node_dir = _make_node_dir(
        tmp_path,
        node_name="node_synth_reducer",
        contract_yaml=contract,
        handler_source=handler,
    )
    assert analyze_monolith(node_dir) is None, "Reducers must be exempt."
    result = validate_orchestration_monolith(str(node_dir))
    assert result.valid and len(result.violations) == 0


def test_non_orchestrator_is_out_of_scope(tmp_path: Path) -> None:
    """A COMPUTE/EFFECT node that is not orchestrator-like is out of scope."""
    contract = (
        "name: node_synth_compute\n"
        "node_type: COMPUTE_GENERIC\n"
        + _build_routing_table("operation_match", n_ops=6)
    )
    handler = _monolith_handler_source(n_branches=80, n_events=30)
    node_dir = _make_node_dir(
        tmp_path,
        node_name="node_synth_compute",
        contract_yaml=contract,
        handler_source=handler,
    )
    assert analyze_monolith(node_dir) is None


# --------------------------------------------------------------------------- #
# (c) Full-audit detection of the real Class-B catalog (synthetic mirror).
# --------------------------------------------------------------------------- #
def test_full_audit_detects_monolith_catalog(tmp_path: Path) -> None:
    """Full-audit mode detects monolith-shaped nodes and ignores thin ones."""
    # build_loop-shaped: large operation_match handler, no fsm in handler-driven
    # sense (the contract may declare a decorative fsm, but Signal B is
    # independent of it).
    _make_node_dir(
        tmp_path,
        node_name="node_build_loop_orchestrator",
        contract_yaml=(
            "name: node_build_loop_orchestrator\n"
            "node_type: ORCHESTRATOR_GENERIC\n"
            + _build_routing_table("operation_match", n_ops=8)
        ),
        handler_source=_monolith_handler_source(n_branches=90, n_events=34),
    )
    # session-shaped: large, no fsm at all.
    _make_node_dir(
        tmp_path,
        node_name="node_session_orchestrator",
        contract_yaml=(
            "name: node_session_orchestrator\n"
            "node_type: ORCHESTRATOR_GENERIC\n"
            + _build_routing_table("operation_match", n_ops=5)
        ),
        handler_source=_monolith_handler_source(n_branches=100, n_events=3),
    )
    # A thin orchestrator that must NOT be flagged.
    _make_node_dir(
        tmp_path,
        node_name="node_thin_orchestrator",
        contract_yaml=(
            "name: node_thin_orchestrator\n"
            "node_type: ORCHESTRATOR_GENERIC\n"
            + _build_routing_table("operation_match", n_ops=2)
        ),
        handler_source=(
            "class HandlerWorkflow:\n    def handle(self, e):\n        return e\n"
        ),
    )

    node_dirs = discover_node_dirs(tmp_path)
    result = scan_node_dirs_for_monolith("synthetic", node_dirs, repo_root=tmp_path)
    flagged = {e.node for e in result.hard_fails}
    assert "node_build_loop_orchestrator" in flagged
    assert "node_session_orchestrator" in flagged
    assert "node_thin_orchestrator" not in flagged
    bl = next(e for e in result.hard_fails if e.node == "node_build_loop_orchestrator")
    assert "B1" in bl.finding_codes
    assert not bl.max_handler_path.startswith("/")


# --------------------------------------------------------------------------- #
# Distinctness from Signal A: separate finding-code namespace.
# --------------------------------------------------------------------------- #
def test_signal_b_finding_codes_are_distinct_namespace(tmp_path: Path) -> None:
    """Signal B finding codes are B*, never the A-namespace H*/W* codes."""
    contract = (
        "name: node_synth_monolith\n"
        "node_type: ORCHESTRATOR_GENERIC\n"
        + _build_routing_table("operation_match", n_ops=6)
    )
    handler = _monolith_handler_source(n_branches=80, n_events=30)
    node_dir = _make_node_dir(
        tmp_path,
        node_name="node_synth_monolith",
        contract_yaml=contract,
        handler_source=handler,
    )
    analysis = analyze_monolith(node_dir)
    assert analysis is not None
    assert analysis.finding_codes, "monolith must produce finding codes"
    for code in analysis.finding_codes:
        assert code.startswith("B"), (
            f"Signal B finding code {code!r} must be in the B* namespace, "
            f"never the Signal A H*/W* namespace."
        )


# --------------------------------------------------------------------------- #
# Rule protocol surface.
# --------------------------------------------------------------------------- #
def test_rule_class_protocol_surface(tmp_path: Path) -> None:
    rule = RuleOrchestrationMonolith()
    assert rule.rule_id == MONOLITH_RULE_ID
    assert rule.rule_id != "ARCH-004", (
        "Signal B must be a DISTINCT rule id, not the Signal A ARCH-004 id."
    )
    assert rule.severity == EnumValidationSeverity.ERROR
    assert rule.name
    assert rule.description

    skipped = rule.check(object())
    assert skipped.skipped is True
    assert skipped.passed is True

    contract = (
        "name: node_synth_monolith\n"
        "node_type: ORCHESTRATOR_GENERIC\n"
        + _build_routing_table("operation_match", n_ops=6)
    )
    node_dir = _make_node_dir(
        tmp_path,
        node_name="node_synth_monolith",
        contract_yaml=contract,
        handler_source=_monolith_handler_source(n_branches=80, n_events=30),
    )
    res = rule.check(str(node_dir / "contract.yaml"))
    assert res.passed is False
    assert "B1" in (res.details.get("finding_codes") or [])


# --------------------------------------------------------------------------- #
# Baseline behaviour (separate dimension / file).
# --------------------------------------------------------------------------- #
def _monolith_entry() -> MonolithBaselineEntry:
    return MonolithBaselineEntry(
        repo="synthetic",
        node="node_build_loop_orchestrator",
        max_handler_path="src/pkg/nodes/node_build_loop_orchestrator/handlers/h.py",
        line_count=1581,
        complexity_score=5,
        fan_in=8,
        fan_out=8,
        finding_codes=("B1",),
        owner_ticket="OMN-13487",
    )


def test_monolith_baseline_path_distinct_from_signal_a() -> None:
    """The monolith baseline file must NOT be the Signal A baseline file."""
    assert MONOLITH_BASELINE_RELATIVE_PATH != (
        "architecture-handshakes/imperative-orchestrator-baseline.yaml"
    )
    assert MONOLITH_BASELINE_RELATIVE_PATH.endswith(".yaml")
    assert "monolith" in MONOLITH_BASELINE_RELATIVE_PATH


def test_monolith_ratchet_passes_when_matching_baseline() -> None:
    entry = _monolith_entry()
    baseline = {f"{entry.repo}::{entry.node}": entry}
    assert monolith_ratchet_violations([entry], baseline) == []


def test_monolith_ratchet_fails_on_new_untracked_node() -> None:
    entry = _monolith_entry()
    failures = monolith_ratchet_violations([entry], baseline={})
    assert len(failures) == 1
    assert "NEW orchestration-monolith" in failures[0]


def test_monolith_ratchet_fails_on_growth() -> None:
    base = _monolith_entry()
    worse = MonolithBaselineEntry(
        repo=base.repo,
        node=base.node,
        max_handler_path=base.max_handler_path,
        line_count=base.line_count + 200,
        complexity_score=base.complexity_score + 1,
        fan_in=base.fan_in,
        fan_out=base.fan_out + 2,
        finding_codes=base.finding_codes,
        owner_ticket=base.owner_ticket,
    )
    failures = monolith_ratchet_violations(
        [worse], baseline={f"{base.repo}::{base.node}": base}
    )
    joined = "\n".join(failures)
    assert "max handler GREW" in joined
    assert "complexity score WORSENED" in joined


def test_monolith_ratchet_fails_when_baseline_entry_has_no_owner_ticket() -> None:
    base = MonolithBaselineEntry(
        repo="synthetic",
        node="node_x",
        max_handler_path="x.py",
        line_count=1581,
        complexity_score=5,
        fan_in=8,
        fan_out=8,
        finding_codes=("B1",),
        owner_ticket="",
    )
    failures = monolith_ratchet_violations(
        [base], baseline={f"{base.repo}::{base.node}": base}
    )
    assert any("no owner_ticket" in f for f in failures)


def test_monolith_baseline_roundtrip(tmp_path: Path) -> None:
    entry = _monolith_entry()
    rendered = render_monolith_baseline_yaml("omnibase_infra", [entry])
    assert "/Users/" not in rendered and "/Volumes/" not in rendered
    # Distinct rule id recorded in the document (not ARCH-004).
    assert MONOLITH_RULE_ID in rendered
    baseline_path = tmp_path / "architecture-handshakes" / "monolith-baseline.yaml"
    write_monolith_baseline(baseline_path, "omnibase_infra", [entry])
    loaded = load_monolith_baseline(baseline_path)
    key = f"{entry.repo}::{entry.node}"
    assert key in loaded
    assert loaded[key].finding_codes == entry.finding_codes
    assert loaded[key].owner_ticket == "OMN-13487"
    assert loaded[key].fan_out == 8


def test_monolith_baseline_entry_supports_not_applicable() -> None:
    """A node assessed as executor-bound/thin records not-applicable rationale."""
    entry = MonolithBaselineEntry(
        repo="omnibase_infra",
        node="node_redeploy_orchestrator",
        max_handler_path="src/x.py",
        line_count=422,
        complexity_score=1,
        fan_in=1,
        fan_out=4,
        finding_codes=(),
        owner_ticket="OMN-13486",
        not_applicable_rationale=(
            "Assessed only (authority surface untouched): 422L operation_match, "
            "below monolith threshold; executor-coordinated. Not Class B."
        ),
    )
    restored = MonolithBaselineEntry.from_dict(entry.to_dict())
    assert restored == entry
    assert restored.not_applicable_rationale


def test_write_baseline_requires_check_all() -> None:
    rc = main(["--check-changed", "--write-baseline"])
    assert rc == 1
