# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for ARCH-004: Contract-Declared Orchestrator Workflow Must Be Bound To An Executor.

ARCH-004 is the cross-file, node-directory rule that catches the imperative
orchestrator anti-pattern ARCH-003 structurally cannot (OMN-13472, epic
OMN-13471).

The HEADLINE proof (``test_arch003_passes_but_arch004_fails_delegation_shape``):
a synthetic node whose contract declares an fsm: table and whose monolithic
handler drives transitions via ``self._transition(...)`` — where the handler
class is NOT named ``*Orchestrator`` — is PASSED by ARCH-003 (single-file,
class-name-gated AST) but FAILED by ARCH-004 (cross-file join).

All fixtures are synthetic, vendored mini node directories under ``tmp_path``;
no test depends on a sibling-repo path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.enums import EnumValidationSeverity
from omnibase_infra.nodes.node_architecture_validator.validators.scanner_imperative_orchestrator_ratchet import (
    BaselineEntry,
    discover_node_dirs,
    load_baseline,
    ratchet_violations,
    render_baseline_yaml,
    scan_node_dirs,
    strict_violations,
    write_baseline,
)
from omnibase_infra.nodes.node_architecture_validator.validators.validator_contract_declared_orchestrator_workflow import (
    EVENT_MARKER_REGEX,
    RULE_ID,
    RuleContractDeclaredOrchestratorWorkflow,
    analyze_node_directory,
    validate_contract_declared_orchestrator_workflow,
)
from omnibase_infra.nodes.node_architecture_validator.validators.validator_no_orchestrator_fsm import (
    validate_no_orchestrator_fsm,
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
    enums_source: str | None = None,
) -> Path:
    """Create a synthetic ``nodes/<node_name>/`` directory under ``root``.

    Returns the node directory path.
    """
    node_dir = root / "src" / "pkg" / "nodes" / node_name
    handlers_dir = node_dir / "handlers"
    handlers_dir.mkdir(parents=True, exist_ok=True)
    (node_dir / "contract.yaml").write_text(contract_yaml, encoding="utf-8")
    if handler_source is not None:
        (handlers_dir / f"{handler_name}.py").write_text(
            handler_source, encoding="utf-8"
        )
    if enums_source is not None:
        (node_dir / "enums.py").write_text(enums_source, encoding="utf-8")
    return node_dir


# Contract fragment: orchestrator with a decorative fsm: table (no executor bind).
_DELEGATION_SHAPE_CONTRACT = """\
name: node_synth_delegation
node_type: ORCHESTRATOR_GENERIC
fsm:
  states:
    - RECEIVED
    - ROUTED
    - EXECUTING
    - INFERENCE_COMPLETED
    - GATE_EVALUATED
    - ESCALATING
    - COMPLETED
    - FAILED
  initial_state: RECEIVED
  terminal_states:
    - COMPLETED
    - FAILED
  transitions:
    - from: RECEIVED
      to: ROUTED
      trigger: accepted
handler_routing:
  routing_strategy: payload_type_match
  handlers:
    - operation: orchestrate
      handler:
        name: HandlerDelegationWorkflow
        module: pkg.nodes.node_synth_delegation.handlers.handler_workflow
      event_model:
        name: ModelDelegationRequest
        module: pkg.models
"""

# Monolithic handler whose CLASS NAME is NOT *Orchestrator, driving state via
# self._transition(...) — the exact gap ARCH-003 misses. Also constructs a
# terminal event (H3).
_DELEGATION_SHAPE_HANDLER = '''\
"""Synthetic monolithic delegation workflow handler."""
from __future__ import annotations


class HandlerDelegationWorkflow:
    """Handler whose name lacks "Orchestrator" but owns an imperative FSM."""

    def _transition(self, workflow, new_state):
        workflow.state = new_state

    def handle(self, envelope):
        workflow = envelope.payload
        self._transition(workflow, "ROUTED")
        self._transition(workflow, "EXECUTING")
        self._transition(workflow, "COMPLETED")
        # H3: constructs a terminal event in the same handler that drives state.
        return ModelTaskDelegatedEvent(state=workflow.state)
'''


def test_arch003_passes_but_arch004_fails_delegation_shape(tmp_path: Path) -> None:
    """HEADLINE: ARCH-003 passes the delegation-shape handler; ARCH-004 fails it.

    (a) Synthetic node: contract fsm table + monolithic handler using
    ``self._transition(`` whose class is NOT named ``*Orchestrator``.
    """
    node_dir = _make_node_dir(
        tmp_path,
        node_name="node_synth_delegation",
        contract_yaml=_DELEGATION_SHAPE_CONTRACT,
        handler_source=_DELEGATION_SHAPE_HANDLER,
    )
    handler_file = node_dir / "handlers" / "handler_workflow.py"

    # ARCH-003 (single-file, class-name-gated AST) PASSES the handler:
    # class is HandlerDelegationWorkflow (no "Orchestrator"); the method match
    # set does not include the *call* self._transition(...).
    arch003 = validate_no_orchestrator_fsm(str(handler_file))
    assert arch003.valid, (
        "ARCH-003 is expected to PASS the delegation-shape handler (it cannot "
        "see _transition calls in a non-*Orchestrator class) — this is the gap."
    )
    assert len(arch003.violations) == 0

    # ARCH-004 (cross-file node-directory join) FAILS the node.
    arch004 = validate_contract_declared_orchestrator_workflow(str(node_dir))
    assert not arch004.valid, "ARCH-004 must FAIL the delegation-shape node."
    codes = {
        str(v.details["finding_code"])
        for v in arch004.violations
        if v.details and "finding_code" in v.details
    }
    assert "H1" in codes, "Expected H1 (decorative FSM + handler-driven transitions)."
    assert all(v.rule_id == RULE_ID for v in arch004.violations)
    # H3: one handler selects state and constructs an event.
    assert "H3" in codes, "Expected H3 (state selection + terminal event construction)."
    # This fixture has a single declared payload type, so H2 (fan-in >= 3) does
    # NOT fire here; the 3+-payload fan-in case is proven in the full-audit test.
    assert "H2" not in codes


def test_payload_type_match_three_plus_payloads_fails(tmp_path: Path) -> None:
    """(b) payload_type_match routing 3+ payloads to one workflow handler -> fail."""
    contract = """\
name: node_synth_fanin
node_type: ORCHESTRATOR_GENERIC
handler_routing:
  routing_strategy: payload_type_match
  handlers:
    - operation: a
      handler:
        name: HandlerWorkflow
        module: pkg.nodes.node_synth_fanin.handlers.handler_workflow
      event_model:
        name: ModelEventA
        module: pkg.models
    - operation: b
      handler:
        name: HandlerWorkflow
        module: pkg.nodes.node_synth_fanin.handlers.handler_workflow
      event_model:
        name: ModelEventB
        module: pkg.models
    - operation: c
      handler:
        name: HandlerWorkflow
        module: pkg.nodes.node_synth_fanin.handlers.handler_workflow
      event_model:
        name: ModelEventC
        module: pkg.models
"""
    handler = '''\
"""Synthetic catchall handler (no FSM, no event construction)."""


class HandlerWorkflow:
    def handle(self, envelope):
        return envelope
'''
    node_dir = _make_node_dir(
        tmp_path,
        node_name="node_synth_fanin",
        contract_yaml=contract,
        handler_source=handler,
    )
    result = validate_contract_declared_orchestrator_workflow(str(node_dir))
    assert not result.valid, "payload_type_match with 3 payloads must hard-fail."
    codes = {
        str(v.details["finding_code"])
        for v in result.violations
        if v.details and "finding_code" in v.details
    }
    assert "H2" in codes
    # No fsm + no handler-driven transitions => no H1.
    assert "H1" not in codes


def test_reducer_with_state_machine_passes(tmp_path: Path) -> None:
    """(c) reducer with state_machine: + pure transition executor -> pass (exempt)."""
    contract = """\
name: node_synth_fsm_reducer
node_type: REDUCER_GENERIC
state_machine:
  states:
    - CREATED
    - PROCESSING
    - DONE
  transitions:
    - from: CREATED
      to: PROCESSING
    - from: PROCESSING
      to: DONE
"""
    handler = '''\
"""Pure transition executor for a reducer (legitimately owns the FSM)."""


class ReducerStateMachine:
    def reduce(self, state, event):
        # Pure: returns next state, no I/O.
        return self._transition(state, event)

    def _transition(self, state, event):
        return state
'''
    node_dir = _make_node_dir(
        tmp_path,
        node_name="node_synth_fsm_reducer",
        contract_yaml=contract,
        handler_source=handler,
    )
    # Reducers are EXEMPT: analysis returns None (out of scope).
    analysis = analyze_node_directory(node_dir)
    assert analysis is None, "Reducers must be exempt from ARCH-004."
    result = validate_contract_declared_orchestrator_workflow(str(node_dir))
    assert result.valid, "Reducer with state_machine must PASS (exempt)."
    assert len(result.violations) == 0


def test_executor_bound_orchestrator_passes(tmp_path: Path) -> None:
    """(d) executor-bound orchestrator workflow DAG + thin handlers -> pass."""
    contract = """\
name: node_synth_bound_orchestrator
node_type: ORCHESTRATOR_GENERIC
workflow_coordination:
  execution_graph:
    - step: route
      next: infer
    - step: infer
      next: emit
handler_routing:
  routing_strategy: operation_match
  handlers:
    - operation: route
      handler:
        name: HandlerRouteStep
        module: pkg.nodes.node_synth_bound_orchestrator.handlers.handler_route
"""
    handler = '''\
"""Thin per-step handler; no FSM ownership, no terminal event construction."""


class HandlerRouteStep:
    def handle(self, envelope):
        return envelope
'''
    node_dir = _make_node_dir(
        tmp_path,
        node_name="node_synth_bound_orchestrator",
        contract_yaml=contract,
        handler_name="handler_route",
        handler_source=handler,
    )
    result = validate_contract_declared_orchestrator_workflow(str(node_dir))
    assert result.valid, (
        "Executor-bound orchestrator with thin handlers must PASS ARCH-004."
    )
    assert len(result.violations) == 0


def test_full_audit_detects_delegation_shaped_fixture(tmp_path: Path) -> None:
    """(e) full-audit mode detects a delegation-shaped fixture.

    Mirrors the real contract.yaml: fsm: table + payload_type_match (3+ payloads)
    + a handler using self._transition(...).
    """
    contract = _DELEGATION_SHAPE_CONTRACT.replace(
        """\
    - operation: orchestrate
      handler:
        name: HandlerDelegationWorkflow
        module: pkg.nodes.node_synth_delegation.handlers.handler_workflow
      event_model:
        name: ModelDelegationRequest
        module: pkg.models
""",
        """\
    - operation: orchestrate
      handler:
        name: HandlerDelegationWorkflow
        module: pkg.nodes.node_synth_delegation.handlers.handler_workflow
      event_model:
        name: ModelDelegationRequest
        module: pkg.models
    - operation: lifecycle
      handler:
        name: HandlerDelegationWorkflow
        module: pkg.nodes.node_synth_delegation.handlers.handler_workflow
      event_model:
        name: ModelAgentLifecycle
        module: pkg.models
    - operation: gate
      handler:
        name: HandlerDelegationWorkflow
        module: pkg.nodes.node_synth_delegation.handlers.handler_workflow
      event_model:
        name: ModelGateResult
        module: pkg.models
""",
    )
    _make_node_dir(
        tmp_path,
        node_name="node_synth_delegation",
        contract_yaml=contract,
        handler_source=_DELEGATION_SHAPE_HANDLER,
    )
    # A clean orchestrator alongside, to prove the scan is selective.
    _make_node_dir(
        tmp_path,
        node_name="node_synth_clean_orchestrator",
        contract_yaml=(
            "name: node_synth_clean_orchestrator\n"
            "node_type: ORCHESTRATOR_GENERIC\n"
            "handler_routing:\n"
            "  routing_strategy: operation_match\n"
            "  handlers:\n"
            "    - operation: go\n"
            "      handler:\n"
            "        name: HandlerGo\n"
            "        module: pkg.nodes.node_synth_clean_orchestrator.handlers.handler_go\n"
        ),
        handler_name="handler_go",
        handler_source="class HandlerGo:\n    def handle(self, e):\n        return e\n",
    )

    node_dirs = discover_node_dirs(tmp_path)
    result = scan_node_dirs("synthetic", node_dirs, repo_root=tmp_path)
    hard_fail_nodes = {e.node for e in result.hard_fails}
    assert "node_synth_delegation" in hard_fail_nodes, (
        "Full-audit must detect the delegation-shaped node."
    )
    assert "node_synth_clean_orchestrator" not in hard_fail_nodes, (
        "Full-audit must NOT flag the clean operation_match orchestrator."
    )
    deleg = next(e for e in result.hard_fails if e.node == "node_synth_delegation")
    assert {"H1", "H2", "H3"}.issubset(set(deleg.finding_codes))
    # Repo-relative handler path (never a machine-absolute path).
    assert not deleg.max_handler_path.startswith("/")


# --------------------------------------------------------------------------- #
# Enum-state-match detection (the leading-underscore-free FSM)
# --------------------------------------------------------------------------- #
def test_enum_state_match_drives_h1(tmp_path: Path) -> None:
    """An Enum*State mirroring contract states (no _transition calls) -> H1."""
    contract = """\
name: node_synth_enum_fsm
node_type: ORCHESTRATOR_GENERIC
fsm:
  states:
    - IDLE
    - RUNNING
    - DONE
    - FAILED
handler_routing:
  routing_strategy: operation_match
  handlers:
    - operation: go
      handler:
        name: HandlerEnumFsm
        module: pkg.nodes.node_synth_enum_fsm.handlers.handler_enum
"""
    handler = '''\
"""Handler that owns an Enum*State mirroring the contract states."""
from enum import StrEnum


class EnumWorkflowState(StrEnum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


class HandlerEnumFsm:
    def handle(self, envelope):
        state = EnumWorkflowState.IDLE
        return state
'''
    node_dir = _make_node_dir(
        tmp_path,
        node_name="node_synth_enum_fsm",
        contract_yaml=contract,
        handler_name="handler_enum",
        handler_source=handler,
    )
    analysis = analyze_node_directory(node_dir)
    assert analysis is not None
    assert analysis.enum_state_match, "Enum*State must match the contract states."
    assert analysis.handler_drives_state
    assert "H1" in analysis.finding_codes


def test_substring_trap_not_a_false_positive(tmp_path: Path) -> None:
    """``record_phase_transition(`` and a ``current_state=%s`` log must NOT match.

    This is the precision check that distinguishes ARCH-004 from a naive
    substring scanner (the audit flagged pr_lifecycle's ``_transition`` as 0*).
    """
    contract = """\
name: node_synth_logonly
node_type: ORCHESTRATOR_GENERIC
fsm:
  states:
    - A
    - B
handler_routing:
  routing_strategy: operation_match
  handlers:
    - operation: go
      handler:
        name: HandlerLogOnly
        module: pkg.nodes.node_synth_logonly.handlers.handler_log
"""
    handler = '''\
"""Handler that only LOGS transition-ish strings; does not drive state."""
import logging

logger = logging.getLogger(__name__)


def record_phase_transition(phase):
    return phase


class HandlerLogOnly:
    def handle(self, envelope):
        logger.info("current_state=%s", envelope)
        record_phase_transition("done")
        return envelope
'''
    node_dir = _make_node_dir(
        tmp_path,
        node_name="node_synth_logonly",
        contract_yaml=contract,
        handler_name="handler_log",
        handler_source=handler,
    )
    analysis = analyze_node_directory(node_dir)
    assert analysis is not None
    assert not analysis.handler_drives_state, (
        "record_phase_transition( and a current_state=%s log string must NOT "
        "register as handler-driven state transitions."
    )
    # 2 fsm states + a log-only handler => no H1 hard fail (only the W4 warning).
    assert "H1" not in analysis.finding_codes
    assert not analysis.has_hard_fail


# --------------------------------------------------------------------------- #
# Protocol-rule surface
# --------------------------------------------------------------------------- #
def test_rule_class_protocol_surface(tmp_path: Path) -> None:
    """RuleContractDeclaredOrchestratorWorkflow implements the rule protocol."""
    rule = RuleContractDeclaredOrchestratorWorkflow()
    assert rule.rule_id == "ARCH-004"
    assert rule.severity == EnumValidationSeverity.ERROR
    assert rule.name
    assert rule.description

    # Non-path target => skipped, not a violation.
    skipped = rule.check(object())
    assert skipped.skipped is True
    assert skipped.passed is True

    # contract.yaml path resolves to the parent node dir.
    node_dir = _make_node_dir(
        tmp_path,
        node_name="node_synth_delegation",
        contract_yaml=_DELEGATION_SHAPE_CONTRACT,
        handler_source=_DELEGATION_SHAPE_HANDLER,
    )
    res = rule.check(str(node_dir / "contract.yaml"))
    assert res.passed is False
    assert res.details is not None
    assert "H1" in (res.details.get("finding_codes") or [])


def test_event_marker_regex_is_reported(tmp_path: Path) -> None:
    """The exact event-marker regex is surfaced in violation details (auditability)."""
    node_dir = _make_node_dir(
        tmp_path,
        node_name="node_synth_delegation",
        contract_yaml=_DELEGATION_SHAPE_CONTRACT,
        handler_source=_DELEGATION_SHAPE_HANDLER,
    )
    result = validate_contract_declared_orchestrator_workflow(str(node_dir))
    assert any(
        v.details and v.details.get("event_marker_regex") == EVENT_MARKER_REGEX
        for v in result.violations
    )


# --------------------------------------------------------------------------- #
# Ratchet behaviour
# --------------------------------------------------------------------------- #
def _delegation_entry() -> BaselineEntry:
    return BaselineEntry(
        repo="synthetic",
        node="node_synth_delegation",
        max_handler_path="src/pkg/nodes/node_synth_delegation/handlers/handler_workflow.py",
        line_count=20,
        risk_score=8,
        finding_codes=("H1", "H2", "H3"),
        owner_ticket="OMN-13471",
    )


def test_ratchet_passes_when_node_matches_baseline() -> None:
    """A scanned hard-fail already in the baseline (unchanged) passes the ratchet."""
    entry = _delegation_entry()
    baseline = {f"{entry.repo}::{entry.node}": entry}
    assert ratchet_violations([entry], baseline) == []


def test_ratchet_fails_on_new_untracked_node() -> None:
    """A scanned hard-fail not in the baseline fails the ratchet."""
    entry = _delegation_entry()
    failures = ratchet_violations([entry], baseline={})
    assert len(failures) == 1
    assert "NEW imperative-orchestrator hard-fail" in failures[0]


def test_ratchet_fails_on_worsened_risk_and_growth() -> None:
    """A scanned hard-fail that worsens risk / grows the handler fails the ratchet."""
    base = _delegation_entry()
    worse = BaselineEntry(
        repo=base.repo,
        node=base.node,
        max_handler_path=base.max_handler_path,
        line_count=base.line_count + 500,
        risk_score=base.risk_score + 2,
        finding_codes=base.finding_codes + ("W1",),
        owner_ticket=base.owner_ticket,
    )
    failures = ratchet_violations([worse], baseline={f"{base.repo}::{base.node}": base})
    joined = "\n".join(failures)
    assert "risk score WORSENED" in joined
    assert "max handler GREW" in joined
    assert "NEW finding codes" in joined


def test_ratchet_fails_when_baseline_entry_has_no_owner_ticket() -> None:
    """Every accepted hard-fail must cite an owner ticket."""
    base = BaselineEntry(
        repo="synthetic",
        node="node_synth_delegation",
        max_handler_path="x.py",
        line_count=20,
        risk_score=8,
        finding_codes=("H1",),
        owner_ticket="",
    )
    failures = ratchet_violations([base], baseline={f"{base.repo}::{base.node}": base})
    assert any("no owner_ticket" in f for f in failures)


def test_strict_mode_fails_on_still_baselined_node() -> None:
    """--strict: a baselined node that still hard-fails must be remediated."""
    entry = _delegation_entry()
    baseline = {f"{entry.repo}::{entry.node}": entry}
    failures = strict_violations([entry], baseline)
    assert len(failures) == 1
    assert "still hard-fails ARCH-004 but is baselined" in failures[0]


def test_baseline_roundtrip(tmp_path: Path) -> None:
    """render -> write -> load round-trips entries, with no absolute paths."""
    entry = _delegation_entry()
    rendered = render_baseline_yaml("omnibase_infra", [entry])
    assert "/Users/" not in rendered and "/Volumes/" not in rendered
    baseline_path = tmp_path / "architecture-handshakes" / "baseline.yaml"
    write_baseline(baseline_path, "omnibase_infra", [entry])
    loaded = load_baseline(baseline_path)
    key = f"{entry.repo}::{entry.node}"
    assert key in loaded
    assert loaded[key].finding_codes == entry.finding_codes
    assert loaded[key].owner_ticket == "OMN-13471"
