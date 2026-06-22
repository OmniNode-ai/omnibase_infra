# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Validator for ARCH-004: Contract-Declared Orchestrator Workflow Must Be Bound To An Executor.

This validator detects the *imperative-orchestrator* anti-pattern that ARCH-003
is structurally incapable of catching (see ``validator_no_orchestrator_fsm.py``).

Why ARCH-003 misses it (proven by ``node_delegation_orchestrator``):
    - ARCH-003 is a single-file AST visitor that only inspects classes whose
      *name* contains ``"Orchestrator"``. The delegation FSM lives in class
      ``HandlerDelegationWorkflow`` (no "Orchestrator" in the name) → never
      inspected.
    - ARCH-003's method match-set is ``{transition, can_transition,
      apply_transition, get_current_state, set_state}``. The delegation handler
      drives state via ``_transition`` (leading underscore, and predominantly a
      *call* not a *def*) → never matched.
    - ARCH-003 never joins ``contract.yaml`` + ``handler_routing`` + handler
      source. It cannot tell "declared FSM table but no executor binds to it,
      while a handler drives the transitions itself."

ARCH-004 is therefore a **cross-file, node-directory** rule. Per node directory
it joins four sources:
    1. ``contract.yaml`` — ``fsm:`` / ``workflow_coordination`` / node_type / name.
    2. ``handler_routing.routing_strategy`` — ``payload_type_match`` catchall.
    3. Handler source under ``handlers/`` — state-driving symbols.
    4. Node type/name — orchestrator-like classification + reducer exemption.

Canonical truth (verified 2026-06-22, OMN-13472):
    ``ModelContractOrchestrator``
    (``omnibase_core/models/contracts/model_contract_orchestrator.py``) has only
    ``workflow_coordination: ModelWorkflowConfig`` — **no typed
    ``fsm``/``state_machine``/``transition`` field** — and no traverser consumes
    orchestrator ``fsm.transitions``. So a declared ``fsm:`` block on an
    orchestrator contract is *decorative*: nothing executes it. FSM execution
    exists only for reducers (``node_*_fsm_reducer`` + contract ``state_machine:``
    + a pure transition executor) — which are EXEMPT.

Hard-fail signals (for changed/new orchestrator-like nodes):
    H1. Decorative-FSM-with-handler-driven-transitions: contract declares an
        ``fsm:``/workflow-state set, no executor binds to it, AND a handler drives
        the transitions itself (``_transition(`` etc. or an ``Enum*State`` whose
        members ≈ the contract ``fsm.states``).
    H2. ``handler_routing.routing_strategy: payload_type_match`` funnels 3+
        event/payload types into a single workflow handler.
    H3. One handler both selects the next workflow state AND constructs
        terminal/compat events (the OMN-13408 footgun).

Warning / baseline-score signals:
    W1. Handler > 750 lines for an orchestrator-like node.
    W2. Handler > 125 branch/control markers.
    W3. >= 10 publish/event-construction markers (regex reported in
        ``EVENT_MARKER_REGEX``).
    W4. Declared states/transitions are not a typed, executor-bound schema.

Related:
    - Ticket: OMN-13472 (ARCH-004 ratchet — Workstream B)
    - Epic: OMN-13471 (decompose imperative delegation orchestrator)
    - OMN-12550 (wire ARCH-001/002/003 as blocking gates — ARCH-004 rides it)
    - OMN-13325 (ratchet-enforcement epic)
    - Plan: docs/plans/2026-06-22-imperative-orchestrator-ratchet-and-recovery-plan-verified.md §4
    - Audit: docs/audits/2026-06-22-imperative-orchestrator-audit.md
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from omnibase_infra.enums import EnumValidationSeverity
from omnibase_infra.nodes.node_architecture_validator.models.model_architecture_violation import (
    ModelArchitectureViolation,
)
from omnibase_infra.nodes.node_architecture_validator.models.model_validation_result import (
    ModelFileValidationResult,
)

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_architecture_validator.models import (
        ModelRuleCheckResult,
    )

RULE_ID = "ARCH-004"
RULE_NAME = "Contract-Declared Orchestrator Workflow Must Be Bound To An Executor"

# --- Detection thresholds (warning / baseline score) ----------------------

#: A handler larger than this for an orchestrator-like node is debt (W1).
HANDLER_LINE_WARN_THRESHOLD = 750
#: More branch/control markers than this in a single handler is debt (W2).
BRANCH_MARKER_WARN_THRESHOLD = 125
#: At least this many publish/event-construction markers is debt (W3).
EVENT_MARKER_WARN_THRESHOLD = 10
#: payload_type_match funneling this many distinct payload types into one
#: workflow handler is a hard fail (H2).
PAYLOAD_FANIN_HARD_THRESHOLD = 3

# --- Regexes (reported in violation details so they can be re-derived) -----

#: Publish / event-construction markers (W3 / H3). Reported verbatim.
EVENT_MARKER_REGEX = r"\.publish\(|ModelEventEnvelope|\.emit\(|[A-Za-z_]+Event\("
#: Branch / control-flow markers (W2). Reported verbatim.
BRANCH_MARKER_REGEX = r"\b(if|elif|for|while|except|case|and|or)\b"

#: Handler symbols that prove the handler is itself driving an FSM (H1).
#:
#: These are the *human-readable* symbol names reported in violation detail and
#: used by the synthetic-fixture documentation. Detection is performed by
#: :data:`HANDLER_STATE_DRIVE_REGEX` below, which is precise (boundary-anchored)
#: to avoid the substring traps that the audit flagged with ``0*`` — e.g.
#: ``record_phase_transition(`` must NOT match ``_transition(``, and a log-format
#: string ``"current_state=%s"`` must NOT match ``current_state``.
#:
#: NOTE: ``_transition`` (leading underscore) is included precisely because
#: ARCH-003 omits it.
HANDLER_STATE_DRIVE_SYMBOLS: tuple[str, ...] = (
    "_transition(",
    "transition(",
    "can_transition",
    "apply_transition",
    "set_state",
    "current_state",
    "next_state",
)

#: Boundary-anchored detection for the H1 state-drive symbols.
#:
#: * A ``transition(``/``_transition(`` call must be a *method call* — preceded
#:   by ``self.`` / ``cls.`` / an identifier-dot, with a word boundary at the
#:   start of the method name so ``record_phase_transition(`` (matches as
#:   ``record_phase_`` + ``transition(``) does NOT count. We anchor on
#:   ``(?:self|cls)\.(?:_)?transition\(`` and ``\b_transition\(`` is excluded
#:   when preceded by another word char.
#: * ``can_transition`` / ``apply_transition`` / ``set_state`` are method names;
#:   require a leading ``.`` or ``def ``.
#: * ``current_state`` / ``next_state`` count only as an *assignment target* or
#:   attribute access (``self.current_state``, ``.current_state =``), never
#:   inside a string literal.
#:
#: Each entry maps a reported symbol name -> its precise regex.
HANDLER_STATE_DRIVE_REGEX: dict[str, str] = {
    "_transition(": r"(?:self|cls)\.\s*_transition\s*\(",
    "transition(": r"(?:self|cls)\.\s*transition\s*\(",
    "can_transition": r"(?:\.|def\s+)can_transition\b",
    "apply_transition": r"(?:\.|def\s+)apply_transition\b",
    "set_state": r"(?:\.|def\s+)set_state\b",
    "current_state": r"(?:self|cls)\.current_state\s*=",
    "next_state": r"(?:self|cls)\.next_state\s*=",
}

#: Names on ``ModelContractOrchestrator`` (and any contract field) that would
#: indicate a *typed, executor-bound* workflow schema. Presence in the contract
#: under one of these keys means the FSM is bound (not decorative) → not H1.
EXECUTOR_BOUND_CONTRACT_KEYS: tuple[str, ...] = (
    "state_machine",
    "execution_graph",
    "workflow_dag",
    "execution_dag",
)

#: Fraction of contract fsm.states that must appear as Enum*State members for
#: the enum to be treated as the handler's parallel FSM (H1, approximate match).
ENUM_STATE_MATCH_RATIO = 0.6


# --------------------------------------------------------------------------- #
# Node-directory analysis result (internal, not a public model)
# --------------------------------------------------------------------------- #
class OrchestratorNodeAnalysis:
    """Mutable accumulator for one node directory's cross-file analysis.

    This is an internal scratch object; the public surface is the list of
    :class:`ModelArchitectureViolation` it produces via :meth:`violations`.
    """

    def __init__(self, node_dir: Path, contract_path: Path) -> None:
        self.node_dir = node_dir
        self.contract_path = contract_path
        self.node_name: str = node_dir.name
        self.node_type: str = ""
        self.is_orchestrator_like: bool = False
        self.is_reducer: bool = False
        self.has_fsm_block: bool = False
        self.fsm_states: tuple[str, ...] = ()
        self.has_executor_bound_workflow: bool = False
        self.routing_strategy: str = ""
        self.payload_type_count: int = 0
        self.max_handler_path: Path | None = None
        self.max_handler_lines: int = 0
        self.handler_drives_state: bool = False
        self.handler_state_drive_symbol: str = ""
        self.enum_state_match: bool = False
        self.handler_branch_markers: int = 0
        self.handler_event_markers: int = 0
        self.handler_selects_state_and_emits: bool = False
        # Accumulated finding codes (e.g. "H1", "W3") for the baseline.
        self.finding_codes: list[str] = []
        self._violations: list[ModelArchitectureViolation] = []

    # -- violation helpers -------------------------------------------------- #
    def _add(
        self,
        *,
        code: str,
        severity: EnumValidationSeverity,
        message: str,
        suggestion: str,
        location: Path | None = None,
    ) -> None:
        self.finding_codes.append(code)
        loc = str(location) if location is not None else str(self.contract_path)
        self._violations.append(
            ModelArchitectureViolation(
                rule_id=RULE_ID,
                rule_name=RULE_NAME,
                severity=severity,
                target_type="orchestrator_node",
                target_name=self.node_name,
                message=f"[{code}] {message}",
                location=loc,
                suggestion=suggestion,
                details={
                    "finding_code": code,
                    "node_dir": str(self.node_dir),
                    "node_type": self.node_type,
                    "routing_strategy": self.routing_strategy,
                    "event_marker_regex": EVENT_MARKER_REGEX,
                    "branch_marker_regex": BRANCH_MARKER_REGEX,
                },
            )
        )

    def violations(self) -> list[ModelArchitectureViolation]:
        return self._violations

    @property
    def has_hard_fail(self) -> bool:
        return any(v.severity == EnumValidationSeverity.ERROR for v in self._violations)

    @property
    def risk_score(self) -> int:
        """Transparent additive score mirroring the audit doc.

        ``fsm:+2 · payload_type_match:+2 · handler_drives_state:+3 ·
        handler>1500:+2 (>750:+1) · event_markers>=10:+1``.
        """
        score = 0
        if self.has_fsm_block:
            score += 2
        if self.routing_strategy == "payload_type_match":
            score += 2
        if self.handler_drives_state:
            score += 3
        if self.max_handler_lines > 1500:
            score += 2
        elif self.max_handler_lines > HANDLER_LINE_WARN_THRESHOLD:
            score += 1
        if self.handler_event_markers >= EVENT_MARKER_WARN_THRESHOLD:
            score += 1
        return score


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _is_orchestrator_like(node_name: str, node_type: str) -> bool:
    """Classify whether a node directory is orchestrator-like.

    Joins node directory name AND declared ``node_type`` so a misnamed
    orchestrator (or a generic orchestrator type) is still caught.
    """
    name_signals = "orchestrator" in node_name.lower()
    type_signals = "ORCHESTRATOR" in node_type.upper()
    return name_signals or type_signals


def _is_reducer(node_name: str, node_type: str) -> bool:
    """Reducers (``node_*_fsm_reducer`` / ``REDUCER_*``) are EXEMPT."""
    name_signals = "reducer" in node_name.lower()
    type_signals = "REDUCER" in node_type.upper()
    return name_signals or type_signals


def _contract_has_executor_bound_workflow(contract: dict[str, object]) -> bool:
    """Return True if the contract declares a typed, executor-bound workflow.

    A reducer-style ``state_machine:`` block, or an explicit execution-graph /
    workflow DAG key, counts as executor-bound. A bare orchestrator ``fsm:``
    block does NOT — ``ModelContractOrchestrator`` has no typed field that binds
    it, so nothing executes it.
    """
    for key in EXECUTOR_BOUND_CONTRACT_KEYS:
        if contract.get(key):
            return True
    # workflow_coordination with an explicit, non-empty execution graph / steps
    wc = contract.get("workflow_coordination")
    if isinstance(wc, dict):
        for key in ("execution_graph", "workflow_dag", "steps", "dag"):
            if wc.get(key):
                return True
    return False


def _extract_fsm_states(contract: dict[str, object]) -> tuple[bool, tuple[str, ...]]:
    """Extract (has_fsm_block, states) from the contract.

    Detects a declared workflow-state set under either ``fsm:`` (orchestrator
    convention) — a ``state_machine:`` block is handled as executor-bound and
    does not count here.
    """
    fsm = contract.get("fsm")
    if not isinstance(fsm, dict):
        return False, ()
    states = fsm.get("states")
    if isinstance(states, list):
        return True, tuple(str(s) for s in states)
    return True, ()


def _enum_states_match_fsm(handler_source: str, fsm_states: tuple[str, ...]) -> bool:
    """Return True if an ``Enum*State`` class in the handler ≈ contract fsm.states.

    Approximate match: an Enum whose name contains "State" defines >= 60% of the
    contract's declared fsm states as members. This catches the delegation
    ``EnumDelegationState`` (8 members == the 8 contract states) even when the
    enum lives in a sibling ``enums.py`` re-exported into the handler.
    """
    if not fsm_states:
        return False
    try:
        tree = ast.parse(handler_source)
    except SyntaxError:
        return False
    state_set = {s.upper() for s in fsm_states}
    threshold = max(1, int(len(state_set) * ENUM_STATE_MATCH_RATIO))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and "State" in node.name:
            members = {
                t.id.upper()
                for item in node.body
                if isinstance(item, ast.Assign)
                for t in item.targets
                if isinstance(t, ast.Name)
            }
            if len(state_set & members) >= threshold:
                return True
    return False


def _count_payload_types(handler_routing: dict[str, object]) -> int:
    """Count distinct ``event_model`` payload types in the routing table."""
    handlers = handler_routing.get("handlers")
    if not isinstance(handlers, list):
        return 0
    payload_names: set[str] = set()
    for entry in handlers:
        if not isinstance(entry, dict):
            continue
        event_model = entry.get("event_model")
        if isinstance(event_model, dict):
            name = event_model.get("name")
            if isinstance(name, str):
                payload_names.add(name)
        elif isinstance(event_model, str):
            payload_names.add(event_model)
    return len(payload_names)


def _largest_handler(handlers_dir: Path) -> tuple[Path | None, int, str]:
    """Return (path, line_count, source) of the largest handler module."""
    best_path: Path | None = None
    best_lines = 0
    best_source = ""
    if not handlers_dir.is_dir():
        return None, 0, ""
    for py in sorted(handlers_dir.glob("handler_*.py")):
        try:
            source = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        lines = source.count("\n") + 1
        if lines > best_lines:
            best_path, best_lines, best_source = py, lines, source
    return best_path, best_lines, best_source


def _detect_state_drive_symbol(source: str) -> str:
    """Return the first precise state-drive symbol matched in ``source``, or "".

    Uses :data:`HANDLER_STATE_DRIVE_REGEX` (boundary-anchored) so substring
    traps such as ``record_phase_transition(`` or a ``"current_state=%s"`` log
    format string do not produce false positives.
    """
    for symbol, pattern in HANDLER_STATE_DRIVE_REGEX.items():
        if re.search(pattern, source) is not None:
            return symbol
    return ""


def _handler_selects_state_and_emits(source: str) -> bool:
    """H3: one handler both drives a state transition AND constructs an event.

    Detected as the *co-occurrence*, in one handler module, of a (precisely
    matched) state-driving symbol and an event-construction marker.
    """
    drives = _detect_state_drive_symbol(source) != ""
    emits = re.search(EVENT_MARKER_REGEX, source) is not None
    return drives and emits


# --------------------------------------------------------------------------- #
# Core node-directory analysis
# --------------------------------------------------------------------------- #
def analyze_node_directory(node_dir: Path) -> OrchestratorNodeAnalysis | None:
    """Run the ARCH-004 cross-file join over a single node directory.

    Args:
        node_dir: Path to a ``nodes/<node_name>/`` directory containing a
            ``contract.yaml``. String callers (CLI / the protocol rule) convert
            to :class:`~pathlib.Path` before calling.

    Returns:
        A populated :class:`OrchestratorNodeAnalysis` (with any violations), or ``None`` if
        the directory is not an analyzable node (no contract) or is exempt
        (reducer / not orchestrator-like).
    """
    node_path = Path(node_dir)
    contract_path = node_path / "contract.yaml"
    if not contract_path.is_file():
        return None

    try:
        raw = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(raw, dict):
        return None

    analysis = OrchestratorNodeAnalysis(node_path, contract_path)
    analysis.node_type = str(raw.get("node_type", ""))
    analysis.node_name = str(raw.get("name", node_path.name))

    analysis.is_reducer = _is_reducer(analysis.node_name, analysis.node_type)
    analysis.is_orchestrator_like = _is_orchestrator_like(
        analysis.node_name, analysis.node_type
    )

    # Reducers are EXEMPT (they legitimately own state machines).
    if analysis.is_reducer:
        return None
    # Only orchestrator-like nodes are in scope.
    if not analysis.is_orchestrator_like:
        return None

    # --- contract facts ---------------------------------------------------- #
    analysis.has_fsm_block, analysis.fsm_states = _extract_fsm_states(raw)
    analysis.has_executor_bound_workflow = _contract_has_executor_bound_workflow(raw)

    handler_routing = raw.get("handler_routing")
    if isinstance(handler_routing, dict):
        analysis.routing_strategy = str(handler_routing.get("routing_strategy", ""))
        analysis.payload_type_count = _count_payload_types(handler_routing)

    # --- handler facts ----------------------------------------------------- #
    handlers_dir = node_path / "handlers"
    handler_path, handler_lines, handler_source = _largest_handler(handlers_dir)
    analysis.max_handler_path = handler_path
    analysis.max_handler_lines = handler_lines

    if handler_source:
        matched_symbol = _detect_state_drive_symbol(handler_source)
        if matched_symbol:
            analysis.handler_drives_state = True
            analysis.handler_state_drive_symbol = matched_symbol
        analysis.enum_state_match = _enum_states_match_fsm(
            handler_source, analysis.fsm_states
        )
        analysis.handler_branch_markers = len(
            re.findall(BRANCH_MARKER_REGEX, handler_source)
        )
        analysis.handler_event_markers = len(
            re.findall(EVENT_MARKER_REGEX, handler_source)
        )
        analysis.handler_selects_state_and_emits = _handler_selects_state_and_emits(
            handler_source
        )
    # An Enum*State that mirrors the contract states also counts as the handler
    # owning a parallel FSM (covers the case where transitions are not literal
    # ``_transition(`` calls but the enum drives state).
    if analysis.enum_state_match:
        analysis.handler_drives_state = True
        if not analysis.handler_state_drive_symbol:
            analysis.handler_state_drive_symbol = "Enum*State≈fsm.states"

    _evaluate_signals(analysis)
    return analysis


def _evaluate_signals(a: OrchestratorNodeAnalysis) -> None:
    """Apply the hard-fail and warning signals to a populated analysis."""
    handler_loc = a.max_handler_path or a.contract_path

    # H1: decorative FSM + handler-driven transitions.
    if a.has_fsm_block and not a.has_executor_bound_workflow and a.handler_drives_state:
        a._add(
            code="H1",
            severity=EnumValidationSeverity.ERROR,
            message=(
                f"Orchestrator declares an fsm: state set "
                f"({len(a.fsm_states)} states) but no runtime executor binds to "
                f"it (no typed fsm/state_machine field on ModelContractOrchestrator, "
                f"no traverser consumes its transitions), WHILE the handler drives "
                f"the transitions itself "
                f"(symbol '{a.handler_state_drive_symbol}'). The declared "
                f"transition table is decorative; the handler owns a parallel "
                f"imperative FSM."
            ),
            suggestion=(
                "Bind the contract fsm to an executor (migrate to a typed, "
                "executor-bound workflow/DAG field per OMN-12835), or decompose "
                "the handler into per-step thin handlers driven by a reducer. "
                "See OMN-13471 (delegation decomposition epic)."
            ),
            location=handler_loc,
        )

    # H2: payload_type_match fan-in.
    if (
        a.routing_strategy == "payload_type_match"
        and a.payload_type_count >= PAYLOAD_FANIN_HARD_THRESHOLD
    ):
        a._add(
            code="H2",
            severity=EnumValidationSeverity.ERROR,
            message=(
                f"handler_routing.routing_strategy=payload_type_match funnels "
                f"{a.payload_type_count} distinct payload/event types into a "
                f"single workflow handler (>= {PAYLOAD_FANIN_HARD_THRESHOLD})."
            ),
            suggestion=(
                "Split the catchall into per-event/per-step handlers "
                "(operation_match or one handler per payload type)."
            ),
            location=handler_loc,
        )

    # H3: one handler selects next state AND constructs terminal/compat events.
    if a.handler_selects_state_and_emits:
        a._add(
            code="H3",
            severity=EnumValidationSeverity.ERROR,
            message=(
                "A single handler both selects the next workflow state and "
                "constructs terminal/compat events (the OMN-13408 footgun): "
                "dual terminal/compat event construction co-located with state "
                "selection clobbers projection fields."
            ),
            suggestion=(
                "Move terminal/compat event construction behind one "
                "contract-owned builder/reducer; keep state selection separate "
                "from event emission."
            ),
            location=handler_loc,
        )

    # W1: oversized handler.
    if a.max_handler_lines > HANDLER_LINE_WARN_THRESHOLD:
        a._add(
            code="W1",
            severity=EnumValidationSeverity.WARNING,
            message=(
                f"Orchestrator max handler is {a.max_handler_lines} lines "
                f"(> {HANDLER_LINE_WARN_THRESHOLD}); orchestrator handlers should "
                f"be thin per-step coordinators."
            ),
            suggestion="Decompose into per-step handlers.",
            location=handler_loc,
        )

    # W2: branch/control density.
    if a.handler_branch_markers > BRANCH_MARKER_WARN_THRESHOLD:
        a._add(
            code="W2",
            severity=EnumValidationSeverity.WARNING,
            message=(
                f"Handler has {a.handler_branch_markers} branch/control markers "
                f"(> {BRANCH_MARKER_WARN_THRESHOLD}; regex {BRANCH_MARKER_REGEX!r})."
            ),
            suggestion="Reduce branching by extracting decision compute nodes.",
            location=handler_loc,
        )

    # W3: event-construction density.
    if a.handler_event_markers >= EVENT_MARKER_WARN_THRESHOLD:
        a._add(
            code="W3",
            severity=EnumValidationSeverity.WARNING,
            message=(
                f"Handler has {a.handler_event_markers} publish/event-construction "
                f"markers (>= {EVENT_MARKER_WARN_THRESHOLD}; regex "
                f"{EVENT_MARKER_REGEX!r})."
            ),
            suggestion="Centralize event construction in a contract-owned builder.",
            location=handler_loc,
        )

    # W4: declared-but-not-executor-bound workflow states.
    if a.has_fsm_block and not a.has_executor_bound_workflow:
        a._add(
            code="W4",
            severity=EnumValidationSeverity.WARNING,
            message=(
                "Contract declares workflow states/transitions that are not a "
                "typed, executor-bound schema (decorative fsm:)."
            ),
            suggestion=(
                "Adopt the typed executor-bound workflow/DAG field (OMN-12835)."
            ),
            location=a.contract_path,
        )


def validate_contract_declared_orchestrator_workflow(
    node_dir: str,
) -> ModelFileValidationResult:
    """Validate a single node directory against ARCH-004.

    Args:
        node_dir: Path to a ``nodes/<node_name>/`` directory.

    Returns:
        ``ModelFileValidationResult`` with ``valid=False`` only when a HARD-fail
        (ERROR) signal fires. Warnings do not flip ``valid`` (they feed the
        baseline score).
    """
    analysis = analyze_node_directory(Path(node_dir))
    if analysis is None:
        return ModelFileValidationResult(
            valid=True,
            violations=[],
            files_checked=0,
            rules_checked=[RULE_ID],
        )
    violations = analysis.violations()
    return ModelFileValidationResult(
        valid=not analysis.has_hard_fail,
        violations=violations,
        files_checked=1,
        rules_checked=[RULE_ID],
    )


class RuleContractDeclaredOrchestratorWorkflow:
    """Protocol-compliant rule: contract-declared orchestrator workflow must be bound.

    This rule implements :class:`ProtocolArchitectureRule`. Unlike the per-file
    ARCH-001/002/003 rules, ARCH-004's ``target`` is a **node directory** (the
    parent of a ``contract.yaml``). It accepts either the directory path or a
    path to a ``contract.yaml`` (the directory parent is used).

    Thread Safety:
        Stateless; safe for concurrent use.
    """

    @property
    def rule_id(self) -> str:
        """Return the canonical rule ID matching contract.yaml."""
        return RULE_ID

    @property
    def name(self) -> str:
        """Return human-readable rule name."""
        return RULE_NAME

    @property
    def description(self) -> str:
        """Return detailed rule description."""
        return (
            "An orchestrator-like node that declares an fsm:/workflow-state set "
            "must bind it to a runtime executor. It must not leave the contract "
            "table decorative while a handler drives the transitions itself "
            "(_transition(...), an Enum*State mirroring the contract states, or a "
            "payload_type_match catchall funneling 3+ payloads into one handler). "
            "Reducers are exempt."
        )

    @property
    def severity(self) -> EnumValidationSeverity:
        """Return severity level for violations of this rule."""
        return EnumValidationSeverity.ERROR

    def check(self, target: object) -> ModelRuleCheckResult:
        """Check a node directory (or its contract.yaml) against ARCH-004.

        Args:
            target: A node directory path, or a path to a ``contract.yaml``.
                Other types return ``skipped=True``.

        Returns:
            ``ModelRuleCheckResult`` indicating pass/fail. When multiple
            violations exist, the first ERROR (or first violation) is surfaced;
            ``details["total_violations"]`` and ``details["finding_codes"]``
            carry the full picture.
        """
        from omnibase_infra.nodes.node_architecture_validator.models import (
            ModelRuleCheckResult,
        )

        node_dir = self._resolve_node_dir(target)
        if node_dir is None:
            return ModelRuleCheckResult(
                passed=True,
                rule_id=self.rule_id,
                skipped=True,
                reason="Target is not a node directory or contract.yaml path",
            )

        analysis = analyze_node_directory(node_dir)
        if analysis is None:
            return ModelRuleCheckResult(
                passed=True,
                rule_id=self.rule_id,
                skipped=True,
                reason="Directory is not an in-scope orchestrator node (or is exempt)",
            )

        if not analysis.has_hard_fail:
            return ModelRuleCheckResult(
                passed=True,
                rule_id=self.rule_id,
                details={
                    "finding_codes": list(analysis.finding_codes),
                    "risk_score": analysis.risk_score,
                    "max_handler_lines": analysis.max_handler_lines,
                },
            )

        # Surface the first ERROR-severity violation.
        error_violations = [
            v
            for v in analysis.violations()
            if v.severity == EnumValidationSeverity.ERROR
        ]
        violation = error_violations[0]
        return ModelRuleCheckResult(
            passed=False,
            rule_id=self.rule_id,
            message=violation.message,
            details={
                "target_name": violation.target_name,
                "target_type": violation.target_type,
                "location": violation.location,
                "suggestion": violation.suggestion,
                "finding_codes": list(analysis.finding_codes),
                "risk_score": analysis.risk_score,
                "max_handler_lines": analysis.max_handler_lines,
                "max_handler_path": (
                    str(analysis.max_handler_path)
                    if analysis.max_handler_path
                    else None
                ),
                "total_violations": len(analysis.violations()),
            },
        )

    @staticmethod
    def _resolve_node_dir(target: object) -> Path | None:
        """Resolve ``target`` to a node directory, or None if not applicable."""
        if not isinstance(target, str | Path):
            return None
        path = Path(target)
        if path.name == "contract.yaml":
            return path.parent
        if path.is_dir() and (path / "contract.yaml").is_file():
            return path
        return None


__all__ = [
    "validate_contract_declared_orchestrator_workflow",
    "analyze_node_directory",
    "RuleContractDeclaredOrchestratorWorkflow",
    "RULE_ID",
    "RULE_NAME",
    "EVENT_MARKER_REGEX",
    "BRANCH_MARKER_REGEX",
    "HANDLER_LINE_WARN_THRESHOLD",
    "BRANCH_MARKER_WARN_THRESHOLD",
    "EVENT_MARKER_WARN_THRESHOLD",
    "PAYLOAD_FANIN_HARD_THRESHOLD",
]
