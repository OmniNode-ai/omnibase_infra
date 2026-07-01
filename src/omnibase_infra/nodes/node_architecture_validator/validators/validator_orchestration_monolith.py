# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Validator for ARCH-004 Signal B: orchestration-monolith complexity (OMN-13486).

Signal B is a **DISTINCT rule** from Signal A
(``validator_contract_declared_orchestrator_workflow``). Signal A catches the
acute corruption-risk triad: a *decorative* contract ``fsm:`` that a handler
re-implements imperatively via ``self._transition(...)`` (the
``node_delegation_orchestrator`` shape). Signal B catches a *different*
anti-pattern entirely: the **Class-B monolith** — a single oversized
orchestrator handler carrying high branch/event complexity and a fan-in/fan-out
routing table — *independent of whether the contract declares any FSM at all*.

Why a separate rule (not a widened Signal A):
    The audit (``docs/audits/2026-06-22-imperative-orchestrator-audit.md``)
    proves the two anti-patterns are orthogonal. build_loop (1581L), session
    (1603L), memory_lifecycle (1237L), swarm_dispatch (890L) and redeploy (422L)
    all carry **0** handler-owned ``_transition`` calls — Signal A's hard-fails
    (H1/H2/H3) never fire for them. They are debt of a different shape:
    operation_match routing funnelling many operations into one giant handler.
    Folding this into Signal A would (a) blur the "decorative FSM" semantics that
    make Signal A precise and (b) commingle two distinct decomposition tracks.
    Signal B therefore owns its own finding-code namespace (``B*``) and its own
    ratchet baseline dimension.

Class-B definition (a monolith hard-fail, ``B1``):
    An orchestrator-like node whose largest handler is **oversized**
    (> ``MONOLITH_HANDLER_LINE_THRESHOLD`` lines) AND **complex** (at least one
    of: branch markers > ``MONOLITH_BRANCH_THRESHOLD``, event-construction
    markers >= ``MONOLITH_EVENT_THRESHOLD``, routing fan-in/out >=
    ``MONOLITH_FANOUT_THRESHOLD``), AND **not executor-bound** (the contract does
    not delegate coordination to a typed execution graph / workflow DAG / state
    machine). An executor-bound orchestrator is exempt: even a large handler is
    acceptable when the coordination has been handed to the executor.

Thin / executor-bound orchestrators PASS. Reducers and non-orchestrator nodes
are out of scope (return ``None`` from :func:`analyze_monolith`).

Related:
    - Ticket: OMN-13486 (ARCH-004 Signal B — orchestration-monolith complexity)
    - Epic: OMN-13485 (sibling decomposition epic)
    - Signal A: OMN-13472 (imperative-orchestrator ratchet)
    - OMN-12550 (validator gating) / OMN-13325 (ratchet enforcement)
    - OMN-12835 (typed executor-bound workflow field) — the remediation target
"""

from __future__ import annotations

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

# Signal B reuses the precise, boundary-anchored markers and the
# executor-bound / orchestrator / reducer classification helpers from the
# Signal A analyzer so the two rules agree on the underlying facts and only
# diverge on what they *signal* about them.
from omnibase_infra.nodes.node_architecture_validator.validators.validator_contract_declared_orchestrator_workflow import (
    BRANCH_MARKER_REGEX,
    EVENT_MARKER_REGEX,
    _contract_has_executor_bound_workflow,
    _is_orchestrator_like,
    _is_reducer,
    _largest_handler,
)

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_architecture_validator.models import (
        ModelRuleCheckResult,
    )

#: Signal B rule id — DISTINCT from Signal A's ``ARCH-004``.
MONOLITH_RULE_ID = "ARCH-004B"
MONOLITH_RULE_NAME = "Orchestrator Handler Must Not Be A Class-B Monolith"

# --- Detection thresholds --------------------------------------------------- #

#: A handler larger than this for an orchestrator-like node is monolith-sized.
MONOLITH_HANDLER_LINE_THRESHOLD = 750
#: Branch/control density above this contributes the complexity signal.
MONOLITH_BRANCH_THRESHOLD = 50
#: Event-construction marker count at/above this contributes the signal.
MONOLITH_EVENT_THRESHOLD = 20
#: Routing fan-in/fan-out at/above this contributes the signal.
MONOLITH_FANOUT_THRESHOLD = 5


# --------------------------------------------------------------------------- #
# Routing fan-in / fan-out
# --------------------------------------------------------------------------- #
def _count_routing_entries(handler_routing: dict[str, object]) -> tuple[int, int]:
    """Return ``(fan_in, fan_out)`` for a handler_routing table.

    * ``fan_in`` — distinct event/payload OR operation keys routed (how many
      inbound shapes the node accepts).
    * ``fan_out`` — distinct downstream handler class names (how many handler
      surfaces the node fans into). A single catchall handler => fan_out 1 even
      with many fan-in keys (that is itself the monolith smell).
    """
    handlers = handler_routing.get("handlers")
    if not isinstance(handlers, list):
        return 0, 0
    in_keys: set[str] = set()
    out_handlers: set[str] = set()
    for entry in handlers:
        if not isinstance(entry, dict):
            continue
        event_model = entry.get("event_model")
        if isinstance(event_model, dict):
            name = event_model.get("name")
            if isinstance(name, str):
                in_keys.add(f"event:{name}")
        elif isinstance(event_model, str):
            in_keys.add(f"event:{event_model}")
        op = entry.get("operation")
        if isinstance(op, str):
            in_keys.add(f"op:{op}")
        handler = entry.get("handler")
        if isinstance(handler, dict):
            hname = handler.get("name")
            if isinstance(hname, str):
                out_handlers.add(hname)
    return len(in_keys), len(out_handlers)


# --------------------------------------------------------------------------- #
# Analysis result
# --------------------------------------------------------------------------- #
class MonolithNodeAnalysis:
    """Cross-file analysis of one node directory for Signal B."""

    def __init__(self, node_dir: Path, contract_path: Path) -> None:
        self.node_dir = node_dir
        self.contract_path = contract_path
        self.node_name: str = node_dir.name
        self.node_type: str = ""
        self.routing_strategy: str = ""
        self.has_executor_bound_workflow: bool = False
        self.max_handler_path: Path | None = None
        self.max_handler_lines: int = 0
        self.branch_markers: int = 0
        self.event_markers: int = 0
        self.fan_in: int = 0
        self.fan_out: int = 0
        self.finding_codes: list[str] = []
        self._violations: list[ModelArchitectureViolation] = []

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
                rule_id=MONOLITH_RULE_ID,
                rule_name=MONOLITH_RULE_NAME,
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
                    "max_handler_lines": str(self.max_handler_lines),
                    "branch_markers": str(self.branch_markers),
                    "event_markers": str(self.event_markers),
                    "fan_in": str(self.fan_in),
                    "fan_out": str(self.fan_out),
                    "branch_marker_regex": BRANCH_MARKER_REGEX,
                    "event_marker_regex": EVENT_MARKER_REGEX,
                },
            )
        )

    def violations(self) -> list[ModelArchitectureViolation]:
        return self._violations

    @property
    def has_hard_fail(self) -> bool:
        return any(v.severity == EnumValidationSeverity.ERROR for v in self._violations)

    @property
    def is_complex(self) -> bool:
        """True if any one complexity dimension crosses its threshold."""
        return (
            self.branch_markers > MONOLITH_BRANCH_THRESHOLD
            or self.event_markers >= MONOLITH_EVENT_THRESHOLD
            or self.fan_in >= MONOLITH_FANOUT_THRESHOLD
            or self.fan_out >= MONOLITH_FANOUT_THRESHOLD
        )

    @property
    def complexity_score(self) -> int:
        """Transparent additive Signal-B complexity score (independent of FSM).

        ``handler>1500:+2 (>750:+1) · branch>threshold:+1 ·
        event>=threshold:+1 · fan_in>=threshold:+1 · fan_out>=threshold:+1``.
        """
        score = 0
        if self.max_handler_lines > 1500:
            score += 2
        elif self.max_handler_lines > MONOLITH_HANDLER_LINE_THRESHOLD:
            score += 1
        if self.branch_markers > MONOLITH_BRANCH_THRESHOLD:
            score += 1
        if self.event_markers >= MONOLITH_EVENT_THRESHOLD:
            score += 1
        if self.fan_in >= MONOLITH_FANOUT_THRESHOLD:
            score += 1
        if self.fan_out >= MONOLITH_FANOUT_THRESHOLD:
            score += 1
        return score


# --------------------------------------------------------------------------- #
# Core analysis
# --------------------------------------------------------------------------- #
def analyze_monolith(node_dir: Path) -> MonolithNodeAnalysis | None:
    """Run the Signal B cross-file join over a single node directory.

    Returns a populated :class:`MonolithNodeAnalysis` (with any violations), or
    ``None`` when the directory is not an analyzable orchestrator node (no
    contract, a reducer, or not orchestrator-like).
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

    node_type = str(raw.get("node_type", ""))
    node_name = str(raw.get("name", node_path.name))

    # Reducers legitimately own state; non-orchestrator nodes are out of scope.
    if _is_reducer(node_name, node_type):
        return None
    if not _is_orchestrator_like(node_name, node_type):
        return None

    analysis = MonolithNodeAnalysis(node_path, contract_path)
    analysis.node_type = node_type
    analysis.node_name = node_name
    analysis.has_executor_bound_workflow = _contract_has_executor_bound_workflow(raw)

    handler_routing = raw.get("handler_routing")
    if isinstance(handler_routing, dict):
        analysis.routing_strategy = str(handler_routing.get("routing_strategy", ""))
        analysis.fan_in, analysis.fan_out = _count_routing_entries(handler_routing)

    handler_path, handler_lines, handler_source = _largest_handler(
        node_path / "handlers"
    )
    analysis.max_handler_path = handler_path
    analysis.max_handler_lines = handler_lines
    if handler_source:
        analysis.branch_markers = len(re.findall(BRANCH_MARKER_REGEX, handler_source))
        analysis.event_markers = len(re.findall(EVENT_MARKER_REGEX, handler_source))

    _evaluate_monolith_signals(analysis)
    return analysis


def _evaluate_monolith_signals(a: MonolithNodeAnalysis) -> None:
    """Apply Signal B's hard-fail (B1) and warnings (B2)."""
    handler_loc = a.max_handler_path or a.contract_path
    oversized = a.max_handler_lines > MONOLITH_HANDLER_LINE_THRESHOLD

    # B1: Class-B monolith hard-fail — oversized AND complex AND not
    # executor-bound. Executor-bound orchestrators are exempt (coordination is
    # delegated to the executor; the large handler is managed debt, not a
    # corruption-risk monolith).
    if oversized and a.is_complex and not a.has_executor_bound_workflow:
        a._add(
            code="B1",
            severity=EnumValidationSeverity.ERROR,
            message=(
                f"Class-B orchestration monolith: largest handler is "
                f"{a.max_handler_lines} lines "
                f"(> {MONOLITH_HANDLER_LINE_THRESHOLD}) with high complexity "
                f"(branch={a.branch_markers}, event={a.event_markers}, "
                f"fan_in={a.fan_in}, fan_out={a.fan_out}) and no executor-bound "
                f"workflow. A single handler coordinates the whole workflow "
                f"imperatively (independent of FSM decorativeness — this is "
                f"distinct from the Signal A decorative-FSM/_transition triad)."
            ),
            suggestion=(
                "Decompose into per-step thin handlers and bind coordination to "
                "a typed executor-bound workflow/DAG (OMN-12835), or split the "
                "operation_match catchall into per-operation handler modules. "
                "See the node's owner decomposition ticket."
            ),
            location=handler_loc,
        )

    # B2: oversized-but-not-yet-monolith warning (oversized handler that is
    # executor-bound, or oversized without the complexity signal). Feeds the
    # baseline score; never blocks on its own.
    elif oversized:
        a._add(
            code="B2",
            severity=EnumValidationSeverity.WARNING,
            message=(
                f"Oversized orchestrator handler ({a.max_handler_lines} lines > "
                f"{MONOLITH_HANDLER_LINE_THRESHOLD}) but not (yet) a Class-B "
                f"monolith hard-fail "
                f"(executor_bound={a.has_executor_bound_workflow}, "
                f"complex={a.is_complex}). Debt, not corruption risk."
            ),
            suggestion="Keep orchestrator handlers thin; decompose per step.",
            location=handler_loc,
        )


def validate_orchestration_monolith(node_dir: str) -> ModelFileValidationResult:
    """Validate a single node directory against Signal B.

    ``valid=False`` only when a B1 (ERROR) hard-fail fires. Warnings (B2) do not
    flip ``valid``.
    """
    analysis = analyze_monolith(Path(node_dir))
    if analysis is None:
        return ModelFileValidationResult(
            valid=True,
            violations=[],
            files_checked=0,
            rules_checked=[MONOLITH_RULE_ID],
        )
    return ModelFileValidationResult(
        valid=not analysis.has_hard_fail,
        violations=analysis.violations(),
        files_checked=1,
        rules_checked=[MONOLITH_RULE_ID],
    )


class RuleOrchestrationMonolith:
    """Protocol-compliant rule: orchestrator handler must not be a Class-B monolith.

    Implements :class:`ProtocolArchitectureRule`. The ``target`` is a node
    directory (the parent of a ``contract.yaml``) or a path to a ``contract.yaml``.

    Thread Safety:
        Stateless; safe for concurrent use.
    """

    @property
    def rule_id(self) -> str:
        return MONOLITH_RULE_ID

    @property
    def name(self) -> str:
        return MONOLITH_RULE_NAME

    @property
    def description(self) -> str:
        return (
            "An orchestrator-like node must not concentrate its whole workflow in "
            "a single oversized, complex handler. A handler that is both oversized "
            f"(> {MONOLITH_HANDLER_LINE_THRESHOLD} lines) and complex (high "
            "branch/event density or routing fan-in/out), without binding "
            "coordination to a typed executor, is a Class-B monolith. This is a "
            "DISTINCT signal from ARCH-004 Signal A (decorative FSM + "
            "handler-driven _transition): a monolith fails here even with no FSM "
            "at all. Reducers are exempt."
        )

    @property
    def severity(self) -> EnumValidationSeverity:
        return EnumValidationSeverity.ERROR

    def check(self, target: object) -> ModelRuleCheckResult:
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

        analysis = analyze_monolith(node_dir)
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
                    "complexity_score": analysis.complexity_score,
                    "max_handler_lines": analysis.max_handler_lines,
                },
            )

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
                "complexity_score": analysis.complexity_score,
                "max_handler_lines": analysis.max_handler_lines,
                "fan_in": analysis.fan_in,
                "fan_out": analysis.fan_out,
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
        if not isinstance(target, str | Path):
            return None
        path = Path(target)
        if path.name == "contract.yaml":
            return path.parent
        if path.is_dir() and (path / "contract.yaml").is_file():
            return path
        return None


__all__ = [
    "validate_orchestration_monolith",
    "analyze_monolith",
    "RuleOrchestrationMonolith",
    "MonolithNodeAnalysis",
    "MONOLITH_RULE_ID",
    "MONOLITH_RULE_NAME",
    "MONOLITH_HANDLER_LINE_THRESHOLD",
    "MONOLITH_BRANCH_THRESHOLD",
    "MONOLITH_EVENT_THRESHOLD",
    "MONOLITH_FANOUT_THRESHOLD",
]
